#include "tdenv.h"
#include "tdhost.h"

#include <map>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

namespace {

// The agent is always player 0.
const uint64_t AGENT = tdhost::GameLib::Player_ID(0);
const int CELL_PIXELS = 24;
// Grid cell at input pixel (0, 0).
const int INPUT_ORIGIN = 1;
// Cell numbers in the megamap build are y * 128 + x.
const int MAP_WIDTH = 128;
// Houses below the first multiplayer house (GDI, Nod, civilians, Jurassic) only own
// neutral things in a skirmish.
const int FIRST_PLAYER_HOUSE = 4;

std::string Error;
std::string LibCopy;
std::string DataDir;
// Deliberately never destroyed. At process exit the game library's static destructors run
// first (it was loaded after this library), so unloading it from a destructor here would
// free the game a second time; exiting frees everything anyway.
tdhost::GameLib& Game = *new tdhost::GameLib;
bool Started = false;
bool GameOver = false;
bool Rendering = false;
std::vector<unsigned char> Page;
int Frame = 0;

// Playable grid, in absolute map cells.
int GridX = 0;
int GridY = 0;
int GridW = 0;
int GridH = 0;

// State captured by tdenv_observe.
int House = -1;
unsigned AllyFlags = 0;
bool Defeated = false;
TDScalars Scalars;
std::vector<unsigned char> Mapped;
std::vector<unsigned char> Tiberium;
std::vector<TDObject> Objects;
std::vector<TDBuildable> Buildables;
std::vector<CNCSidebarEntryStruct> SidebarEntries;

std::vector<unsigned char> Buffer(16 * 1024 * 1024);

int Fail(const std::string& message)
{
    Error = message;
    return TDENV_ERROR;
}

void __cdecl Event_Callback(const EventCallbackStruct& event)
{
    if (event.EventType == CALLBACK_EVENT_GAME_OVER) {
        GameOver = true;
    }
}

template <typename T> T* Get_State(GameStateRequestEnum request)
{
    if (!Game.Get_Game_State(request, AGENT, Buffer.data(), (unsigned)Buffer.size())) {
        return nullptr;
    }
    return reinterpret_cast<T*>(Buffer.data());
}

bool In_Grid(int x, int y)
{
    return x >= 0 && y >= 0 && x < GridW && y < GridH;
}

bool Is_Mapped(int abs_x, int abs_y)
{
    int x = abs_x - GridX;
    int y = abs_y - GridY;
    return In_Grid(x, y) && Mapped[y * GridW + x] != 0;
}

void Copy_Name(char* out, const char* in)
{
    strncpy(out, in, TDENV_NAME_LENGTH - 1);
    out[TDENV_NAME_LENGTH - 1] = '\0';
}

int Relation(int owner)
{
    if (owner == House) {
        return TDENV_SELF;
    }
    if (owner < FIRST_PLAYER_HOUSE || owner >= 32) {
        return TDENV_NEUTRAL;
    }
    return (AllyFlags & (1u << owner)) ? TDENV_ALLY : TDENV_ENEMY;
}

/*
** Human-level visibility: the agent sees its own objects, and anything else with at
** least one cell the agent has explored.
*/
void Decode_Offset(int offset, int& dx, int& dy)
{
    dy = (offset + MAP_WIDTH / 2) / MAP_WIDTH;
    dx = offset - dy * MAP_WIDTH;
}

// The grid cells each building occupies, keyed by building id, as a bounding box
// (x0, y0, x1, y1) inclusive. Exported object cells are where an object's center is,
// which for a building is somewhere inside its footprint.
std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> BuildingCells;

void Observe_Occupiers()
{
    BuildingCells.clear();
    CNCOccupierHeaderStruct* header = Get_State<CNCOccupierHeaderStruct>(GAME_STATE_OCCUPIER);
    if (header == nullptr || header->Count != GridW * GridH) {
        return;
    }
    const unsigned char* p = reinterpret_cast<const unsigned char*>(header + 1);
    for (int cell = 0; cell < header->Count; ++cell) {
        const CNCOccupierEntryHeaderStruct* entry = reinterpret_cast<const CNCOccupierEntryHeaderStruct*>(p);
        const CNCOccupierObjectStruct* occupiers = reinterpret_cast<const CNCOccupierObjectStruct*>(entry + 1);
        int x = cell % GridW;
        int y = cell / GridW;
        for (int i = 0; i < entry->Count; ++i) {
            if (occupiers[i].Type != BUILDING) {
                continue;
            }
            auto found = BuildingCells.find(occupiers[i].ID);
            if (found == BuildingCells.end()) {
                BuildingCells[occupiers[i].ID] = {{x, y}, {x, y}};
            } else {
                auto& box = found->second;
                box.first.first = x < box.first.first ? x : box.first.first;
                box.first.second = y < box.first.second ? y : box.first.second;
                box.second.first = x > box.second.first ? x : box.second.first;
                box.second.second = y > box.second.second ? y : box.second.second;
            }
        }
        // The export leaves one unused slot after each cell's occupiers.
        p = reinterpret_cast<const unsigned char*>(occupiers + entry->Count + 1);
    }
}

bool Is_Mapped_Box(int x0, int y0, int x1, int y1)
{
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            if (Is_Mapped(x + GridX, y + GridY)) {
                return true;
            }
        }
    }
    return false;
}

bool Observe_Player()
{
    CNCPlayerInfoStruct* info = Get_State<CNCPlayerInfoStruct>(GAME_STATE_PLAYER_INFO);
    if (info == nullptr) {
        Error = "player info unavailable";
        return false;
    }
    House = info->House;
    AllyFlags = info->AllyFlags;
    Defeated = info->IsDefeated;
    return true;
}

bool Observe_Shroud()
{
    CNCShroudStruct* shroud = Get_State<CNCShroudStruct>(GAME_STATE_SHROUD);
    if (shroud == nullptr || shroud->Count != GridW * GridH) {
        Error = "shroud does not match the playable grid";
        return false;
    }
    for (int i = 0; i < shroud->Count; ++i) {
        Mapped[i] = shroud->Entries[i].IsMapped ? 1 : 0;
    }
    return true;
}

void Observe_Objects()
{
    Objects.clear();
    CNCObjectListStruct* list = Get_State<CNCObjectListStruct>(GAME_STATE_LAYERS);
    if (list == nullptr) {
        return; // Nothing to draw.
    }
    for (int i = 0; i < list->Count; ++i) {
        const CNCObjectStruct& obj = list->Objects[i];
        if (obj.SubObject || obj.Type < INFANTRY || obj.Type > TERRAIN) {
            continue;
        }
        // Owner is a plain char, signed on some platforms and unsigned on others.
        int owner = (unsigned char)obj.Owner;
        int relation = Relation(owner);
        int x0 = obj.CellX - GridX, y0 = obj.CellY - GridY, x1 = x0, y1 = y0;
        if (obj.Type == BUILDING) {
            auto found = BuildingCells.find(obj.ID);
            if (found != BuildingCells.end()) {
                x0 = found->second.first.first;
                y0 = found->second.first.second;
                x1 = found->second.second.first;
                y1 = found->second.second.second;
            }
        }
        // Human-level visibility: the agent sees its own objects, and anything else with at
        // least one cell the agent has explored.
        if (relation != TDENV_SELF && !Is_Mapped_Box(x0, y0, x1, y1)) {
            continue;
        }
        TDObject out = {};
        out.id = obj.ID;
        out.type = obj.Type;
        Copy_Name(out.name, obj.TypeName);
        out.relation = relation;
        out.house = owner;
        out.cell_x = x0;
        out.cell_y = y0;
        out.size_x = x1 - x0 + 1;
        out.size_y = y1 - y0 + 1;
        out.pixel_x = obj.PositionX + obj.Width / 2;
        out.pixel_y = obj.PositionY + obj.Height / 2;
        out.strength = obj.Strength;
        out.max_strength = obj.MaxStrength;
        out.can_move = House >= 0 && House < MAX_HOUSES && obj.CanMove[House];
        // The export leaves CanDeploy unset in Tiberian Dawn, where only the MCV deploys.
        out.can_deploy = obj.Type == UNIT && strcmp(out.name, "MCV") == 0;
        out.can_harvest = obj.CanHarvest;
        out.is_factory = obj.IsFactory;
        out.is_selectable = obj.IsSelectable;
        out.pips = obj.NumPips;
        out.max_pips = obj.MaxPips;
        Objects.push_back(out);
    }
}

void Observe_Tiberium()
{
    std::fill(Tiberium.begin(), Tiberium.end(), 0);
    CNCDynamicMapStruct* map = Get_State<CNCDynamicMapStruct>(GAME_STATE_DYNAMIC_MAP);
    if (map == nullptr) {
        return;
    }
    for (int i = 0; i < map->Count; ++i) {
        const CNCDynamicMapEntryStruct& entry = map->Entries[i];
        int x = entry.CellX - GridX;
        int y = entry.CellY - GridY;
        // Tiberium overlays are TI1 to TI12 by density.
        if (!entry.IsResource || !In_Grid(x, y) || !Mapped[y * GridW + x]
            || strncmp(entry.AssetName, "TI", 2) != 0) {
            continue;
        }
        int density = atoi(entry.AssetName + 2);
        Tiberium[y * GridW + x] = (unsigned char)(density > 0 ? density : 1);
    }
}

void Observe_Sidebar()
{
    Buildables.clear();
    SidebarEntries.clear();
    CNCSidebarStruct* sidebar = Get_State<CNCSidebarStruct>(GAME_STATE_SIDEBAR);
    if (sidebar == nullptr) {
        return;
    }
    Scalars.credits = sidebar->Credits;
    Scalars.tiberium = sidebar->Tiberium;
    Scalars.max_tiberium = sidebar->MaxTiberium;
    Scalars.power_produced = sidebar->PowerProduced;
    Scalars.power_drained = sidebar->PowerDrained;
    Scalars.units_killed = sidebar->UnitsKilled;
    Scalars.buildings_killed = sidebar->BuildingsKilled;
    Scalars.units_lost = sidebar->UnitsLost;
    Scalars.buildings_lost = sidebar->BuildingsLost;
    Scalars.harvested_credits = sidebar->TotalHarvestedCredits;

    int count = sidebar->EntryCount[0] + sidebar->EntryCount[1];
    for (int i = 0; i < count; ++i) {
        const CNCSidebarEntryStruct& entry = sidebar->Entries[i];
        TDBuildable out = {};
        Copy_Name(out.name, entry.AssetName);
        // INFANTRY_TYPE and so on are 11 above the matching object types.
        out.type = (entry.Type >= INFANTRY_TYPE && entry.Type <= BUILDING_TYPE) ? entry.Type - 11 : 0;
        out.cost = entry.Cost;
        out.build_time = entry.BuildTime;
        out.progress = entry.Progress;
        out.completed = entry.Completed;
        out.constructing = entry.Constructing;
        out.on_hold = entry.ConstructionOnHold;
        out.busy = entry.Busy;
        Buildables.push_back(out);
        SidebarEntries.push_back(entry);
    }
}

const CNCSidebarEntryStruct* Find_Buildable(const char* name, const TDBuildable** buildable = nullptr)
{
    for (size_t i = 0; i < Buildables.size(); ++i) {
        if (strncmp(Buildables[i].name, name, TDENV_NAME_LENGTH) == 0) {
            if (buildable != nullptr) {
                *buildable = &Buildables[i];
            }
            return &SidebarEntries[i];
        }
    }
    return nullptr;
}

const TDObject* Find_Own_Object(int type, int id)
{
    for (const TDObject& obj : Objects) {
        if (obj.relation == TDENV_SELF && obj.type == type && obj.id == id) {
            return &obj;
        }
    }
    return nullptr;
}

int Select(const int* types, const int* ids, int count)
{
    Game.Clear_Object_Selection(AGENT);
    int selected = 0;
    for (int i = 0; i < count; ++i) {
        const TDObject* obj = Find_Own_Object(types[i], ids[i]);
        if (obj != nullptr && obj->is_selectable && Game.Select_Object(AGENT, obj->type, obj->id)) {
            ++selected;
        }
    }
    return selected;
}

} // namespace

extern "C" {

const char* tdenv_last_error(void)
{
    return Error.c_str();
}

int tdenv_open(const char* lib, const char* data, const char* disc, const char* work)
{
    if (!tdhost::Prepare_Work_Dir(lib, data, disc ? disc : "gdi", work, LibCopy, DataDir, Error)) {
        return TDENV_ERROR;
    }
    return 0;
}

int tdenv_reset(int map, int num_ais, unsigned seed, int agent_side, int credits, int ai_difficulty)
{
    if (LibCopy.empty()) {
        return Fail("tdenv_open has not succeeded");
    }
    Started = false;
    // Restarting a scenario in place leaks state into the next game; reloading doesn't.
    if (!Game.Load(LibCopy, Event_Callback, Error)) {
        return TDENV_ERROR;
    }
    Game.Set_Headless(!Rendering);
    tdhost::SkirmishSettings settings;
    settings.Map = map;
    settings.Credits = credits;
    settings.Seed = seed;
    settings.AIDifficulty = ai_difficulty;
    settings.Players.push_back({false, agent_side});
    for (int i = 0; i < num_ais; ++i) {
        settings.Players.push_back({true, (agent_side + 1 + i) % 2});
    }
    GameOver = false;
    Frame = 0;
    if (!Game.Start_Skirmish(settings, DataDir, Error)) {
        return TDENV_ERROR;
    }

    CNCMapDataStruct* static_map = Get_State<CNCMapDataStruct>(GAME_STATE_STATIC_MAP);
    if (static_map == nullptr) {
        return Fail("static map unavailable");
    }
    GridX = static_map->MapCellX;
    GridY = static_map->MapCellY;
    GridW = static_map->MapCellWidth;
    GridH = static_map->MapCellHeight;
    Mapped.assign(GridW * GridH, 0);
    Tiberium.assign(GridW * GridH, 0);
    Started = true;
    // Clicks only find objects in visible cells, and nothing is visible until the first
    // frame has run, so commands issued before it are ignored.
    Game.Advance_Instance(AGENT);
    Frame = 1;
    return tdenv_observe();
}

int tdenv_grid_size(int* width, int* height)
{
    if (!Started) {
        return Fail("no game in progress");
    }
    *width = GridW;
    *height = GridH;
    return 0;
}

int tdenv_set_rendering(int enabled)
{
    Rendering = enabled != 0;
    if (Game.Is_Loaded()) {
        Game.Set_Headless(!Rendering);
    }
    return 0;
}

int tdenv_frame(unsigned char* rgb, int max_bytes, int* width, int* height)
{
    if (!Started || !Rendering) {
        return Fail("tdenv_frame needs a game in progress with rendering on");
    }
    unsigned w = 0, h = 0;
    Page.resize(GridW * GridH * CELL_PIXELS * CELL_PIXELS);
    if (!Game.Get_Visible_Page(Page.data(), w, h) || (size_t)w * h > Page.size()) {
        return Fail("no frame available");
    }
    if ((int)(w * h * 3) > max_bytes) {
        return Fail("frame buffer too small");
    }
    unsigned char palette[256][3];
    Game.Get_Palette(palette);
    // The palette is 6 bits per channel, as on VGA.
    for (unsigned i = 0; i < w * h; ++i) {
        const unsigned char* c = palette[Page[i]];
        rgb[3 * i] = (unsigned char)(c[0] << 2 | c[0] >> 4);
        rgb[3 * i + 1] = (unsigned char)(c[1] << 2 | c[1] >> 4);
        rgb[3 * i + 2] = (unsigned char)(c[2] << 2 | c[2] >> 4);
    }
    *width = (int)w;
    *height = (int)h;
    return 0;
}

int tdenv_step(int frames)
{
    if (!Started) {
        return Fail("no game in progress");
    }
    bool running = true;
    for (int i = 0; i < frames && running && !GameOver; ++i) {
        running = Game.Advance_Instance(AGENT);
        ++Frame;
    }
    if (!Observe_Player()) {
        return TDENV_ERROR;
    }
    if (Defeated) {
        return TDENV_LOST;
    }
    return (running && !GameOver) ? TDENV_RUNNING : TDENV_WON;
}

int tdenv_observe(void)
{
    if (!Started) {
        return Fail("no game in progress");
    }
    memset(&Scalars, 0, sizeof(Scalars));
    if (!Observe_Player() || !Observe_Shroud()) {
        return TDENV_ERROR;
    }
    Observe_Occupiers();
    Observe_Objects();
    Observe_Tiberium();
    Observe_Sidebar();
    Scalars.frame = Frame;
    Scalars.defeated = Defeated;
    return 0;
}

int tdenv_scalars(TDScalars* out)
{
    *out = Scalars;
    return 0;
}

int tdenv_objects(TDObject* out, int max)
{
    int count = (int)Objects.size() < max ? (int)Objects.size() : max;
    memcpy(out, Objects.data(), count * sizeof(TDObject));
    return count;
}

int tdenv_shroud(unsigned char* mapped)
{
    memcpy(mapped, Mapped.data(), Mapped.size());
    return 0;
}

int tdenv_tiberium(unsigned char* density)
{
    memcpy(density, Tiberium.data(), Tiberium.size());
    return 0;
}

int tdenv_buildables(TDBuildable* out, int max)
{
    int count = (int)Buildables.size() < max ? (int)Buildables.size() : max;
    memcpy(out, Buildables.data(), count * sizeof(TDBuildable));
    return count;
}

int tdenv_build(const char* name)
{
    const TDBuildable* buildable;
    const CNCSidebarEntryStruct* entry = Find_Buildable(name, &buildable);
    if (entry == nullptr || buildable->busy || buildable->constructing || buildable->completed) {
        return 0;
    }
    Game.Handle_Sidebar_Request(
        SIDEBAR_REQUEST_START_CONSTRUCTION, AGENT, entry->BuildableType, entry->BuildableID, 0, 0);
    return 1;
}

int tdenv_cancel(const char* name)
{
    const TDBuildable* buildable;
    const CNCSidebarEntryStruct* entry = Find_Buildable(name, &buildable);
    if (entry == nullptr || !(buildable->constructing || buildable->completed || buildable->on_hold)) {
        return 0;
    }
    Game.Handle_Sidebar_Request(
        SIDEBAR_REQUEST_CANCEL_CONSTRUCTION, AGENT, entry->BuildableType, entry->BuildableID, 0, 0);
    return 1;
}

int tdenv_place(const char* name, int x, int y)
{
    const TDBuildable* buildable;
    const CNCSidebarEntryStruct* entry = Find_Buildable(name, &buildable);
    if (entry == nullptr || !buildable->completed || buildable->type != TDENV_BUILDING || !In_Grid(x, y)) {
        return 0;
    }
    Game.Handle_Sidebar_Request(SIDEBAR_REQUEST_START_PLACEMENT, AGENT, entry->BuildableType, entry->BuildableID, 0, 0);
    Game.Handle_Sidebar_Request(
        SIDEBAR_REQUEST_PLACE, AGENT, entry->BuildableType, entry->BuildableID, (short)x, (short)y);
    return 1;
}

int tdenv_placement(const char* name, unsigned char* valid)
{
    memset(valid, 0, GridW * GridH);
    const TDBuildable* buildable;
    const CNCSidebarEntryStruct* entry = Find_Buildable(name, &buildable);
    if (entry == nullptr || !buildable->completed || buildable->type != TDENV_BUILDING) {
        return 0;
    }
    // The placement state describes the building being placed, as a human would see it.
    Game.Handle_Sidebar_Request(SIDEBAR_REQUEST_START_PLACEMENT, AGENT, entry->BuildableType, entry->BuildableID, 0, 0);
    CNCPlacementInfoStruct* info = Get_State<CNCPlacementInfoStruct>(GAME_STATE_PLACEMENT);
    if (info == nullptr || info->Count != GridW * GridH) {
        return 0;
    }
    // A cell can take the building's top left corner if that passes the proximity check
    // and every cell of the footprint is clear.
    int count = 0;
    for (int y = 0; y < GridH; ++y) {
        for (int x = 0; x < GridW; ++x) {
            if (!info->CellInfo[y * GridW + x].PassesProximityCheck) {
                continue;
            }
            bool clear = true;
            for (int c = 0; c < entry->PlacementListLength && clear; ++c) {
                int dx, dy;
                Decode_Offset(entry->PlacementList[c], dx, dy);
                clear = In_Grid(x + dx, y + dy) && info->CellInfo[(y + dy) * GridW + x + dx].GenerallyClear;
            }
            if (clear) {
                valid[y * GridW + x] = 1;
                ++count;
            }
        }
    }
    return count;
}

int tdenv_command(const int* types, const int* ids, int count, int x, int y)
{
    if (!In_Grid(x, y) || Select(types, ids, count) == 0) {
        return 0;
    }
    // Input pixels count from the map's playable area, which excludes the one cell border
    // the grid includes.
    int px = (x - INPUT_ORIGIN) * CELL_PIXELS + CELL_PIXELS / 2;
    int py = (y - INPUT_ORIGIN) * CELL_PIXELS + CELL_PIXELS / 2;
    Game.Handle_Input(INPUT_REQUEST_COMMAND_AT_POSITION, 0, AGENT, px, py, 0, 0);
    Game.Clear_Object_Selection(AGENT);
    return 1;
}

int tdenv_stop(const int* types, const int* ids, int count)
{
    if (Select(types, ids, count) == 0) {
        return 0;
    }
    Game.Handle_Unit_Request(INPUT_UNIT_STOP, AGENT);
    Game.Clear_Object_Selection(AGENT);
    return 1;
}

int tdenv_hunt(const int* types, const int* ids, int count)
{
    if (Select(types, ids, count) == 0) {
        return 0;
    }
    Game.Selected_Hunt(AGENT);
    Game.Clear_Object_Selection(AGENT);
    return 1;
}

int tdenv_sell(int building_id)
{
    if (Find_Own_Object(TDENV_BUILDING, building_id) == nullptr) {
        return 0;
    }
    Game.Handle_Structure_Request(INPUT_STRUCTURE_SELL, AGENT, building_id);
    return 1;
}

} // extern "C"

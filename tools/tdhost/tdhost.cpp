#include "tdhost.h"

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tdhost {

std::string Absolute(const std::string& path)
{
    char buf[PATH_MAX];
    return realpath(path.c_str(), buf) ? std::string(buf) : path;
}

static std::string Errno_Message(const std::string& what)
{
    return what + ": " + strerror(errno);
}

static bool Copy_File(const std::string& from, const std::string& to, std::string& error)
{
    FILE* in = fopen(from.c_str(), "rb");
    if (in == nullptr) {
        error = Errno_Message(from);
        return false;
    }
    FILE* out = fopen(to.c_str(), "wb");
    if (out == nullptr) {
        error = Errno_Message(to);
        fclose(in);
        return false;
    }
    char buf[1 << 16];
    size_t n;
    bool ok = true;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, n, out) != n) {
            ok = false;
            break;
        }
    }
    ok = ok && !ferror(in);
    fclose(in);
    ok = (fclose(out) == 0) && ok;
    if (!ok) {
        error = "failed copying " + from + " to " + to;
    }
    return ok;
}

/*
** Symlink every .MIX file in 'from' into 'to'. Returns how many were linked.
*/
static int Link_Mix_Files(const std::string& from, const std::string& to)
{
    DIR* dir = opendir(from.c_str());
    if (dir == nullptr) {
        return 0;
    }
    int count = 0;
    while (struct dirent* ent = readdir(dir)) {
        std::string name = ent->d_name;
        if (name.size() < 4 || strcasecmp(name.c_str() + name.size() - 4, ".MIX") != 0) {
            continue;
        }
        std::string link = to + "/" + name;
        unlink(link.c_str());
        if (symlink((from + "/" + name).c_str(), link.c_str()) == 0) {
            ++count;
        }
    }
    closedir(dir);
    return count;
}

static bool Prepare_Data_Dir(const std::string& data_in,
                             const std::string& disc_name,
                             const std::string& work,
                             std::string& data_dir,
                             std::string& error)
{
    std::string data = Absolute(data_in);
    std::string disc = data + "/" + disc_name;
    struct stat st;
    if (stat(disc.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        data_dir = data; // Already flat.
        return true;
    }
    data_dir = work + "/data";
    if (mkdir(data_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        error = Errno_Message(data_dir);
        return false;
    }
    int shared = Link_Mix_Files(data, data_dir);
    int disc_files = Link_Mix_Files(disc, data_dir);
    if (shared == 0 || disc_files == 0) {
        error = "no .MIX files found in " + data + " or " + disc;
        return false;
    }
    return true;
}

bool Prepare_Work_Dir(const std::string& lib,
                      const std::string& data,
                      const std::string& disc,
                      const std::string& work_in,
                      std::string& lib_copy,
                      std::string& data_dir,
                      std::string& error)
{
    if (mkdir(work_in.c_str(), 0755) != 0 && errno != EEXIST) {
        error = Errno_Message(work_in);
        return false;
    }
    std::string work = Absolute(work_in);
    if (!Prepare_Data_Dir(data, disc, work, data_dir, error)) {
        return false;
    }

    // The game reads INI lines into a fixed buffer (INIClass::MAX_LINE_LENGTH) and ignores
    // values that don't fit, which would silently fall back to the default data path.
    const size_t max_value = 256 - sizeof("DataPath=");
    if (data_dir.size() >= max_value || work.size() >= max_value) {
        error = "work directory path too long for the game's INI parser (max " + std::to_string(max_value - 1)
                + " characters)";
        return false;
    }

    std::string ini_path = work + "/CONQUER.INI";
    FILE* ini = fopen(ini_path.c_str(), "w");
    if (ini == nullptr) {
        error = Errno_Message(ini_path);
        return false;
    }
    fprintf(ini, "[Paths]\nDataPath=%s\nUserPath=%s\n\n[Intro]\nPlayIntro=no\n", data_dir.c_str(), work.c_str());
    fclose(ini);

    lib_copy = work + "/" TDHOST_LIB_NAME;
    unlink(lib_copy.c_str());
    return Copy_File(lib, lib_copy, error);
}

/*
** Difficulty rules, indexed by handicap (easy, normal, hard). Remaster builds of the dll have
** none of their own (the remaster passed its own in), so every handicap would play the same.
** Easy and hard are the values the vanilla game uses; normal is the dll's built-in default,
** which keeps normal games exactly as they were.
*/
static const CNCDifficultyDataStruct DifficultyRules[3] = {
    {1.1f, 1.1f, 1.1f, 1.0f, 0.8f, 0.8f, 0.6f, 0.001f, 0.002f, false, true, true},
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.02f, 0.03f, false, true, false},
    {0.9f, 0.9f, 0.9f, 1.05f, 1.05f, 1.0f, 1.0f, 0.05f, 0.1f, true, true, true},
};

GameLib::~GameLib()
{
    Unload();
}

template <typename T> static bool Lookup(void* handle, const char* name, T& func, std::string& error)
{
    func = reinterpret_cast<T>(dlsym(handle, name));
    if (func == nullptr) {
        error = std::string("missing symbol ") + name;
        return false;
    }
    return true;
}

bool GameLib::Load(const std::string& path, EventCallback callback, std::string& error)
{
    Unload();
    Handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (Handle == nullptr) {
        error = dlerror();
        return false;
    }
    bool ok = Lookup(Handle, "CNC_Init", Init, error)
              && Lookup(Handle, "CNC_Set_Multiplayer_Data", Set_Multiplayer_Data, error)
              && Lookup(Handle, "CNC_Start_Instance_Variation", Start_Instance_Variation, error)
              && Lookup(Handle, "CNC_Advance_Instance", Advance_Instance, error)
              && Lookup(Handle, "CNC_Get_Game_State", Get_Game_State, error)
              && Lookup(Handle, "CNC_Set_Random_Seed", Set_Random_Seed, error)
              && Lookup(Handle, "CNC_Set_Headless", Set_Headless, error)
              && Lookup(Handle, "CNC_Free_Game", Free_Game, error)
              && Lookup(Handle, "CNC_Set_AI_Difficulty", Set_AI_Difficulty, error)
              && Lookup(Handle, "CNC_Selected_Hunt", Selected_Hunt, error)
              && Lookup(Handle, "CNC_Config", Config, error)
              && Lookup(Handle, "CNC_Get_Visible_Page", Get_Visible_Page, error)
              && Lookup(Handle, "CNC_Get_Palette", Get_Palette, error)
              && Lookup(Handle, "CNC_Handle_Input", Handle_Input, error)
              && Lookup(Handle, "CNC_Handle_Sidebar_Request", Handle_Sidebar_Request, error)
              && Lookup(Handle, "CNC_Handle_Structure_Request", Handle_Structure_Request, error)
              && Lookup(Handle, "CNC_Handle_Unit_Request", Handle_Unit_Request, error)
              && Lookup(Handle, "CNC_Select_Object", Select_Object, error)
              && Lookup(Handle, "CNC_Clear_Object_Selection", Clear_Object_Selection, error);
    if (!ok) {
        Unload();
        return false;
    }
    Set_Headless(true);
    Init("", callback);
    CNCRulesDataStruct rules;
    for (int i = 0; i < 3; ++i) {
        rules.Difficulties[i] = DifficultyRules[i];
    }
    Config(rules);
    return true;
}

void GameLib::Unload()
{
    if (Handle != nullptr) {
        if (Free_Game != nullptr) {
            Free_Game();
            Free_Game = nullptr;
        }
        dlclose(Handle);
        Handle = nullptr;
    }
}

bool GameLib::Start_Skirmish(const SkirmishSettings& settings, const std::string& data_dir, std::string& error)
{
    int count = (int)settings.Players.size();
    if (count < 2 || count > MAX_PLAYERS) {
        error = "need 2 to " + std::to_string(MAX_PLAYERS) + " players";
        return false;
    }

    CNCMultiplayerOptionsStruct game = {};
    game.MPlayerBases = 1;
    game.MPlayerCredits = settings.Credits;
    game.MPlayerTiberium = 1;
    game.MPlayerGhosts = 1;
    game.MPlayerSolo = 1;
    game.MPlayerUnitCount = 0;

    CNCPlayerInfoStruct players[MAX_PLAYERS];
    memset(players, 0, sizeof(players));
    for (int i = 0; i < count; ++i) {
        const PlayerSetup& setup = settings.Players[i];
        snprintf(players[i].Name, sizeof(players[i].Name), "%s%d", setup.IsAI ? "AI" : "Player", i + 1);
        players[i].House = (unsigned char)setup.Side;
        players[i].ColorIndex = i;
        players[i].GlyphxPlayerID = Player_ID(i);
        players[i].Team = i;
        players[i].StartLocationIndex = i;
        players[i].IsAI = setup.IsAI;
    }

    if (!Set_Multiplayer_Data(settings.Map, game, count, players, MAX_PLAYERS)) {
        error = "CNC_Set_Multiplayer_Data failed";
        return false;
    }
    if (settings.Seed != 0) {
        Set_Random_Seed(settings.Seed);
    }
    Set_AI_Difficulty(settings.AIDifficulty);
    // Scenario variation A, east, build level 7, multiplayer game, no sabotaged structure.
    if (!Start_Instance_Variation(
            settings.Map, 0, 0, 7, "GDI", "GAME_GLYPHX_MULTIPLAYER", data_dir.c_str(), -1, nullptr)) {
        error = "CNC_Start_Instance_Variation failed for map " + std::to_string(settings.Map);
        return false;
    }
    return true;
}

} // namespace tdhost

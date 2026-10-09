// tdhost -- shared code for programs that host the Tiberian Dawn remaster dll
// interface outside the remaster: data setup, library loading and starting games.
#pragma once

#include "common/wwstd.h"
#include "dllinterface.h"

#include <stdint.h>
#include <string>
#include <vector>

namespace tdhost {

#ifdef __APPLE__
#define TDHOST_LIB_NAME "TiberianDawn.dylib"
#else
#define TDHOST_LIB_NAME "TiberianDawn.so"
#endif

// Normal game speed, from TICKS_PER_SECOND in defines.h.
const int TICKS_PER_SECOND = 15;
// Player limit, from MAX_PLAYERS in defines.h.
const int MAX_PLAYERS = 6;

typedef void(__cdecl* EventCallback)(const EventCallbackStruct& event);

std::string Absolute(const std::string& path);

/*
** Give the library a private work directory holding a CONQUER.INI that points at the
** game data, and a copy of the library to load. The library finds its data via that
** ini next to itself; a symlink won't do because macOS reports a loaded library by its
** resolved path. In dll mode the game only searches one data directory, so data with
** gdi/ and nod/ disc folders is presented as a flat directory of links, using 'disc'
** for GENERAL.MIX and MOVIES.MIX.
*/
bool Prepare_Work_Dir(const std::string& lib,
                      const std::string& data,
                      const std::string& disc,
                      const std::string& work,
                      std::string& lib_copy,
                      std::string& data_dir,
                      std::string& error);

struct PlayerSetup
{
    bool IsAI;
    int Side; // 0 = GDI, 1 = Nod.
};

struct SkirmishSettings
{
    int Map = 1; // Multiplayer scenario number, e.g. 1 for SCM01EA.
    int Credits = 5000;
    unsigned Seed = 0; // 0 leaves the library's fixed starting state.
    int AIDifficulty = 1; // AI players: 0 easy, 1 normal, 2 hard.
    std::vector<PlayerSetup> Players;
};

/*
** The game library, its entry points and a running game. The library keeps all game
** state in globals, so there can only be one game per process; Load() after Unload()
** gives a completely fresh state, which restarting a scenario in place does not.
*/
class GameLib
{
public:
    ~GameLib();

    // Loads the library in headless mode: it never draws the legacy game screen.
    bool Load(const std::string& path, EventCallback callback, std::string& error);
    // Frees the game's allocations and unloads the library.
    void Unload();
    bool Is_Loaded() const
    {
        return Handle != nullptr;
    }

    // Player n in the settings gets GlyphX player id n + 1.
    bool Start_Skirmish(const SkirmishSettings& settings, const std::string& data_dir, std::string& error);

    static uint64_t Player_ID(int index)
    {
        return index + 1;
    }

    void(__cdecl* Init)(const char*, EventCallback) = nullptr;
    bool(__cdecl* Set_Multiplayer_Data)(int, CNCMultiplayerOptionsStruct&, int, CNCPlayerInfoStruct*, int) = nullptr;
    bool(__cdecl* Start_Instance_Variation)(int, int, int, int, const char*, const char*, const char*, int, const char*) =
        nullptr;
    bool(__cdecl* Advance_Instance)(uint64_t) = nullptr;
    bool(__cdecl* Get_Game_State)(GameStateRequestEnum, uint64_t, unsigned char*, unsigned int) = nullptr;
    void(__cdecl* Set_Random_Seed)(unsigned int) = nullptr;
    void(__cdecl* Set_Headless)(bool) = nullptr;
    void(__cdecl* Free_Game)(void) = nullptr;
    void(__cdecl* Set_AI_Difficulty)(int) = nullptr;
    void(__cdecl* Config)(const CNCRulesDataStruct&) = nullptr;
    bool(__cdecl* Get_Visible_Page)(unsigned char*, unsigned int&, unsigned int&) = nullptr;
    bool(__cdecl* Get_Palette)(unsigned char (&)[256][3]) = nullptr;
    void(__cdecl* Handle_Input)(InputRequestEnum, unsigned char, uint64_t, int, int, int, int) = nullptr;
    void(__cdecl* Handle_Sidebar_Request)(SidebarRequestEnum, uint64_t, int, int, short, short) = nullptr;
    void(__cdecl* Handle_Structure_Request)(StructureRequestEnum, uint64_t, int) = nullptr;
    void(__cdecl* Handle_Unit_Request)(UnitRequestEnum, uint64_t) = nullptr;
    bool(__cdecl* Select_Object)(uint64_t, int, int) = nullptr;
    bool(__cdecl* Clear_Object_Selection)(uint64_t) = nullptr;

private:
    void* Handle = nullptr;
};

} // namespace tdhost

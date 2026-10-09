// tdbench -- runs AI-only Tiberian Dawn skirmishes through the remaster dll
// interface with no rendering or frame limiting, and reports simulation speed.
//
// Each process runs one game at a time. Run several processes for parallel
// throughput; give each its own --work directory.

#include "tdhost.h"

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

struct Options
{
    std::string Lib;
    std::string Data;
    std::string Work = "tdbench-run";
    std::string Disc = "gdi";
    int AIs = 2;
    int Map = 1;
    int Credits = 5000;
    int AIDifficulty = 1;
    unsigned Seed = 0; // 0 leaves the library's fixed starting state.
    int Games = 1;
    bool Reload = false;
    long MaxFrames = 15L * 60 * 60; // One hour of game time.
};

static bool Verbose = false;
static bool DumpObjects = false;
static bool GameOver = false;

static void __cdecl Event_Callback(const EventCallbackStruct& event)
{
    if (event.EventType == CALLBACK_EVENT_GAME_OVER) {
        GameOver = true;
    }
    if (Verbose && event.EventType == CALLBACK_EVENT_DEBUG_PRINT && event.DebugPrint.PrintString != nullptr) {
        fprintf(stderr, "[dll] %s", event.DebugPrint.PrintString);
    }
}

static void Usage(const char* prog)
{
    fprintf(stderr,
            "usage: %s --lib PATH --data DIR [--disc gdi|nod] [--work DIR] [--ais N] [--map N] [--credits N] [--seed N]\n"
            "       [--ai-difficulty 0-2] [--games N] [--reload] [--frames N] [--verbose] [--dump-objects]\n"
            "  --lib      path to " TDHOST_LIB_NAME "\n"
            "  --data     directory holding the game .MIX files, either flat or with gdi/ and nod/ disc folders\n"
            "  --disc     disc folder to take GENERAL.MIX and MOVIES.MIX from (default gdi)\n"
            "  --work     per-process scratch directory (default tdbench-run)\n"
            "  --ais      number of AI players, 2-%d (default 2)\n"
            "  --map      multiplayer scenario number, e.g. 1 for SCM01EA (default 1)\n"
            "  --credits  starting credits (default 5000)\n"
            "  --ai-difficulty  0 easy, 1 normal, 2 hard (default 1)\n"
            "  --seed     random seed; runs with the same seed and settings play identically\n"
            "  --games    games to play back to back in this process; game g uses seed + g (default 1)\n"
            "  --reload   reload the library between games for a completely fresh game state\n"
            "  --frames   stop after this many game frames (default %d, one hour of game time)\n"
            "  --verbose  print the library's debug messages\n"
            "  --dump-objects  list every object at the end of the run on stdout\n",
            prog,
            tdhost::MAX_PLAYERS,
            15 * 60 * 60);
}

static bool Parse_Args(int argc, char** argv, Options& opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose") {
            Verbose = true;
            continue;
        }
        if (arg == "--dump-objects") {
            DumpObjects = true;
            continue;
        }
        if (arg == "--reload") {
            opt.Reload = true;
            continue;
        }
        if (i + 1 >= argc) {
            return false;
        }
        const char* val = argv[++i];
        if (arg == "--lib") {
            opt.Lib = val;
        } else if (arg == "--data") {
            opt.Data = val;
        } else if (arg == "--work") {
            opt.Work = val;
        } else if (arg == "--disc") {
            opt.Disc = val;
        } else if (arg == "--ais") {
            opt.AIs = atoi(val);
        } else if (arg == "--map") {
            opt.Map = atoi(val);
        } else if (arg == "--credits") {
            opt.Credits = atoi(val);
        } else if (arg == "--ai-difficulty") {
            opt.AIDifficulty = atoi(val);
        } else if (arg == "--seed") {
            opt.Seed = (unsigned)strtoul(val, nullptr, 0);
        } else if (arg == "--games") {
            opt.Games = atoi(val);
        } else if (arg == "--frames") {
            opt.MaxFrames = atol(val);
        } else {
            return false;
        }
    }
    return !opt.Lib.empty() && !opt.Data.empty() && opt.Games >= 1 && opt.AIs >= 2 && opt.AIs <= tdhost::MAX_PLAYERS
           && opt.MaxFrames > 0;
}

static unsigned char StateBuffer[8 * 1024 * 1024];

static const CNCObjectListStruct* Get_Objects(tdhost::GameLib& game)
{
    if (!game.Get_Game_State(GAME_STATE_LAYERS, tdhost::GameLib::Player_ID(0), StateBuffer, sizeof(StateBuffer))) {
        return nullptr;
    }
    return reinterpret_cast<const CNCObjectListStruct*>(StateBuffer);
}

/*
** FNV-1a hash of every object's identity, owner, position and health. Two runs that
** end with the same hash ended in the same state.
*/
static uint64_t State_Hash(tdhost::GameLib& game)
{
    const CNCObjectListStruct* list = Get_Objects(game);
    if (list == nullptr) {
        return 0;
    }
    uint64_t hash = 14695981039346656037ULL;
    auto mix = [&hash](const void* data, size_t len) {
        const unsigned char* bytes = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < len; ++i) {
            hash = (hash ^ bytes[i]) * 1099511628211ULL;
        }
    };
    for (int i = 0; i < list->Count; ++i) {
        const CNCObjectStruct& obj = list->Objects[i];
        mix(obj.TypeName, strnlen(obj.TypeName, sizeof(obj.TypeName)));
        // Owner is a plain char, signed on some platforms and unsigned on others.
        int fields[] = {(int)obj.Type, obj.ID, obj.PositionX, obj.PositionY, obj.Strength, (unsigned char)obj.Owner};
        mix(fields, sizeof(fields));
    }
    return hash;
}

/*
** Print how many objects each house owns, as seen through the game state interface.
*/
static void Dump_Objects(tdhost::GameLib& game, const char* when)
{
    const CNCObjectListStruct* list = Get_Objects(game);
    if (list == nullptr) {
        fprintf(stderr, "tdbench: GAME_STATE_LAYERS failed (%s)\n", when);
        return;
    }
    int per_owner[256] = {};
    for (int i = 0; i < list->Count; ++i) {
        ++per_owner[(unsigned char)list->Objects[i].Owner];
    }
    fprintf(stderr, "tdbench: %s: %d objects;", when, list->Count);
    for (int owner = 0; owner < 256; ++owner) {
        if (per_owner[owner]) {
            fprintf(stderr, " house %d: %d", owner, per_owner[owner]);
        }
    }
    fprintf(stderr, "\n");
}

int main(int argc, char** argv)
{
    Options opt;
    if (!Parse_Args(argc, argv, opt)) {
        Usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string error;
    std::string lib_copy;
    std::string data_dir;
    if (!tdhost::Prepare_Work_Dir(opt.Lib, opt.Data, opt.Disc, opt.Work, lib_copy, data_dir, error)) {
        fprintf(stderr, "tdbench: %s\n", error.c_str());
        return EXIT_FAILURE;
    }

    tdhost::GameLib game;
    auto t_start = std::chrono::steady_clock::now();
    if (!game.Load(lib_copy, Event_Callback, error)) {
        fprintf(stderr, "tdbench: %s\n", error.c_str());
        return EXIT_FAILURE;
    }

    for (int g = 0; g < opt.Games; ++g) {
        GameOver = false;
        auto t_game = std::chrono::steady_clock::now();
        if (opt.Reload && g > 0 && !game.Load(lib_copy, Event_Callback, error)) {
            fprintf(stderr, "tdbench: %s\n", error.c_str());
            return EXIT_FAILURE;
        }

        tdhost::SkirmishSettings settings;
        settings.Map = opt.Map;
        settings.Credits = opt.Credits;
        settings.AIDifficulty = opt.AIDifficulty;
        settings.Seed = opt.Seed != 0 ? opt.Seed + g : 0;
        for (int i = 0; i < opt.AIs; ++i) {
            settings.Players.push_back({true, i % 2}); // Alternate GDI and Nod.
        }
        if (!game.Start_Skirmish(settings, data_dir, error)) {
            fprintf(stderr, "tdbench: %s\n", error.c_str());
            return EXIT_FAILURE;
        }
        auto t_ready = std::chrono::steady_clock::now();
        if (Verbose) {
            Dump_Objects(game, "start");
        }

        long frames = 0;
        const char* reason = "frame limit";
        while (frames < opt.MaxFrames) {
            bool running = game.Advance_Instance(tdhost::GameLib::Player_ID(0));
            ++frames;
            if (Verbose && frames <= 2) {
                Dump_Objects(game, frames == 1 ? "frame 1" : "frame 2");
            }
            if (GameOver) {
                reason = "game over";
                break;
            }
            if (!running) {
                reason = "advance returned false";
                break;
            }
        }
        auto t_end = std::chrono::steady_clock::now();
        Dump_Objects(game, "end");
        uint64_t state = State_Hash(game);
        if (DumpObjects) {
            const CNCObjectListStruct* list = Get_Objects(game);
            for (int i = 0; list != nullptr && i < list->Count; ++i) {
                const CNCObjectStruct& obj = list->Objects[i];
                printf("object %-12.*s type=%d id=%d owner=%d pos=%d,%d strength=%d\n",
                       (int)sizeof(obj.TypeName),
                       obj.TypeName,
                       (int)obj.Type,
                       obj.ID,
                       (unsigned char)obj.Owner,
                       obj.PositionX,
                       obj.PositionY,
                       obj.Strength);
            }
        }

        double setup_s = std::chrono::duration<double>(t_ready - (g == 0 ? t_start : t_game)).count();
        double sim_s = std::chrono::duration<double>(t_end - t_ready).count();
        double fps = frames / sim_s;
        printf("map=%d ais=%d seed=%u frames=%ld game_minutes=%.1f setup_s=%.2f sim_s=%.2f fps=%.0f realtime_x=%.1f "
               "end=\"%s\" state=%016llx\n",
               opt.Map,
               opt.AIs,
               settings.Seed,
               frames,
               frames / (tdhost::TICKS_PER_SECOND * 60.0),
               setup_s,
               sim_s,
               fps,
               fps / tdhost::TICKS_PER_SECOND,
               reason,
               (unsigned long long)state);
        fflush(stdout);
    }
    return EXIT_SUCCESS;
}

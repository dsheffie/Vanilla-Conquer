/*
** tdenv -- C interface to a Tiberian Dawn skirmish for reinforcement learning.
**
** One agent plays player 0 against built-in AI players. Everything is seen from the
** agent's side with human-level information: other houses' objects are only reported
** in cells the agent has explored (Tiberian Dawn has shroud but no fog of war, so an
** explored cell stays observable).
**
** Positions use the playable grid of the map, the same grid used for building
** placement: cell (0, 0) is the top left of the map including its one cell border.
**
** The game library keeps its state in globals, so a process hosts one environment.
** Reset reloads the library, giving a completely fresh game every episode.
**
** Typical use: tdenv_open, then per episode tdenv_reset followed by repeated
** tdenv_step / tdenv_observe and the getters and actions below.
*/
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define TDENV_NAME_LENGTH 16

/* tdenv_step results. */
#define TDENV_RUNNING 0
#define TDENV_WON     1
#define TDENV_LOST    2
#define TDENV_ERROR   -1

/* TDObject.relation */
#define TDENV_SELF    0
#define TDENV_ALLY    1
#define TDENV_ENEMY   2
#define TDENV_NEUTRAL 3

/* TDObject.type, matching DllObjectTypeEnum. */
#define TDENV_INFANTRY 1
#define TDENV_UNIT     2
#define TDENV_AIRCRAFT 3
#define TDENV_BUILDING 4
#define TDENV_TERRAIN  5

typedef struct
{
    int id;   /* Per type id, used to command the object. */
    int type; /* TDENV_INFANTRY etc. */
    char name[TDENV_NAME_LENGTH];
    int relation; /* TDENV_SELF etc. */
    int house;
    int cell_x, cell_y;   /* Grid cell; a building's top left cell. */
    int size_x, size_y;   /* Cells covered: a building's footprint, 1 x 1 otherwise. */
    int pixel_x, pixel_y; /* Center in pixels, 24 per cell. */
    int strength, max_strength;
    int can_move;
    int can_deploy;
    int can_harvest;
    int is_factory;
    int is_selectable;
    int pips, max_pips; /* E.g. a harvester's load. */
} TDObject;

typedef struct
{
    char name[TDENV_NAME_LENGTH];
    int type; /* TDENV_INFANTRY etc., or 0 for a special weapon. */
    int cost;
    int build_time;
    float progress; /* 0 to 1. */
    int completed;  /* Finished; a building then needs placing. */
    int constructing;
    int on_hold;
    int busy; /* Its factory is building something else. */
} TDBuildable;

typedef struct
{
    int frame; /* Game frames since reset, 15 per second of game time. */
    int credits;
    int tiberium; /* Stored in refineries and silos. */
    int max_tiberium;
    int power_produced;
    int power_drained;
    int units_killed;
    int buildings_killed;
    int units_lost;
    int buildings_lost;
    int harvested_credits;
    int defeated;
} TDScalars;

/* Last error message, for any call that failed. */
const char* tdenv_last_error(void);

/*
** Prepare a work directory for this environment. lib is TiberianDawn.so/.dylib, data the
** game data directory (flat, or with gdi/ and nod/ folders, picking 'disc'). Returns 0 on
** success. Each environment in a process tree needs its own work directory.
*/
int tdenv_open(const char* lib, const char* data, const char* disc, const char* work);

/*
** Start a new game on multiplayer map 'map' (e.g. 1 for SCM01EA) against num_ais AI
** players. agent_side is 0 for GDI, 1 for Nod; AI sides alternate starting with the
** other one. A non-zero seed makes the game reproducible. Returns 0 on success.
*/
int tdenv_reset(int map, int num_ais, unsigned seed, int agent_side, int credits);

/* Size of the playable grid. */
int tdenv_grid_size(int* width, int* height);

/*
** Draw the game screen every frame, so tdenv_frame can capture it. Off by default: drawing
** slows the simulation down. Applies from the next reset, or immediately in a game.
*/
int tdenv_set_rendering(int enabled);
/*
** The current frame as RGB, 3 bytes per pixel in row order, as the agent sees it (its own
** shroud). Writes at most max_bytes and sets the size; the whole map at 24 pixels a cell.
*/
int tdenv_frame(unsigned char* rgb, int max_bytes, int* width, int* height);

/* Advance the game by 'frames' frames, stopping early if the game ends. */
int tdenv_step(int frames);

/* Capture the current state; the getters below return what this captured. */
int tdenv_observe(void);

int tdenv_scalars(TDScalars* out);
/* Visible objects, excluding sub-objects such as shadows. Returns the count. */
int tdenv_objects(TDObject* out, int max);
/* Per cell, width * height bytes in row order: 1 if the agent has explored the cell. */
int tdenv_shroud(unsigned char* mapped);
/* Per cell Tiberium density, 0 for none, 1-12 otherwise; 0 in unexplored cells. */
int tdenv_tiberium(unsigned char* density);
/* What the agent can currently build. Returns the count. */
int tdenv_buildables(TDBuildable* out, int max);

/*
** Actions, taking effect on the next step. Each returns 1 if the request was issued,
** 0 if it could not apply (unknown name, not ready, not the agent's object...).
*/
int tdenv_build(const char* name);
int tdenv_cancel(const char* name);
/* Place a completed building with its top left cell at (x, y). */
int tdenv_place(const char* name, int x, int y);
/*
** Where a completed building can be placed: per cell, width * height bytes in row order,
** 1 if the building can have its top left cell there. Returns the number of such cells,
** 0 if the building isn't waiting to be placed.
*/
int tdenv_placement(const char* name, unsigned char* valid);
/*
** Select the agent's objects (types[i], ids[i]) and command them at cell (x, y), as a
** human clicking there: move, attack what is there, harvest Tiberium, enter a building,
** or deploy an MCV commanded onto its own cell.
*/
int tdenv_command(const int* types, const int* ids, int count, int x, int y);
int tdenv_stop(const int* types, const int* ids, int count);
int tdenv_sell(int building_id);

#ifdef __cplusplus
}
#endif

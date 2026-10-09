"""Tiberian Dawn as a Gymnasium environment, via the remaster dll interface.

Build the TiberianDawn library and libtdenv (configure with -DBUILD_REMASTERTD=ON), then
point the environment at them and at the game data:

    import gymnasium as gym
    import tdgym

    env = gym.make("TiberianDawn-v0", game_lib=".../TiberianDawn.so", data=".../data",
                   tdenv_lib=".../libtdenv.so")

The paths can also come from $TDGYM_GAME_LIB, $TDGYM_DATA and $TDGYM_TDENV.
"""

from gymnasium.envs.registration import register

from .env import (
    ACTION_KINDS,
    BUILDABLE_FEATURES,
    CANCEL,
    CATALOG,
    COMMAND,
    GRID,
    NOOP,
    PLACE,
    PLANES,
    SCALARS,
    SELL,
    STOP,
    BUILD,
    TiberianDawnEnv,
)

register(id="TiberianDawn-v0", entry_point="tdgym.env:TiberianDawnEnv")

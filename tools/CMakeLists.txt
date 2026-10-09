if (BUILD_TOOLS)
    add_subdirectory(miniposix)
    add_subdirectory(mixtool)
endif()

# Headless hosts for the remaster dll interface: a simulation benchmark and an RL environment.
if (BUILD_REMASTERTD AND NOT WIN32)
    add_subdirectory(tdhost)
    add_subdirectory(tdbench)
    add_subdirectory(tdgym)
endif()

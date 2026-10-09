add_executable(tdbench tdbench.cpp)
target_link_libraries(tdbench tdhost)
# Loaded at runtime with dlopen, but build it alongside.
add_dependencies(tdbench TiberianDawn)

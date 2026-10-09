add_library(tdhost STATIC tdhost.cpp)
target_compile_definitions(tdhost PUBLIC TIBERIAN_DAWN MEGAMAPS)
target_include_directories(tdhost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tiberiandawn)
target_link_libraries(tdhost PUBLIC ${CMAKE_DL_LIBS})
# Linked into the tdenv shared library.
set_target_properties(tdhost PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tdenv SHARED tdenv.cpp)
target_link_libraries(tdenv PRIVATE tdhost)
# Loaded from Python as tdgym/libtdenv.so or .dylib.
set_target_properties(tdenv PROPERTIES PREFIX "lib" SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}")
# The game library it hosts is loaded at runtime with dlopen, but build it alongside.
add_dependencies(tdenv TiberianDawn)

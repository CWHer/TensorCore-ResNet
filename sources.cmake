# Include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ... Add more if you need them

# Common source files **without** main function
set(SOURCE_FILES
    src/half.cpp
    src/reg_file.cpp
    src/simulator.cpp
    src/functions.cpp
)
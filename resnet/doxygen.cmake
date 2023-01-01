option(GENERATE_DOCS "Build documentation" OFF)

if (NOT GENERATE_DOCS)
    message(STATUS "Reference documentation generation is disabled.")
    message(STATUS "Set CMake option GENERATE_DOCS to ON to enable it.")
    return()
endif ()

# Documentation
message(CHECK_START "Check if Doxygen is installed")

find_package(Doxygen)
if (DOXYGEN_FOUND)
    message(CHECK_PASS "Found")
    # set input and output files
    set(__doxygen_input ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${CMAKE_CURRENT_SOURCE_DIR}/include/
            src/python_helpers/
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/ ${CMAKE_CURRENT_SOURCE_DIR}/tests/
            ${CMAKE_CURRENT_SOURCE_DIR}/../README.md)

    # Combine all the files into a single string
    string(REPLACE ";" " " DOXYGEN_IN "${__doxygen_input}")
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/docs/)
    set(README_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../README.md)

    set(PROJECT_BRIEF "Final Project for CS433")
    set(ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/)

    # request to configure the file
    configure_file(doxygen.conf.in doxygen.conf @ONLY)
    message(STATUS "Doxygen configuration started")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(doc
            COMMAND ${DOXYGEN_EXECUTABLE} doxygen.conf
            COMMAND $(MAKE) -C ${CMAKE_CURRENT_BINARY_DIR}/docs/latex
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/docs/latex/refman.pdf
            ${CMAKE_CURRENT_BINARY_DIR}/docs.pdf
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating documentation with Doxygen"
            VERBATIM)
else (DOXYGEN_FOUND)
    message(CHECK_FAIL "Not found")
    message(STATUS "Doxygen is needed to build the documentation.")
endif (DOXYGEN_FOUND)

cmake_minimum_required(VERSION 3.20)

project(dudu-datascience LANGUAGES CXX)

set(DUC_EXECUTABLE "" CACHE FILEPATH "Path to the duc compiler")
if(NOT DUC_EXECUTABLE)
    find_program(DUC_EXECUTABLE_FOUND duc REQUIRED)
    set(DUC_EXECUTABLE "${DUC_EXECUTABLE_FOUND}" CACHE FILEPATH "Path to the duc compiler" FORCE)
endif()

set(DUDU_ENTRY "${CMAKE_CURRENT_SOURCE_DIR}/src/main.dd")
set(DUDU_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/dudu.toml")
set(DUDU_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/dudu-generated")
set(DUDU_GENERATED_STAMP "${DUDU_GENERATED_DIR}/.dudu_emit.stamp")
file(GLOB_RECURSE DUDU_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.dd")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${DUDU_GENERATED_DIR}"
    COMMAND "${DUC_EXECUTABLE}" emit-modules "${DUDU_ENTRY}" -o "${DUDU_GENERATED_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
)
file(GLOB_RECURSE DUDU_GENERATED_SOURCES CONFIGURE_DEPENDS "${DUDU_GENERATED_DIR}/*.cpp")

add_custom_command(
    OUTPUT "${DUDU_GENERATED_STAMP}"
    BYPRODUCTS ${DUDU_GENERATED_SOURCES}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${DUDU_GENERATED_DIR}"
    COMMAND "${DUC_EXECUTABLE}" emit-modules "${DUDU_ENTRY}" -o "${DUDU_GENERATED_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${DUDU_GENERATED_STAMP}"
    DEPENDS "${DUC_EXECUTABLE}" "${DUDU_MANIFEST}" ${DUDU_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM
)
add_custom_target(dudu_generate DEPENDS "${DUDU_GENERATED_STAMP}")

add_executable(dudu-datascience ${DUDU_GENERATED_SOURCES})
add_dependencies(dudu-datascience dudu_generate)
target_include_directories(dudu-datascience PRIVATE "${DUDU_GENERATED_DIR}")
target_compile_features(dudu-datascience PRIVATE cxx_std_20)

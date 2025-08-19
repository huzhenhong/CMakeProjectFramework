
 aux_source_directory(. SRCS)
message("SRCS: ${SRCS}")

add_executable(example ${SRCS})

set(link_libs libCMakeProjectFramework_d.so)

# target_link_libraries(example PRIVATE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${link_libs})
target_link_libraries(example PRIVATE CMakeProjectFramework)

target_include_directories(example 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
)


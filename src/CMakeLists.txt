
set(Target ${CMAKE_PROJECT_NAME})

message("\n------------------ Start generate : ${Target} ------------------")

file(GLOB_RECURSE H_SRCS *.h)
file(GLOB_RECURSE C_SRCS *.c)
file(GLOB_RECURSE HPP_SRCS *.hpp)
file(GLOB_RECURSE CPP_SRCS *.cpp)
set(${Target}_SRCS ${H_SRCS} ${C_SRCS} ${HPP_SRCS} ${CPP_SRCS})

add_library(${Target} SHARED ${${Target}_SRCS})

# 导出接口头文件
set_target_properties(${Target} 
    PROPERTIES 
        PUBLIC_HEADER 
            ${CMAKE_CURRENT_SOURCE_DIR}/Algorithm/IAlgorithm.h
    )

message("------------------ Stop generate : ${Target} ------------------\n")

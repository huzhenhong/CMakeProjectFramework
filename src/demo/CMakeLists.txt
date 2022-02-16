
set(Target ${PROJECT_NAME})

message("\n------------------ Start generate : ${Target} ------------------")

file(GLOB_RECURSE H_SRCS *.h)
file(GLOB_RECURSE C_SRCS *.c)
file(GLOB_RECURSE HPP_SRCS *.hpp)
file(GLOB_RECURSE CPP_SRCS *.cpp)
set(${Target}_SRCS ${H_SRCS} ${C_SRCS} ${HPP_SRCS} ${CPP_SRCS})

add_library(${Target} SHARED ${${Target}_SRCS})
add_library(${Target}::${Target} ALIAS ${Target})
# set_target_properties(${Target} 
#     PROPERTIES 
#         # OUTPUT_NAME TEST::${Target})
#         NAMESPACE TEST)

# 导出接口头文件
set_target_properties(${Target} 
    PROPERTIES 
        PUBLIC_HEADER 
            ${CMAKE_CURRENT_SOURCE_DIR}/IAlgorithm.h
    )

message("------------------ Stop generate : ${Target} ------------------\n")

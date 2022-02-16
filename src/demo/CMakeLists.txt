
set(Target ${PROJECT_NAME})

message("\n------------------ Start generate : ${Target} ------------------")

aux_source_directory(. ${Target}_SRCS)

add_library(${Target} SHARED ${${Target}_SRCS})

# 导出接口头文件
set_target_properties(${Target} 
    PROPERTIES 
        PUBLIC_HEADER 
            ${CMAKE_CURRENT_SOURCE_DIR}/IAlgorithm.h
    )

message("------------------ Stop generate : ${Target} ------------------\n")

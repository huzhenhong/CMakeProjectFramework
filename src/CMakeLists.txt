
set(Target ${PROJECT_NAME})

message("\n------------------ Start generate : ${Target} ------------------")


aux_source_directory(./ExportC ExportC_SRCS)
aux_source_directory(./ExportCplus ExportCplus_SRCS)
aux_source_directory(./ExportClass ExportClass_SRCS)
aux_source_directory(./ExportAbstractClass ExportAbstractClass_SRCS)

set(${Target}_SRCS 
        ${ExportC_SRCS} 
        ${ExportCplus_SRCS} 
        ${ExportClass_SRCS} 
        ${ExportAbstractClass_SRCS})

add_library(${Target} SHARED ${${Target}_SRCS})

# 开启导出动态库
target_compile_definitions(${Target}
    PRIVATE
        LIBRARY_EXPORT
)

if(IS_BUILD_TRACY)
    add_definitions(-D TRACY_ENABLE)

    include(FetchContent)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git 
        GIT_TAG v0.10
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/tracy
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        OPTIONS "-DBUILD_EXTRAS=OFF" # 禁止构建额外组件
        )
    FetchContent_MakeAvailable(tracy)

    target_link_libraries(${Target} PUBLIC Tracy::TracyClient)
endif()



# 导出接口头文件
set(${Target}_EXPORT_HEADER 
    ${CMAKE_CURRENT_SOURCE_DIR}/common/SysInclude.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/ExportMarco.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ExportAbstractClass/AbstractClassInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ExportC/CInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ExportCplus/CplusInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ExportClass/ClassInterface.h)

set_target_properties(${Target} 
    PROPERTIES 
        PUBLIC_HEADER 
            "${${Target}_EXPORT_HEADER}"
    )

message("------------------ Stop generate : ${Target} ------------------\n")


set(Target ${PROJECT_NAME})

message("\n------------------ Start generate : ${Target} ------------------")


# aux_source_directory(./ExportC ExportC_SRCS)
# aux_source_directory(./ExportCplus ExportCplus_SRCS)
# aux_source_directory(./ExportClass ExportClass_SRCS)
# aux_source_directory(./ExportAbstractClass ExportAbstractClass_SRCS)

# set(${Target}_SRCS 
#         ${ExportC_SRCS} 
#         ${ExportCplus_SRCS} 
#         ${ExportClass_SRCS} 
#         ${ExportAbstractClass_SRCS})

# add_library(${Target} SHARED ${${Target}_SRCS})
# link_directories(${CMAKE_SOURCE_DIR}/include)
aux_source_directory(. SRCS)
add_library(${Target} SHARED ${SRCS})

message("src: " ${SRCS})
# message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

target_include_directories(${Target} 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${Target} PUBLIC CUDA::cudart)

target_compile_options(${Target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: -O3 -lineinfo --expt-extended-lambda --expt-relaxed-constexpr>
)

# 库名称导出统一（Windows 下 .dll/.lib，非 Windows 下 .so/.dylib）
# set_target_properties(${Target} PROPERTIES OUTPUT_NAME "myalgo")

# link_directories(${CMAKE_SOURCE_DIR}/include)

# set_target_properties(${Target} PROPERTIES LINKER_LANGUAGE CXX)

# 打印 target 的 include 路径
# get_target_property(MYAPP_INCLUDES ${Target} INCLUDE_DIRECTORIES)
# message(STATUS "MyApp include dirs = ${MYAPP_INCLUDES}")

# # 开启导出动态库
# target_compile_definitions(${Target}
#     PRIVATE
#         LIBRARY_EXPORT
# )

# if(IS_BUILD_TRACY)
#     add_definitions(-D TRACY_ENABLE)

#     include(FetchContent)
#     FetchContent_Declare(
#         tracy
#         GIT_REPOSITORY https://github.com/wolfpld/tracy.git 
#         GIT_TAG v0.10
#         SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/tracy
#         GIT_SHALLOW TRUE
#         GIT_PROGRESS TRUE
#         OPTIONS "-DBUILD_EXTRAS=OFF" # 禁止构建额外组件
#         )
#     FetchContent_MakeAvailable(tracy)

#     target_link_libraries(${Target} PUBLIC Tracy::TracyClient)
# endif()



# 导出接口头文件
# set(${Target}_EXPORT_HEADER 
#     ${CMAKE_CURRENT_SOURCE_DIR}/common/SysInclude.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/common/ExportMarco.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/ExportAbstractClass/AbstractClassInterface.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/ExportC/CInterface.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/ExportCplus/CplusInterface.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/ExportClass/ClassInterface.h)

# set_target_properties(${Target} 
#     PROPERTIES 
#         PUBLIC_HEADER 
#             "${${Target}_EXPORT_HEADER}"
#     )

message("------------------ Stop generate : ${Target} ------------------\n")

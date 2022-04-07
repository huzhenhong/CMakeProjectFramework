## C++跨平台开发框架

#### 开启代码提交格式化（可选）

运行根目录下的 `pre-install.sh` 或者 `pre-install.bat`

#### 使用方法

- 修改根目录下的CMakeLists.txt，修改工程名字等信息

  ```cmake
  # 这里工程名称为 "CMakeProjectFramework"，可以自行修改为想要导出的库的名字
  project("CMakeProjectFramework" 
      VERSION 1.0.0.0
      LANGUAGES CXX C
      DESCRIPTION "something about this project"
      HOMEPAGE_URL "project site"
      )
  ```

  同步修改 config\\**CMakeProjectFramework**Config.cmake.in 为工程名

- 修改src目录下的CMakeLists.txt

  ```cmake
  # 开启导出动态库，不开启 set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) 时需要
  target_compile_definitions(${Target}
      PRIVATE
          LIBRARY_EXPORT
  )
  
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
  ```

- src目录下添加库实现代码

- test目录下添加测试代码

- 子仓库

  ```sh
  # 添加
  git submodule add [-b master]<仓库地址> <本地路径>
  
  # 递归更新子模块
  git submodule update --init --recursive
  
  # 删除
  git submodule deinit sub-name
  git rm sub-name
  ```
  
- 运行根目录下的build_for_xxx，在对应平台完成编译和安装，可以修改编译类型，是否编译测试模块，是否进行安装

  - windows

    ```bash
    set build_type=Debug
    @REM set build_type=Release
    @REM set build_type=RelWithDebInfo
    set is_build_test=OFF
    set is_install=ON
    set install_prefix=./install/
    ```

  - linux/mac

    ```sh
    build_type=Debug
    # build_type=Release
    # build_type=RelWithDebInfo 
    is_build_test=OFF
    is_install=ON
    install_prefix=./install/
    ```

    

​	
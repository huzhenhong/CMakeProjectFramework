### 一个比较简单的跨平台C++开发框架

##### 使用方法

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

- 修改src目录下的CMakeLists.txt

  ```cmake
  # 可以更改生成Target的名称
  set(Target ${CMAKE_PROJECT_NAME})
  ```

- src/lib-impl目录下添加库实现代码

- src/test目录下添加测试代码

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
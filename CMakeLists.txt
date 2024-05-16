cmake_minimum_required(VERSION 3.8.0)

project(WatchOut
    VERSION 0.1
    DESCRIPTION "Driver Monitoring System v0.1"
    LANGUAGES CXX)

set(DLIB_SOURCE_DIR /home/jetson/dlib)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_executable(WatchOut
    ${CMAKE_SOURCE_DIR}/watchout/main.cpp
    ${CMAKE_SOURCE_DIR}/watchout/watchout.cpp)

add_subdirectory(${CMAKE_SOURCE_DIR}/lib)
add_subdirectory(${DLIB_SOURCE_DIR}/dlib dlib_build)

target_include_directories(WatchOut PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS})

target_link_libraries(WatchOut PUBLIC
    Threads::Threads
    dlib::dlib
    ${OpenCV_LIBS}
    Qt5::Widgets)

cmake_minimum_required(VERSION 3.11)

project(DMS
    VERSION 0.1
    DESCRIPTION "Driver Monitoring System v0.1"
    LANGUAGES CXX)

set(DLIB_SOURCE_DIR ~/dlib)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(DMS ${CMAKE_SOURCE_DIR}/dms/main.cpp)

add_subdirectory(${CMAKE_SOURCE_DIR}/lib)
add_subdirectory(${DLIB_SOURCE_DIR} dlib_build)

target_include_directories(DMS PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(DMS PUBLIC
    Threads::Threads
    dlib::dlib
    ${OpenCV_LIBS})

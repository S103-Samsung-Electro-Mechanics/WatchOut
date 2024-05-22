# cmake_minimum_required(VERSION 3.8.0)

# project(WatchOut
#     VERSION 0.1
#     DESCRIPTION "Driver Monitoring System v0.1"
#     LANGUAGES CXX)

# set(DLIB_SOURCE_DIR ${CMAKE_SOURCE_DIR}/dependencies/dlib)

# set(MEDIAPIPE_DIR ${CMAKE_SOURCE_DIR}/dependencies/mediapipe)
# set(MEDIAPIPE_DESKTOP_DIR ${MEDIAPIPE_DIR}/mediapipe/examples/desktop)
# set(MEDIAPIPE_DESKTOP_INCLUDE_DIRS ${MEDIAPIPE_DIR}/mediapipe/examples/desktop)
# set(MEDIAPIPE_DESKTOP_LIBRARIES ${MEDIAPIPE_DIR}/bazel-bin/mediapipe/examples/desktop/librun_graph_main_gpu.so)

# find_package(Threads REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
# find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

# add_executable(WatchOut
#     ${CMAKE_SOURCE_DIR}/watchout/main.cpp)

# add_subdirectory(${CMAKE_SOURCE_DIR}/lib)
# add_subdirectory(${DLIB_SOURCE_DIR}/dlib dlib_build)

# target_include_directories(WatchOut PUBLIC
#     ${CMAKE_SOURCE_DIR}/include
#     ${OpenCV_INCLUDE_DIRS}
#     ${Qt5Widgets_INCLUDE_DIRS}
#     ${MEDIAPIPE_DIR}
#     ${MEDIAPIPE_DESKTOP_INCLUDE_DIRS}
# )

# target_link_libraries(WatchOut PUBLIC
#     Threads::Threads
#     dlib::dlib
#     ${OpenCV_LIBS}
#     Qt5::Widgets
#     ${MEDIAPIPE_DESKTOP_LIBRARIES}
# )


cmake_minimum_required(VERSION 3.5)

project(DMS
    VERSION 0.1
    DESCRIPTION "Driver Monitoring System v0.1"
    LANGUAGES CXX)

set(DLIB_SOURCE_DIR ${CMAKE_SOURCE_DIR}/dependencies/dlib)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MEDIAPIPE_DIR ${CMAKE_SOURCE_DIR}/dependencies/mediapipe)
set(MEDIAPIPE_DESKTOP_DIR ${MEDIAPIPE_DIR}/mediapipe/examples/desktop)
set(MEDIAPIPE_DESKTOP_INCLUDE_DIRS ${MEDIAPIPE_DIR}/mediapipe/examples/desktop)
set(MEDIAPIPE_DESKTOP_LIBRARIES ${MEDIAPIPE_DIR}/bazel-bin/mediapipe/examples/desktop/librun_graph_main_gpu.so)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
    ${CMAKE_SOURCE_DIR}/watchout/main.cpp
    ${CMAKE_SOURCE_DIR}/mainwindow.cpp
    ${CMAKE_SOURCE_DIR}/mainwindow.h
    ${CMAKE_SOURCE_DIR}/mainwindow.ui
)

add_subdirectory(${DLIB_SOURCE_DIR}/dlib dlib_build)
# add_subdirectory(${CMAKE_SOURCE_DIR}/lib libDMS)

# add_executable(WatchOut ${CMAKE_SOURCE_DIR}/watchout/main.cpp)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(WatchOut
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
else()
    if(ANDROID)
        add_library(WatchOut SHARED
            ${PROJECT_SOURCES}
        )
    else()
        add_executable(WatchOut
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_include_directories(WatchOut PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${MEDIAPIPE_DIR}
    ${MEDIAPIPE_DESKTOP_INCLUDE_DIRS}
)

target_link_libraries(WatchOut PUBLIC
    Threads::Threads
    dlib::dlib
    ${OpenCV_LIBS}
    Qt5::Widgets
    ${MEDIAPIPE_DESKTOP_LIBRARIES}
    Qt${QT_VERSION_MAJOR}::Widgets
    # DMS
)

# set_target_properties(DMS PROPERTIES
#     MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
#     MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
#     MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
#     MACOSX_BUNDLE TRUE
#     WIN32_EXECUTABLE TRUE
# )

# install(TARGETS DMS
#     BUNDLE DESTINATION .
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(WatchOut)
endif()

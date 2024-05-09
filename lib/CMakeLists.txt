add_library(DMS_lib STATIC
	face_parser.cpp
	face_recognizer.cpp)

target_include_directories(DMS_lib PUBLIC
	${CMAKE_SOURCE_DIR}/include)
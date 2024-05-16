add_library(DMS STATIC
	face_parser.cpp
	face_recognizer.cpp)

target_include_directories(DMS PUBLIC
	${CMAKE_SOURCE_DIR}/include)
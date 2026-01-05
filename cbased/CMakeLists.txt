cmake_minimum_required(VERSION 3.10)
project(VideoCombine)

# Set C++ standard to C++11 or later
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV 4 REQUIRED)

# Add the executable (must come BEFORE target_include_directories)
add_executable(video_combine video_combine.cpp)

# Include OpenCV headers
target_include_directories(video_combine PUBLIC ${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries
target_link_libraries(video_combine PRIVATE ${OpenCV_LIBS})

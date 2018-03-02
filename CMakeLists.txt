cmake_minimum_required(VERSION 3.9)
project(HotCold)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)   # See below (1)

set(SOURCE_FILES main.cpp tsa.h 10000_1.000000_Request_Input.txt 86400_1.000000_Request_Input.txt)

add_executable(HotCold ${SOURCE_FILES})





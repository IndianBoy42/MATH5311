cmake_minimum_required(VERSION 3.19)

option(USE_LTO "Use LTO (idk how this works with ispc)" OFF)

project(MATH5312CPP)

add_executable(math)

enable_language(ISPC)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_C_EXTENSIONS ON)

target_include_directories(math PUBLIC ./inc/)

target_compile_options(math PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native -mtune=native -mavx -mavx2>)
target_link_libraries(math pthread m stdc++)
target_compile_options(math PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fPIE>)

if(USE_LTO)
target_compile_options(math PRIVATE -flto)
endif(USE_LTO)

target_sources(math PRIVATE main.cpp)
target_sources(math PRIVATE gausselim.ispc)
target_sources(math PRIVATE gausselim.cpp)
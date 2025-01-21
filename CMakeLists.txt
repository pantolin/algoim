cmake_minimum_required(VERSION 3.14)

project(algoim LANGUAGES CXX)

add_library(algoim INTERFACE)
add_library(algoim::algoim ALIAS algoim)

target_compile_features(algoim INTERFACE cxx_std_17)

target_include_directories(algoim
SYSTEM INTERFACE
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
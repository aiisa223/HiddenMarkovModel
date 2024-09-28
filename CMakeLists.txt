   cmake_minimum_required(VERSION 3.25)
   project(Markov)

   set(CMAKE_CXX_STANDARD 17)

   find_package(Python COMPONENTS Interpreter Development REQUIRED)
   find_package(pybind11 CONFIG REQUIRED)

   add_library(stock_hmm_lib STATIC stock_hmm.cpp)
   target_link_libraries(stock_hmm_lib PRIVATE pybind11::headers)

   add_executable(Markov main.cpp)
   target_link_libraries(Markov PRIVATE stock_hmm_lib pybind11::headers)

   pybind11_add_module(cpp_stock_hmm stock_hmm.cpp)
   target_link_libraries(cpp_stock_hmm PRIVATE pybind11::module)
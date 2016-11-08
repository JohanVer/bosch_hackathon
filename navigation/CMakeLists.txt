project(bosch_hackathon)
cmake_minimum_required(VERSION 2.8)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  image_transport
  cv_bridge
  sensor_msgs
  tf
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
  CATKIN_DEPENDS roscpp rospy std_msgs image_transport cv_bridge sensor_msgs tf
  DEPENDS system_lib
)

add_library(nav_lib
    src/bosch_navigation.cpp
    src/bosch_visualization.cpp
    include/bosch_navigation.h
    include/bosch_visualization.h
)

target_link_libraries ( nav_lib ${OpenCV_LIBS} ${catkin_LIBRARIES} )

include_directories(  ${catkin_INCLUDE_DIRS}  ${OpenCV_INCLUDE_DIRS}  include )

find_package( OpenCV 3.1 REQUIRED )

add_executable(run_navigation src/main.cpp include/main.h ${INCLUDE_GLOBAL_HEADERS})
target_link_libraries ( run_navigation nav_lib ${OpenCV_LIBS} ${catkin_LIBRARIES} )


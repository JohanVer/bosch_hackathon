project(bosch_vision)
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
  message_generation
  genmsg
)

#find_package(catkin REQUIRED genmsg)

add_message_files(
  FILES
  BoschVision.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES vision_lib
  CATKIN_DEPENDS roscpp rospy std_msgs image_transport cv_bridge sensor_msgs message_runtime message_generation
  DEPENDS system_lib Eigen
)

include_directories(  ${catkin_INCLUDE_DIRS}  ${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS}  include)

add_library(vision_lib
    src/bosch_vision.cpp
    include/bosch_vision.h
)

target_link_libraries ( vision_lib ${OpenCV_LIBS} ${catkin_LIBRARIES} )


find_package( OpenCV 3.1 REQUIRED )

add_executable(run_vision src/main.cpp include/main.h ${INCLUDE_GLOBAL_HEADERS})
target_link_libraries ( run_vision vision_lib ${OpenCV_LIBS} ${catkin_LIBRARIES} ${Eigen_LIBRARIES} )


cmake_minimum_required(VERSION 2.8.3)
project(liko)

SET(CMAKE_BUILD_TYPE "Debug")

ADD_COMPILE_OPTIONS(-std=c++14 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
# find_package(pinocchio REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include/IKFoM_toolkit
  include)

add_message_files(
  FILES
  Pose6D.msg
)
add_library(common_lib STATIC
    src/common_lib.cpp
)
target_include_directories(common_lib
    PUBLIC ${CMAKE_SOURCE_DIR}/include)
add_library(use-ikfom STATIC src/use-ikfom.cpp) 

target_include_directories(use-ikfom
    PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(particle_filter STATIC
    src/particle_filter.cpp
    src/IMU_Processing.cpp
)
target_link_libraries(particle_filter PUBLIC
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        Eigen3::Eigen 
        common_lib 
        particle_filter
        use-ikfom 
)
target_include_directories(particle_filter
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime sensor_msgs
  DEPENDS EIGEN3 PCL
  LIBRARIES particle_filter 
  INCLUDE_DIRS
)

add_executable(liko_se src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
target_link_libraries(liko_se PRIVATE particle_filter ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} )
target_include_directories(liko_se PRIVATE ${PYTHON_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} )



# DEBUG
set(CMAKE_BUILD_TYPE"Debug")
# target_compile_features(liko_se PUBLIC cxx_std_11)

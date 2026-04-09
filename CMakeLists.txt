cmake_minimum_required(VERSION 3.0.2)
project(p2_dduong3)

# Find all required packages based on p2a.py and p2b.py imports
find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib
  move_base_msgs
  geometry_msgs
  tf
  actionlib_msgs
  sensor_msgs
  cv_bridge
  message_filters
)

catkin_package()

# Mark the python scripts as executable and install them
catkin_install_python(PROGRAMS
  script/p2a.py
  script/p2b.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
cmake_minimum_required(VERSION 3.8)
project(swiftpro)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(controller_manager REQUIRED)

add_library(
  swiftpro
  SHARED
  src/swiftpro_hardware.cpp
  src/swiftpro/swiftpro.cpp
  src/swiftpro/serial_cobs.cpp
)
target_compile_features(swiftpro PUBLIC cxx_std_17)
target_include_directories(swiftpro PUBLIC
$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include/swiftpro>
)
ament_target_dependencies(
  swiftpro PUBLIC
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  std_msgs
  std_srvs
  sensor_msgs
)

pluginlib_export_plugin_description_file(hardware_interface swiftpro.xml)

install(
  DIRECTORY include
  DESTINATION include
)
install(TARGETS swiftpro
  EXPORT export_swiftpro
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY
  config  
  launch
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
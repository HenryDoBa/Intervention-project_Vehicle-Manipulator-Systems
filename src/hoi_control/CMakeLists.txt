cmake_minimum_required(VERSION 3.8)
project(hoi_control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install the Python package (all scripts under hoi_control/)
ament_python_install_package(${PROJECT_NAME})

# Install each node script so it is executable via ros2 run
install(PROGRAMS
  hoi_control/swiftpro_robotics_rrc.py 
  hoi_control/config.py             # All constants — topics, frames, gains, tolerances, geometry
  hoi_control/fsm_states.py         # State enum
  hoi_control/bezier.py             # _bezier, _bezier_vel, _bezier_cubic, _bezier_cubic_vel
  hoi_control/math_utils.py         # _angle_wrap (and any other pure helpers)
  hoi_control/camera_intrinsics.py  # CAMERA_MATRIX, DIST_COEFFS, image size constants
  hoi_control/aruco_detector.py     # ArucoDetectorMixin — _camera_cb + HUD overlay logic
  hoi_control/tf_helpers.py         # TFHelperMixin — _tf_pos, _tf_pose, _camera_to_world, _tf_ready, _update_vms_state
  hoi_control/controllers.py        # ControllersMixin — _vms_step, _vms_nav_step, _arm_step, _arm_bezier_step, _arm_step_direct, _task_priority_weighted
  hoi_control/fsm_handlers.py       # FSMHandlersMixin — all _run_* methods + _compute_align_target, _compute_floor_target
  hoi_control/rviz_markers.py       # RVizMarkersMixin — _publish_markers
  hoi_control/logging_utils.py      # LoggingMixin — _terminal_log
  hoi_control/lab2_pick_place_vms_node.py  # PickPlaceVMSNode(...mixins) + __init__ + main()
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 1. Resolve paths as strings immediately
    pkg_dir = get_package_share_directory('turtlebot_simulation')
    
    launch_file = os.path.join(pkg_dir, 'launch', 'turtlebot_basic.launch.py')
    scenario_file = os.path.join(pkg_dir, 'scenarios', 'turtlebot_hoi_circuit2.scn')

    # 2. Return the Description
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file),
            launch_arguments={
                'scenario_description': scenario_file
            }.items()
        )
    ])
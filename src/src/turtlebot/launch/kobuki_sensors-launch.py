import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from launch_ros.actions import PushRosNamespace


def generate_launch_description():

    robot_name_arg = DeclareLaunchArgument(
        'turtlebot_name',
        default_value='turtlebot',
        description='Name of the robot'
    )

    # RPLidar
    lidar_dict_args = {
        'serial_port':      ('Specifying usb port to connected lidar', '/dev/rplidar'),
        'serial_baudrate':  ('Specifying usb port baudrate to connected lidar', '115200'),
        'frame_id':         ('Specifying frame_id of lidar', 'rplidar'),
        #'frame_id':         ('Specifying frame_id of lidar', 'turtlebot/kobuki/rplidar'),
        'inverted':         ('Specifying whether or not to invert scan data', 'false'),
        'angle_compensate': ('Specifying whether or not to enable angle_compensate of scan data', 'true')
    }

    lidar_args = [
        DeclareLaunchArgument(
            arg_name, 
            default_value=LaunchConfiguration(arg_name, default=arg_default_value), 
            description=arg_description
        )

        for arg_name, (arg_description, arg_default_value) in lidar_dict_args.items()
    ]

    lidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[{
            arg_name: LaunchConfiguration(arg_name)
            for arg_name in lidar_dict_args.keys()
        }],
        output='screen'
    )

    # real sense camera
    rs_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        )
    )

    image_compressed = Node(
        package='image_transport',
        executable='republish',
        name='image_raw_to_compressed',
        arguments=[
            'raw', 'compressed',
            '--ros-args',
            '-r', 'in:=camera/camera/color/image_raw',
            '-r', 'out:=camera/camera/color/image_compressed'
        ],
    )

    return LaunchDescription([
        robot_name_arg,
        *lidar_args,
        lidar_node,
        rs_camera_launch,
        image_compressed
    ])

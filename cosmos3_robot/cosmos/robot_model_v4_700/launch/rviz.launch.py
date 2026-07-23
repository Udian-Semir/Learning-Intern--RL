import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
#--------------------------------
# 这个东西没啥用 不用管 rokae官方拿来用的
#--------------------------------
def generate_launch_description():
    package_name = 'robot_model'
    urdf_file_name = 'half_finish.urdf'
    rviz_config_file = 'half_finish.rviz'

    rviz_config_path = os.path.join(get_package_share_directory(package_name), 'rviz', rviz_config_file)
    urdf = os.path.join(
        get_package_share_directory(package_name),
        'urdf',
        urdf_file_name)
    
    # Read the URDF file content
    with open(urdf, 'r') as infp:
        robot_description_content = infp.read()

    # Wrap the URDF content in ParameterValue to specify it as a string parameter
    robot_description = ParameterValue(robot_description_content, value_type=str)

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    ))

    ld.add_action(Node(
        package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
    ))

    ld.add_action(Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    ))

    ld.add_action(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d',rviz_config_path]

    ))
    
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
    ))

    return ld

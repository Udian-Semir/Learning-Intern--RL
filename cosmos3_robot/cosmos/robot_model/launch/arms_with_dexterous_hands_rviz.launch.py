import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('robot_model')
    urdf_path = os.path.join(
        package_share, 'urdf', 'left_arm_with_wrist_camera.urdf')
    rviz_path = os.path.join(package_share, 'rviz', 'dual_hand.rviz')
    use_rviz = LaunchConfiguration('use_rviz')
    ros_domain_id = LaunchConfiguration('ros_domain_id')

    with open(urdf_path, 'r') as urdf_file:
        robot_description = ParameterValue(urdf_file.read(), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Start RViz for both AR5 arms with RH6 hands on their TCPs.'),
        DeclareLaunchArgument(
            'ros_domain_id', default_value='47',
            description='DDS domain used only by this robot visualization.'),
        SetEnvironmentVariable(
            name='ROS_DOMAIN_ID', value=ros_domain_id),
        SetEnvironmentVariable(
            name='ROS_LOCALHOST_ONLY', value='1'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen'),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_path],
            condition=IfCondition(use_rviz),
            output='screen'),
    ])

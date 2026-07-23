#!/usr/bin/env bash
# ROS setup scripts intentionally probe optional environment variables, so load
# them before enabling nounset mode.
set -eo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${PACKAGE_DIR}/../../.." && pwd)"
DOMAIN_ID="${1:-101}"

if ! [[ "${DOMAIN_ID}" =~ ^[0-9]+$ ]] || (( DOMAIN_ID > 232 )); then
  echo "ROS_DOMAIN_ID must be an integer from 0 to 232." >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${DOMAIN_ID}"
export ROS_LOCALHOST_ONLY=1

cd "${WORKSPACE_DIR}"
colcon build --base-paths "${PACKAGE_DIR}" --symlink-install
source "${WORKSPACE_DIR}/install/setup.bash"
set -u

cleanup() {
  if [[ -n "${launch_pid:-}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
    kill "${launch_pid}"
    wait "${launch_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ros2 launch robot_model left_arm_rviz.launch.py use_rviz:=false &
launch_pid=$!

sleep 2
ros2 run foxglove_bridge foxglove_bridge

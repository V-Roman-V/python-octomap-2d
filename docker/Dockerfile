FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# 1) Minimal runtime deps for your Python node
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-numpy python3-yaml python3-skimage \
    ros-humble-rclpy \
    ros-humble-tf2-ros ros-humble-tf2-py ros-humble-tf-transformations \
    ros-humble-rmw-cyclonedds-cpp \
 && rm -rf /var/lib/apt/lists/*

# 2) yacs from PyPI
RUN pip3 install --no-cache-dir yacs

# 3) App sources
WORKDIR /app
COPY . /app

# 4) Entrypoint (sources ROS, then runs CMD)
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["python3", "src/octomap_node.py"]

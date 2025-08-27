# python-octomap-2d

2D OctoMap in Python for ROS 2 — fast log-odds occupancy grid from PointCloud2 with occlusion-aware raycasting &amp; dynamic map growth.

The `python-octomap-2d` project is a ROS2-based module for generating and managing octomaps, which are 2D occupancy grids used for robotic navigation. 

## Implemented features

- log-odds logic
- occlusion-aware raycasting
- dynamic map growth

## Key Components

- **`config/configs.yaml`**: Configuration file for octomap parameters.
- **`src/`**: Contains Python scripts for octomap generation and processing:
  - `msg_converter.py`: Handles message conversions.
  - `octomap_node.py`: Main ROS2 node for octomap generation.
  - `octomap2d.py`: Converts 3D octomaps to 2D representations.
  - `preprocessor.py`: Preprocesses data for octomap generation.


## Getting Started

### Build the Docker Image

To build the Docker image for the `octomap_ros2` service, run:

```bash
docker-compose build
```

### Run the Octomap Node

To start the `octomap_ros2` service, use:

```bash
docker-compose up
```

This will launch the `octomap_node.py` script inside the container.

### Configuration

Modify the `config/configs.yaml` file to adjust octomap parameters as needed.

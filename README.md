# python-octomap-2d

The `python-octomap-2d` project is a ROS2-based module for generating and managing octomaps, which are 2D occupancy grids used for robotic navigation. 

The node subscribes to a point cloud, looks up the sensor origin via TF, integrates the scan into a **2D log-odds grid**, and publishes **two** maps:
- **Binary grid** (`-1` unknown / `0` free / `100` occupied)
- **Probability grid** (`-1` unknown, else `0..100`)
  
## Implemented features

- Log-odds updates with global clamping & occupancy threshold
- Occlusion-aware free-space carving (first-hit stopping, optional occluder clearing)
- Angle binning (closest point per bearing) for efficient ray stops  
- Dynamic canvas growth around the explored area  
- Preprocessing: Z-band, range/FOV filters, density outlier removal

## Topics

- **Subscribes:** `sensor_msgs/PointCloud2` on `io.input_topic` (see config)
- **Publishes:**  
  - Binary grid on `io.output_topic` (e.g., `/projected_map`)  
  - Probability grid on `io.output_prob_topic` (e.g., `/projected_map_prob`)

## Key Components

- **`config/configs.yaml`**: Configuration file for octomap parameters.
- **`src/`**: Contains Python scripts for octomap generation and processing:
  - `msg_converter.py`: Handles message conversions.
  - `octomap_node.py`: ROS 2 node (TF lookup, subscribers/publishers, timing)
  - `octomap2d.py`: core 2D mapper (log-odds + raycasting, dynamic growth)
  - `preprocessor.py`: Preprocesses data for octomap generation.
  - `docker/` + `docker-compose.yml` — containerized runtime

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

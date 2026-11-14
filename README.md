# Ghost hunt

A projector with the Airy on top is a torch: ghosts are fixed in the room and show up where the
beam lands, drifting slowly along the surfaces. Hold the reticle on one for 2 s to catch it. Pose comes from airy_slam's live SLAM.

```sh
bash ../airy_slam/scripts/start_slam.sh   # hold the rig still for the first seconds
./run.sh                                  # ROS bridge container + the game on the HDMI projector
```

The game runs on the host (python3-pygame, numpy, pyyaml) and draws through DRM/KMS at 720p,
no desktop needed; from a container SDL never reached the projector. ROS exists only in
containers, so `ghost_hunt/bridge.py` runs in one (airy_slam's foxbridge image) and passes
the pose and commands by UDP on localhost and the voxel map through `/dev/shm`. Turn the
projector's auto keystone off: the lens model assumes the plain, unwarped image.

Flow: 20 s of "scan the room" (the lidar map is projected back onto the room), then the hunt,
then a new round. Keys, if a keyboard is attached: `r` new round, `s` rescan, `m` map overlay,
`q` quit; `snap` saves the projected frame to `out/frame.png`. The same over ROS: `ros2 topic pub --once /ghost_hunt/cmd std_msgs/String "data: restart"`
(`rescan`, `map`).

Foxglove (airy_slam's bridge, fixed frame `odom`): `/ghost_hunt/markers` has the ghosts
(white, yellow while aimed at, green caught) and the projector's frustum; `/ghost_hunt/status`.

## Aligning the projector

`config.yaml` holds the projector's lens model and its pose on the Airy, and is re-read while
the game runs. `m` (or `show_map: true`) projects the lidar map back onto the room as a check.

To fit it, with a USB camera on the rig looking the same way (any roll), the game stopped and
the rig still, facing something with depth edges above the Airy's horizon (a shelf, a door frame):

```sh
PY=../project_map_defence/.venv/bin/python        # any python with OpenCV, scipy and the distro pygame
docker compose exec bridge bash -c '. /opt/ros/jazzy/setup.bash; python3 /game/tools/dump_scans.py 30'
cp /dev/shm/ghost_hunt_scans.npz out/calib/scans.npz
$PY tools/capture_graycode.py                     # stripes -> which projector pixel each camera pixel sees
$PY tools/calib_view.py out/calib/v.jpg 1.18 0 0.5 0 0 -0.08 0.4 -0.4 -8.2   # throw, offset xy, xyz, rpy
$PY tools/calib_fit.py 1.18 0.5 0.4 -0.4 -8.2     # local refinement: throw, offset y, rpy
```

`calib_view.py` draws the lidar's depth edges, seen through the model, on the camera photo
resampled into projector pixels: adjust the numbers until the edges sit on the objects, then
let `calib_fit.py` refine. The camera needs no calibration of its own. The lens offset is
assumed (0.5): over the narrow field it only trades against pitch.

## Without the sensor

```sh
docker compose up -d
docker compose exec -d bridge bash -c '. /opt/ros/jazzy/setup.bash && python3 /game/tools/fake_slam.py'
GHOST_HEADLESS=1 python3 -m ghost_hunt.game   # no projector: one frame a second to out/frame.png
```

## Layout

- `ghost_hunt/bridge.py` (container): pose (the last `/slam/odom`, turned on by the gyro samples
  since, so the picture follows the hand at IMU rate), the 10 cm voxel map from
  `/slam/scan_preview` (voxels seen in 4+ scans), markers for Foxglove.
- `ghost_hunt/link.py`: the host end of the link to the bridge.
- `ghost_hunt/game.py` (host): projector view, ghost placement on mapped surfaces, line of sight,
  catching, drawing.
- `tools/`: calibration (above) and `fake_slam.py`.

Everything is in FAST-LIO2's `odom` frame; rko_slam's loop-closure corrections are not applied.
The Airy sits dome-up and sees nothing below its own height, so the map, and the ghosts, are
all above the rig.

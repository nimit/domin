The purpose of this project is to generate datasets for training and testing of VLA models.

## Repository Overview

The contents of the folder `third-party/IsaacLab` refer to the Isaac Lab repository for using the Isaac Sim environment through code and Reinforcement Learning.

For any IsaacLab related documentation, you can use `third-party/IsaacLab` as a reference. Make sure whatever code you write is in accordance with the API there.

The code in `archive/` represents artifacts from an older version of code to generate datasets/infer with VLAs. It **is incomplete** and only here to provide a reference on how it was used.

The code in `src/` is a draft vectorized abstraction putting into code how simple I want to make the dataset generation. The file: `test_simulation.py` runs the simulation (to be abstracted into a "runner" later on).
IsaacLab provides a ManagerBased alternative too, but since this was already an abstraction I decided on using the direct method to have more control over the simulation.

My idea was that for every new task/scene, there should be two files. 

Example: `test_scene.py` & `test_dataset_config.py`.
The test scene would reproduce the environment while inference while `test_dataset_config` would define the dataset configuration to produce.

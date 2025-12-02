from pathlib import Path
import time
from dataclasses import dataclass, field

from .src.image_writer import safe_stop_image_writer
from .src.lerobot_dataset import LeRobotDataset
from .src.utils import build_dataset_frame, hw_to_dataset_features
from .src.control_utils import sanity_check_dataset_resume
import os

# DEFAULT_FEATURES = {
#     "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
#     "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
#     "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
#     "index": {"dtype": "int64", "shape": (1,), "names": None},
#     "task_index": {"dtype": "int64", "shape": (1,), "names": None},
# }

# (height, width, channels) for image
# FEATS = {
#     "observation.images.fixed_cam": {"dtype": "image", "shape": (480, 640, 3), "names": None},
#     "observation.joint_pos": {"dtype": "float32", "shape": (6, ), "names": None},
#     "action.joint_pos": {"dtype": "float32", "shape": (6, ), "names": None},

#     "observation.a_1": {"dtype": "float32", "shape": (1, ), "names": None},
#     "observation.a_2": {"dtype": "float32", "shape": (1, ), "names": None},
#     "observation.a_3": {"dtype": "float32", "shape": (1, ), "names": None},
#     "observation.a_4": {"dtype": "float32", "shape": (1, ), "names": None},
#     "observation.a_5": {"dtype": "float32", "shape": (1, ), "names": None},
#     "observation.a_6": {"dtype": "float32", "shape": (1, ), "names": None},
#     "action.a_1": {"dtype": "float32", "shape": (1, ), "names": None},
#     "action.a_2": {"dtype": "float32", "shape": (1, ), "names": None},
#     "action.a_3": {"dtype": "float32", "shape": (1, ), "names": None},
#     "action.a_4": {"dtype": "float32", "shape": (1, ), "names": None},
#     "action.a_5": {"dtype": "float32", "shape": (1, ), "names": None},
#     "action.a_6": {"dtype": "float32", "shape": (1, ), "names": None},

# }


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    default_task: str
    # names of all the joints
    joint_names: list[str]
    # A name and camera resolution in (width, height) tuple
    cameras: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False
    # Upload on private repository on the Hugging Face hub.
    private: bool = True
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable
    num_image_writer_threads_per_camera: int = 4

    resume_recording: bool = False

    robot_type: str = "SO100"

    def __post_init__(self):
        if self.default_task is None:
            raise ValueError(
                "You need to provide a task as argument in `default_task`."
            )


class DatasetRecord:
    def __init__(self, cfg: DatasetRecordConfig):
        print("DatasetRecord init called")
        self.cfg = cfg
        self.motor_features = {motor: float for motor in cfg.joint_names}
        self.camera_features = {
            cam: (res[1], res[0], 3) for cam, res in cfg.cameras.items()
        }
        self.observation_features = {**self.motor_features, **self.camera_features}
        self.action_features = {**self.motor_features}
        self.features = {
            **hw_to_dataset_features(
                self.observation_features, "observation", cfg.video
            ),
            **hw_to_dataset_features(self.action_features, "action", cfg.video),  # type: ignore
        }

        # print(f"Motor Features: {self.motor_features}")
        # print("=" * 45)
        # print(f"Camera Features: {self.camera_features}")
        # print("=" * 45)
        # print(f"Observation Features: {self.observation_features}")
        # print("=" * 45)
        # print(f"Action Features: {self.action_features}")
        # print("=" * 45)
        # print(f"Features: {self.features}")

        self.current_task = None

        if cfg.resume_recording and os.path.exists(cfg.root):
            self.dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
            if cfg.cameras:
                self.dataset.start_image_writer(
                    num_processes=cfg.num_image_writer_processes,
                    num_threads=cfg.num_image_writer_threads_per_camera
                    * len(cfg.cameras),
                )
            sanity_check_dataset_resume(
                self.dataset, cfg.robot_type, cfg.fps, self.features
            )
        else:
            self.dataset = LeRobotDataset.create(
                cfg.repo_id,
                cfg.fps,
                root=cfg.root,
                robot_type=cfg.robot_type,
                features=self.features,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=cfg.num_image_writer_threads_per_camera
                * len(cfg.cameras)
                if cfg.cameras
                else 0,
            )

        self.rerecord_count = 0

    def __enter__(self):
        print("Started Recording")
        now = time.perf_counter()
        self.recording_start = now
        self.episode_start = now
        self.episode_num = 0
        self.steps = 0

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exited context manager....")
        now = time.perf_counter()
        self.recording_end = now
        self._finish_recording()

        if exc_type:
            print(f"Exception: {exc_type}, {exc_value}")
            print(traceback)

        return False

    def new_epsiode(self, task: str | None = None):
        self.current_task = task
        if self.steps == 0:  # type: ignore
            print(
                f"Changed current task to {task}. No step recorded; not creating a new episode"
            )
            return
        print(f"Saving episode (idx: {self.episode_num}/re: {self.rerecord_count})")
        self.dataset.save_episode()
        self.episode_start = time.perf_counter()
        self.episode_num += 1
        self.steps = 0

    def rerecord(self):
        if self.dataset.image_writer is not None:
            self.dataset.image_writer.wait_until_done()
        self.dataset.clear_episode_buffer()
        self.steps = 0
        self.rerecord_count += 1
        self.episode_start = time.perf_counter()
        print(
            f"\n\nRerecording current episode (idx: {self.episode_num} | re: {self.rerecord_count})"
        )

    def _finish_recording(self):
        print("Finishing recording")
        self.dataset.save_episode()

        if self.cfg.push_to_hub:
            self.dataset.push_to_hub(tags=self.cfg.tags, private=self.cfg.private)

    # observation = {images: {}, motors: []}
    # action = {motors: []}
    @safe_stop_image_writer
    def step(self, motor_obs, action, cam_obs={}):
        joint_names = self.cfg.joint_names

        assert len(cam_obs) == len(self.cfg.cameras), (
            f"{len(cam_obs)} != {len(self.cfg.cameras)}"
        )
        assert len(motor_obs) == len(joint_names), (
            f"{len(motor_obs)} != {len(joint_names)}"
        )
        assert len(action) == len(joint_names), f"{len(action)} != {len(joint_names)}"

        observation = {**{x[0]: x[1] for x in zip(joint_names, motor_obs)}, **cam_obs}
        action = {x[0]: x[1] for x in zip(joint_names, action)}

        observation_frame = build_dataset_frame(
            self.features, observation, prefix="observation"
        )
        action_frame = build_dataset_frame(self.features, action, prefix="action")
        frame = {**observation_frame, **action_frame}
        self.dataset.add_frame(
            frame,
            task=self.current_task
            if self.current_task is not None
            else self.cfg.default_task,
        )
        self.steps += 1

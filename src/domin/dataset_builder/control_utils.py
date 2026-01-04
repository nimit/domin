from deepdiff import DeepDiff
from .utils import DEFAULT_FEATURES
from .lerobot_dataset import LeRobotDataset


def sanity_check_dataset_resume(
    dataset: LeRobotDataset, robot_type: str, fps: int, features: dict
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(
            dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"]
        )
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n"
            + "\n".join(mismatches)
        )

# Dataset Builder

The `dataset_builder` package provides tools for recording and managing robotic manipulation datasets in the LeRobot format. It is designed to support high-throughput simulation environments with simultaneous recording capabilities.

## Major Features

### 1. Simultaneous Episode Recording
Record multiple episodes (environments) in parallel. The `DatasetRecord` class now accepts `num_envs` in its configuration.
-   **Batched Inputs**: The `step()` method expects batched `torch.Tensor` inputs of shape `(num_envs, ...)` for both observations and actions.
-   **Efficient Management**: Handles multiple active episode buffers concurrently.

### 2. Story Mode
Episodes are grouped into "stories" (batches).
-   **`new_story()`**: Starts a new batch of episodes. This is useful for resetting environments and starting fresh recording cycles simultaneously.
-   **Automatic Indexing**: Manages episode indices automatically across stories.

### 3. Scheduled Re-recording
Robust handling of failed episodes.
-   **`rerecord(env_idx)`**: If an episode fails, calling this method clears its current buffer and schedules it for a retry in the *next* story.
-   **Index Preservation**: The re-recorded episode retains its original episode index, ensuring no gaps in the dataset.

### 4. Metadata & Custom Metrics
-   **`save_metadata(key, value)`**: Save arbitrary metadata (e.g., success rates, simulation metrics) directly to the dataset's `info.json`.
-   **Automatic Stats**: Computes and saves episode statistics automatically.

### 5. LeRobot Format Compatibility
-   Produces datasets compatible with Hugging Face LeRobot.
-   Saves data in Parquet format with embedded images.
-   **Raw Images**: Option to preserve raw image files during recording for debugging or other pipelines.

## Changes from Previous Version

-   **Multi-Environment Support**: `DatasetRecord` was refactored from single-episode to multi-episode management.
-   **Input Standardization**: All inputs to `step` must now be `torch.Tensor`.
-   **`robot_type` Required**: The `robot_type` argument in `DatasetRecordConfig` is now mandatory.
-   **Re-record Logic**: Changed from immediate reset to scheduled retry in the next story to maintain batch synchronization.
-   **Image Handling**: Improved robustness of image saving. Raw images are now preserved by default (deletion disabled) to aid debugging.

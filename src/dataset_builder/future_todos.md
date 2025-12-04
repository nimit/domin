# Future Dataset Builder TODOs

- [ ] Offload `dataset.step` and saving operations to a background thread/process for efficiency
- [ ] `src/dataset_builder/src/lerobot_dataset.py:337`: implement sanity check for features
- [ ] `src/dataset_builder/src/lerobot_dataset.py:632`: implement faster transfer
- [ ] `src/dataset_builder/src/lerobot_dataset.py:670`: hf_dataset.set_format("torch")
- [ ] `src/dataset_builder/src/lerobot_dataset.py:1101`: Merge this with OnlineBuffer/DataBuffer
- [ ] `src/dataset_builder/src/image_writer.py:42`: handle 1 channel and 4 for depth images
- [ ] `src/dataset_builder/src/utils.py:489`: Implement "type" in dataset features and simplify this

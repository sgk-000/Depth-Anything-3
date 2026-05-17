## Overlapping Inference
Please update `da3_kitti.py` to improve the inference-window strategy so that poses are also predicted around the boundaries between inference windows.

Currently, inference is performed using fixed windows. For example, if `window_size = 150`, the script runs inference on frames `0–150`, `150–300`, and so on. This can miss or degrade predictions near the boundary between windows.

Please modify the logic so that the script also runs overlapping windows. For example, when `window_size = 150`, it should run inference on:

- `0–150`
- `75–125`
- `150–300`
- `125–175`
- ...

In other words, use a stride of `window_size // 2` so that each boundary region is covered by an additional overlapping inference window.

Requirements:

1. Update the inference-window generation logic in `da3_kitti.py`.
2. Ensure that overlapping windows are correctly handled near the start and end of each KITTI Raw sequence.
3. Save only the poses predicted by the model.
4. Do not save intermediate outputs such as depth maps, point maps, confidence maps, or visualization results unless they are already explicitly required by the existing script.
5. Make sure the saved pose format remains compatible with the current downstream code.
6. Avoid duplicated or inconsistent pose entries when multiple overlapping windows predict the same frame pair.
7. Add clear comments explaining how overlapping windows are generated and how duplicate predictions are handled.
# Motion_Pending head-pose predictor

Team Motion_Pending, 9th ABC Challenge. Given the last 4 s of head orientation (400 quaternions at 100 Hz), the model
predicts the next 20 samples (200 ms). Architecture, size and training are in [`model_card.md`](model_card.md).

| File | What it is |
| --- | --- |
| `model_fp32.onnx` | The submission (primary file). Opset 17, stateful, see [Streaming state](#streaming-state) |
| `run_inference.py` | `load_model` / `predict` and the command line the challenge asks for |
| `requirements.txt` | Pinned dependencies: `numpy` and `onnxruntime` (`2.2.6` and `1.31.0` on Python 3.11 to 3.13) |
| `sample_input.npy` | A synthetic 400-sample trace for checking the frame conversion (not BOXRR data) |
| `sample_expected_output.npy` | What `run_inference.py` returns for `sample_input.npy` |
| `model_card.md` | Architecture, parameters, MACs, state, training data, failure modes |
| `Motion_Pending_ABC_Challenge_2027.pdf` | Our paper, "Symmetry in Motion: A Compact Network for Head-Pose Prediction" (IJABC format) |
| `weights/` | The PyTorch checkpoint and the training record it came from |

## Quick start

Python 3.10 to 3.14. Linux, macOS or Windows. `requirements.txt` pins one version of each package per Python version
(Python 3.10 gets onnxruntime 1.23.2, Python 3.14 gets numpy 2.4.6). No network access is needed after installing.

```bash
pip install -r requirements.txt
python run_inference.py --model_dir . --input sample_input.npy --output pred.npy
```

From Python:

```python
import numpy as np
from run_inference import load_model, predict

model = load_model(".")                      # once per session
context = np.load("sample_input.npy")        # float32 [400, 4], (w, x, y, z), oldest sample first
prediction = predict(model, context)         # float32 [20, 4], (w, x, y, z); row 19 is +200 ms
```

`predict` returns **`[20, 4]`**. The ONNX graph itself outputs `prediction` as `[1, 20, 4]`. The model runs on the CPU
execution provider on purpose. One streaming step is about 16k multiply-adds, so a GPU would not make it faster, and
the same requirements install cleanly on a CUDA machine. We tested a clean install with the network switched off on
Linux under Python 3.10, 3.11, 3.12, 3.13 and 3.14, and on Ubuntu 24.04 with a GPU attached.

## Input frame and axis convention

The model was trained on BOXRR-23 head rotations exactly as the dataset records them. We only reordered them to
scalar-first. Please convert your test stream into this convention:

| | Our convention |
| --- | --- |
| Quaternion order | **(w, x, y, z)**, scalar first, unit length. Hamilton product |
| What q means | **Head → world**: `v_world = R(q) · v_head` |
| World up | **+y**, aligned with gravity (opposite to gravity) |
| Handedness | **Left-handed** (Unity's convention) |
| Head axes | **+x right, +y up (top of the head), +z forward (where the face points)** |
| World heading | Arbitrary. The model does not depend on which way +z points in the world |
| Pitch and roll | **Keep them as tracked.** Do not re-centre or level them: the model uses the real tilt relative to gravity |
| Rate | About 100 Hz, oldest sample first. Unit length, hemisphere-aligned as the challenge harness provides |

Only the heading (yaw about world +y) is free. A world frame with a different up axis or the opposite handedness
changes the input the model sees. On the submitted model, streamed over 20 test users (931,051 windows), a z-up world
raised the error at 200 ms from 4.18° to 5.01° (+0.83°) and a handedness flip to 4.34° (+0.16°).

### How we checked it

The convention above was measured from raw BOXRR-23 recordings (Beat Saber, Quest 2 headset and controllers), not
assumed:

- **Up axis:** world +y expressed in the head frame has a median of (0.008, 0.981, −0.120). The head is level to
  within about 7°, slightly looking down.
- **Forward:** in Beat Saber the notes always come from one direction. The head's +z points along world +z in every
  recording (median (−0.02, −0.12, 0.98)).
- **Right and handedness:** in the head frame the right controller is on the +x side and the left on −x in all 40
  recordings we checked. Both are almost always below the eyes and in front of them. So x is right, y up and z
  forward, which is left-handed.
- **Head → world:** fitting the headset as a point on a head turning about the neck puts the eyes 13.8 cm above and
  7.8 cm in front of the pivot when q is read as head → world. Read as world → head, they would sit below and behind
  the neck.

### Worked examples

| Head pose | q = (w, x, y, z) | Face direction in the world | Top of the head in the world |
| --- | --- | --- | --- |
| Level, looking ahead | (1, 0, 0, 0) | (0, 0, 1) | (0, 1, 0) |
| Looking down 30° | (0.9659, 0.2588, 0, 0) | (0, −0.5, 0.866) | (0, 0.866, 0.5) |
| Looking up 30° | (0.9659, −0.2588, 0, 0) | (0, 0.5, 0.866) | (0, 0.866, −0.5) |
| Turned 90° to the right | (0.7071, 0, 0.7071, 0) | (1, 0, 0) | (0, 1, 0) |
| Head tilted right 20° (right ear down) | (0.9848, 0, 0, −0.1736) | (0, 0, 1) | (0.342, 0.940, 0) |
| Turned right 45°, then looking down 30° | (0.8924, 0.2391, 0.3696, −0.0990) | (0.612, −0.5, 0.612) | (0.354, 0.866, 0.354) |

Looking down is a positive rotation about the head's +x (right) axis, and turning right is a positive rotation about
world +y. Both follow from the left-handed frame.

### Converting from a right-handed frame

For a right-handed frame with **+x right, +y up, −z forward** (OpenXR, OpenGL), mirror z:

```
q_ours = (w, −x, −y, z)
```

For any other frame, let `c_w` be the rotation that converts world coordinates from your frame to ours (+y up), and `c_h`
the one that converts head coordinates from yours to ours. Then use `q_ours = c_w ⊗ q_yours ⊗ c_h⁻¹`. A handedness change is a mirror, not a rotation, so do
that step first with a sign flip like the one above. Check the result against the worked examples.

### Sample file

`sample_input.npy` is a made-up trace in our convention (no BOXRR data). The head is pitched 10° down and turns right
about world +y at a steady 45°/s for 4 s. The first sample is (0.9962, 0.0872, 0, 0), looking 10° down along world
+z. The last is (0.0039, 0.0003, 0.9962, −0.0872), turned 179.6°. If you build the same motion in your own frame and
convert it, you should get this array back. `sample_expected_output.npy` is the model's prediction for it. The forecast
continues the right turn and keeps the 10° pitch.

```bash
python run_inference.py --model_dir . --input sample_input.npy --output pred.npy
python -c "import numpy as np; print(abs(np.load('pred.npy') - np.load('sample_expected_output.npy')).max())"
```

The difference should be 0 or within float rounding.

## Streaming state

`model_fp32.onnx` is stateful, using the challenge's naming. The inputs are `context` `[1, 400, 4]`, `state_in_0`
`[1, 32, 2, 7]`, `state_in_1` `[1, 32, 2, 56]` and `state_in_2` `[1, 32, 2, 329]`. The outputs are `prediction`
`[1, 20, 4]` and `state_out_0` to `state_out_2`. All tensors are float32 with static shapes. The state holds the network's own history (dilated
convolution buffers) and comes only from past samples of the same session. Start each session at zero and feed
`state_out_i` back as `state_in_i` on every call. The full 400-sample window still goes into `context` every time.
`model_card.md` gives the warm-up.

`run_inference.py` does this for you. It keeps the state on the model object. When a call's window is the previous
window shifted by one sample, it runs one streaming step (about 0.3 ms). Otherwise it treats the window as a new
session: it resets the state and replays the window from its first sample (about 35 ms), so every call gives the same
answer whatever came before.

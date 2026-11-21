window.PLAIN = {
 "input": {
  "title": "Input: recent head motion",
  "oneLine": "The model's input: the last 4 s of headset orientation, sampled 100 times a second.",
  "simple": "In: the headset reports its orientation as a quaternion (four numbers, w x y z, that describe a 3D rotation), one sample every 10 ms. The window holds 400 samples. On each call (one run of the model, made for every new sample) the model reads only the 2 newest. Everything older is already summarised in the blocks' memory, which you meet at Block 1. Out: the 2 newest poses, passed to the features.",
  "why": "Reading only the 2 newest samples keeps the work the same on every call, however long the history is.",
  "numbers": [
   "4 seconds of history at 100 Hz",
   "400 orientation samples in total",
   "Only 2 newest samples read per call"
  ]
 },
 "feat": {
  "title": "Features: 7 motion measurements",
  "oneLine": "Turns the 2 newest poses into 7 physical measurements, with no learned weights.",
  "simple": "In: the 2 newest poses from the input. A pose says where the head is, not how it is moving. So fixed geometry works out 7 features (the measurements the model works from): the turning rate about 3 axes (ωx, ωy, ωz), the overall turning speed, and the direction of gravity as seen from the head (gx, gy, gz). Out: 7 numbers for this sample, plus a mirrored copy of them for the second stream (see Mirror).",
  "why": "Working these physical values out by hand gives the model the geometry it needs with 0 learned parameters.",
  "numbers": [
   "7 motion numbers per sample",
   "0 learned parameters",
   "3 turning rates, 1 speed, 3 gravity components"
  ]
 },
 "proj": {
  "title": "Projection: 7 features to 32 channels",
  "oneLine": "Mixes the 7 features into 32 channels, the signals the rest of the model works on.",
  "simple": "In: the 7 features for this sample. Each of the 32 channels (the model's internal signals) is a weighted sum of the 7 features plus a bias, with negative results set to zero (ReLU). The weights are learned. Out: 32 channel values that describe combinations of turn rate, speed and tilt at this instant. Change over time comes later, from the blocks. The same weights serve both streams.",
  "why": "32 channels give the later layers room to separate head-movement patterns that 7 numbers alone would not.",
  "numbers": [
   "Turns 7 numbers into 32 channels",
   "256 parameters across weights and biases",
   "Shared by both streams"
  ]
 },
 "blk0": {
  "title": "Block 1 · short range",
  "oneLine": "Reads the last 8 samples (0.08 s) to catch sudden changes.",
  "simple": "In: the projection's 32 channels. A block looks back in time through 8 taps (a tap is one sample that it reads). Block 1's taps are 1 sample apart: now and the 7 before it. For each channel it takes a weighted sum over the taps, then mixes the channels, within a stream and across the mirror, and adds its input back on (a skip connection). Out: 32 channels, each summarising the last 8 samples. The 7 older inputs wait in its memory, so a call only processes the newest.",
  "why": "Closely spaced taps follow fast motion, so the model can respond quickly to a sudden change in head direction.",
  "numbers": [
   "8 taps, 1 sample apart",
   "Sees 8 samples (0.08 s)",
   "2,336 parameters, 7-step memory"
  ]
 },
 "blk1": {
  "title": "Block 2 · medium range",
  "oneLine": "The same block with taps 8 samples apart: looks back 0.64 s to follow turns already under way.",
  "simple": "In: Block 1's output. Block 2 is built like Block 1, but its 8 taps are 8 samples apart, so its own taps span 57 samples. Each tap reads a Block 1 output that already summarises 8 samples, so one Block 2 output sees 64 samples. Out: 32 channels summarising the last 0.64 s. Its memory holds 56 samples, so nothing older is recomputed.",
  "why": "Spacing the taps costs no extra weights or multiply-adds, only a longer memory (56 samples instead of 7).",
  "numbers": [
   "8 taps, 8 samples apart",
   "Sees 64 samples with block 1 (0.64 s)",
   "2,336 parameters, 56-step memory"
  ]
 },
 "blk2": {
  "title": "Block 3 · long range",
  "oneLine": "The same block with taps 47 samples apart: covers nearly the whole 4 s window.",
  "simple": "In: Block 2's output. Block 3's 8 taps are 47 samples apart, so its own taps span 330 samples. Each tap reads a Block 2 output that already summarises 64 samples, so one Block 3 output sees 393 samples, nearly the whole 4 s window. Out: 32 channels per stream, which the readout reads. Its memory holds 329 samples.",
  "why": "Seeing 393 samples of history with no gaps lets the model anticipate where a long head turn will finish.",
  "numbers": [
   "8 taps, 47 samples apart",
   "Sees 393 samples with blocks 1 and 2 (3.93 s)",
   "2,336 parameters, 329-step memory"
  ]
 },
 "head": {
  "title": "Readout: 9 coefficients",
  "oneLine": "Turns Block 3's output into 9 coefficients that shape a smooth correction curve.",
  "simple": "In: Block 3's output for now, 32 channels from each stream (2 × 32 numbers). Instead of predicting 20 future poses one by one, the readout produces 9 coefficients: 3 for each rotation axis (x, y, z). The 3 set how much of a straight line, a squared curve and a cubed curve go into that axis's correction. Out: the 9 coefficients. The trajectory turns them into a curve that bends, speeds up and settles smoothly.",
  "why": "9 coefficients need 291 parameters instead of 1,940 for 60 separate outputs, about 85% fewer, at the same accuracy when tested on an earlier version of the model (v6).",
  "numbers": [
   "Turns 2×32 numbers into 9 coefficients",
   "291 parameters instead of 1,940 for 60 outputs",
   "Same accuracy as 60 outputs when tested on v6 (+0.001°)"
  ]
 },
 "traj": {
  "title": "Trajectory: prior + learned residual",
  "oneLine": "Adds the learned residual to the zero-parameter damped-velocity prior, for each of the next 20 steps.",
  "simple": "In: the latest turning rate ω from the features, and the readout's 9 coefficients. First the damped-velocity prior (a prior is a starting guess that needs no learning): the head keeps turning at its current rate while gently slowing down. Then the learned residual (the correction to that guess): the smooth cubic curve set by the 9 coefficients. Out: their sum r, the turn the model expects from now to each of the next 20 steps.",
  "why": "Physical inertia supplies the baseline motion, so the learned part only has to make small steering adjustments.",
  "numbers": [
   "Damped-velocity prior: 0 parameters",
   "Slows down with a 100 ms time constant",
   "Alone it scores 5.11°, versus 6.04° for no change (20 test users)"
  ]
 },
 "out": {
  "title": "Forecast: 20 future poses",
  "oneLine": "Turns the 20 expected turns into 20 future head orientations, 10 ms apart.",
  "simple": "In: the trajectory's 20 turns and the current pose. Each turn is converted to a rotation and applied to the current pose, which gives one predicted orientation per step. Every step is built directly from now, not from the step before it, so errors do not compound. Out: 20 poses covering the next 200 ms. The challenge scores only the 20th.",
  "why": "A chained forecast would carry each step's error into the next; working every step out from the present orientation avoids that.",
  "numbers": [
   "20 future orientations at 100 Hz",
   "10 ms spacing up to 200 ms",
   "Scored strictly at the 20th sample"
  ]
 },
 "gaze": {
  "title": "Head direction: the forecast as a view",
  "oneLine": "Shows the same forecast as the direction the headset faces, drawn on a sphere.",
  "simple": "In: the 20 predicted poses, and what the head really did. The headset in the middle turns exactly as the real one did. Each dot is where its forward direction meets the sphere; the Key on the right gives the colours. This is head direction, not eye gaze. The dots and arrows only show where the face points. The scored error also counts roll (tilting the head sideways), drawn as small ticks at the arrow tips, so it can be larger than the gap between the tips.",
  "why": "A direction on a sphere is easier to read than the four numbers of a quaternion.",
  "numbers": [
   "grey: past 400 ms",
   "blue: predicted, next 200 ms",
   "green: true path",
   "ticks: headset up (roll)"
  ]
 },
 "mirror": {
  "title": "Mirror: two streams, one set of weights",
  "oneLine": "Runs the motion and its left-right mirror image through exactly the same weights.",
  "simple": "The design assumes a left turn and its mirror-image right turn follow the same rules, and builds that into the weights. In: the 7 features. A mirrored copy is made by flipping the sign of three of them (ωy, ωz, gx). The original is the observed stream, h⁰, above the mirror plane; the copy is the mirrored stream, h¹, below it. Both pass through identical weights in every later layer. Out: two streams of 32 channels.",
  "why": "Sharing the weights halves the parameters at the same width. On an earlier version of the model (v6), removing the mirroring was also slightly less accurate (by 0.029°).",
  "numbers": [
   "Halves the weights: 15,113 to 7,555",
   "2 streams, one set of weights",
   "Flips ωy, ωz and gx"
  ]
 },
 "reach": {
  "title": "Reach: how far back one output sees",
  "oneLine": "Stacked, the three blocks let one output see 393 samples, nearly the whole 4 s.",
  "simple": "This step adds no new part. It shows the receptive field (the stretch of history that one output depends on) of the three blocks together. Seen from above, the newest output of Block 3 branches into its 8 taps on Block 2, each of those into 8 taps on Block 1, and each of those into 8 on the projection: 512 paths. A path's lag (how far back it ends) is a + 8b + 47c samples. That covers every sample from now back to 392 with no gaps. Where Block 3's spacing of 47 overlaps Block 2's span of 64, some lags are reached twice.",
  "why": "Spacing the taps reaches 3.93 s of history with the same 2,336 weights per block; only the memories get longer.",
  "numbers": [
   "393 feature steps = 394 raw poses",
   "512 paths, 393 distinct lags",
   "119 lags reached twice, 0 gaps"
  ]
 },
 "overview": {
  "title": "Model overview",
  "oneLine": "A compact neural network that forecasts head orientation for virtual reality systems, trained on recordings from 5,243 players.",
  "simple": "Left to right you move through the model, from input to forecast. Up and down are its 32 channels (internal signals): the observed stream h⁰ above the mirror plane and its mirror image, the stream h¹, reflected below. Into the screen is the past; the 200 ms forecast comes toward you. Press Next to build it one part at a time.",
  "why": "Small and fast by design: about 78 microseconds per prediction (median, one ONNX Runtime thread on an AMD Ryzen 7 5800X3D desktop CPU; relative only, the organisers time it on XR hardware).",
  "numbers": [
   "7,555 parameters (7,552 stored + 3 folded)",
   "44,196 bytes ONNX file",
   "≈78 µs median on a Ryzen 7 5800X3D CPU"
  ]
 }
};

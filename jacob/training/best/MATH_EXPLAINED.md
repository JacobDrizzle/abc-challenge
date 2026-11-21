# How the model predicts, in plain English

This explains the maths behind k8_combined (the model in `best/`) without assuming a maths background. It covers what
the prior is, how one prediction is made, how the model is trained, and the jargon used in the code and the paper.
The mirror equivariance and gravity inputs are explained in `MIRROR_AND_GRAVITY_EXPLAINED.md`. Written 7 Nov 2026.

## What the prior is

The prior is the model's built-in starting guess: a simple rule for where the head will go, written by hand rather
than learned. The network then only has to learn how to adjust that guess.

**The word.** "Prior" comes from statistics. It's what you believe before you look at the evidence. In our model it's
the assumption built in before any training: the guess you'd make about head motion with no learning at all.

**Our prior in one sentence.** The head keeps turning the way it's turning now, but slows down smoothly until it
stops. Think of a car when the driver takes their foot off the accelerator: it keeps rolling but gradually slows. Our
prior assumes the head does the same. It loses about 10% of its speed every 10 ms, so after 100 ms about a third of
the speed is left.

**Why that rule.** Heads move in quick bursts: you turn to look at something, then stop. So at any moment the best
simple guess is "it'll carry on a little, then settle". Here are the three simple guesses we compared, as validation
error at 200 ms on the 100-user data:

| Guess | What it assumes | Error |
|---|---|---:|
| Hold last | The head freezes where it is | 6.08° |
| Constant velocity | The head keeps turning at the same speed forever | 6.71° |
| **Damped velocity (our prior)** | It keeps turning but slows down | **5.05°** |

Constant velocity is the worst of the three, because heads always stop and it never does. Damped velocity sits between
the other two and beats both.

**What it does in the model.**
1. The prior makes its guess from the head's current turning speed.
2. The network looks at the last 4 seconds of movement and adds a correction: "this player usually turns further
   than that", or "they're slowing down faster than usual".
3. Prior plus correction is the prediction.

Before training, the correction is zero, so the model *is* the prior. Training teaches the network only what the prior
gets wrong.

**Things worth knowing.**
- It has no learned weights. There's only one setting, the 100 ms slow-down time. It was picked by trying several
  values (25 to 500 ms) on validation data and keeping the best. It's the same for every player, so it can't memorise
  the training users.
- It's free at run time. In the deployed model it's one extra column in a calculation that already happens (see
  MatMul below).
- The network could manage without it. In the leave-one-out run (`experimental/v10_loo`), replacing the prior with
  "hold last" cost about 0.015° on seed 42 (4.262° against 4.247°); seeds 43 and 44 were still to come on 7 Nov. The
  network learns most of what the prior knows anyway. The prior is a sensible head start rather than the main source
  of accuracy.

The rest of this file shows the prior's formula (Step 2 below) and how it fits with everything else.

## Building blocks

**Quaternion (q).** Four numbers (w, x, y, z) that describe which way the head is pointing. They have to have
"length 1": the square root of w² + x² + y² + z² must equal 1. Only then is it a valid orientation. Turning any four
numbers into a valid quaternion by scaling them to length 1 is called **normalising**. One quirk: q and −q (all four
signs flipped) describe the same orientation.

**Rotation vector (r).** Three numbers that describe a turn, as an arrow. The arrow's direction is the axis to turn
about, and its length is how far to turn. For example, (0, 0.1, 0) means "turn 0.1 radians about the up axis", which
is looking left by about 5.7°. Rotation vectors are handy because small turns can simply be added, like adding arrows.

**Radians.** The unit maths uses for angles. 1 radian ≈ 57.3°, so a full circle is 2π ≈ 6.28 radians. The challenge
scores in degrees.

**⊗ (quaternion multiplication).** "Do one turn, then another." `q_last ⊗ turn` means: start at the current head pose,
then apply the turn. The order matters. Written this way, the turn is measured from the head's own point of view, so
"left" means the player's left.

**conj (conjugate).** "Undo" a turn. So `conj(q[t−1]) ⊗ q[t]` means "undo where the head was, then go to where it is
now". That gives the small turn the head made between the last two samples.

## One prediction, step by step

Example: a player is turning their head left at 100° per second. At 100 samples a second, that's 1° per sample.

**Step 1: the rate.** From the last two samples the model measures the turn per sample: 1° about the up axis. The
deployed model uses a shortcut here, sin θ instead of θ ("native maps"). For angles this small the two are
practically identical: at 3° per sample they differ by 0.05%.

**Step 2: the damped prior, rate × τ(1 − e^(−k/τ)).** This answers: "If the head keeps turning but gradually slows
down, how far will it have turned after k more samples?"
- **e^(−k/τ)** is "the fraction of the speed left after k samples". With τ (tau) = 10 samples (100 ms), the speed
  fades by about 10% each sample. After 10 samples about 37% is left, and after 20 about 14%.
- **τ(1 − e^(−k/τ))** adds up all that fading speed: the total distance covered, counted in "samples' worth at the
  current speed".

For the player turning 1° per sample:

| Samples ahead (k) | Hold last | Damped prior | Constant velocity |
|---:|---:|---:|---:|
| 1 (10 ms) | 0° | 0.95° | 1° |
| 5 (50 ms) | 0° | 3.9° | 5° |
| 10 (100 ms) | 0° | 6.3° | 10° |
| 20 (200 ms, scored) | 0° | 8.6° | 20° |

Hold last assumes the head stops dead. Constant velocity assumes it never slows, which overshoots. The damped prior
sits in between.

**Step 3: the correction curve, c₁(k/20) + c₂(k/20)² + c₃(k/20)³.** The network outputs 9 numbers: three "knobs"
(c₁, c₂, c₃) for each of the three axes (nod, turn, tilt). k/20 runs from 0.05 at the first future sample to 1 at the
20th.
- c₁ × (k/20) grows in a straight line. It's like adjusting the speed.
- c₂ × (k/20)² starts slowly and bends upward later. It's like adding acceleration.
- c₃ × (k/20)³ bends even later. It's like "jerk", a change in acceleration.

A sum of powers like this is called a **polynomial**. Up to power 3, it's a **cubic**. Each term is 0 at k = 0, so the
prediction always starts from the current pose. The curve is also always smooth, with no zig-zags. Made-up example: if
the network thinks this player will keep turning a bit longer than the prior expects, it might set c₁ = +2°. That adds
2° by sample 20, so the forecast becomes 10.6° instead of 8.6°.

**Step 4: add them.** r_k = prior + correction, a rotation vector for each of the 20 future samples. Adding is only
exact when both turns share an axis. For small turns it's very close, and training adjusts for the small difference
anyway.

**Step 5: Cayley(r), from rotation vector to quaternion.** To apply the turn we need it as a quaternion. The textbook
way (the "exponential map", or **exp**) uses sin and cos. The Cayley map is a cheap shortcut: write the four numbers
(2, r) (the number 2 followed by the three numbers of r) and normalise them. It needs no sin or cos, only a rescale.
It turns slightly less than the exact way: at 20° it's about 1% short. Since the model is trained with Cayley, it
learns to account for that. It's named after Arthur Cayley, a 19th-century mathematician.

**Step 6: apply.** `q_last ⊗ Cayley(r_k)`, normalised: start from the current pose and apply the predicted turn. The
result is 20 predicted head orientations.

## Training

**Loss.** A single number for "how wrong was the model". Ours is the **angular** (or **geodesic**) error: the smallest
angle you'd have to turn the predicted head to match the true one, like the shortest route between two cities on a
globe. The formula in the code, 4·atan2(|p − t|, |p + t|), is just a numerically safe way to get that angle. Before
using it, the code flips the true quaternion's sign if needed (the q = −q quirk), so it always measures the shorter
way round.

**Endpoint weight 0.5.** Loss = half × (average error over all 20 samples) + half × (error at sample 20). Sample 20 is
what the challenge scores. The average keeps the whole path sensible.

**Gradient.** For every weight in the network, the direction to nudge it so the loss goes down a little. Training
repeats this nudge many thousands of times. Because the prior is part of the calculation, the nudges take it into
account. The network ends up learning only "what the prior gets wrong".

**Readout initialised to zero.** The last layer, which produces the 9 knobs, starts with all its weights at 0. So
before any training the knobs are 0 and the model's answer is exactly the damped prior.

**"Explicit residual" vs what we do.** We could compute "truth minus prior" for every training example and train the
network to output that. That needs a **log map**, the reverse of exp, which turns a rotation back into a rotation
vector. We don't do that. We judge only the final predicted pose, and the gradient sorts out the rest.

## Inference jargon

**MatMul (matrix multiplication).** A table of numbers times a list of numbers. Each output is a weighted sum:
multiply each input by a number from one row of the table and add them up. In the deployed graph, the table has one
row per future sample k:

| | knob 1 | knob 2 | knob 3 | rate |
|---|---:|---:|---:|---:|
| Row k | (k/20) | (k/20)² | (k/20)³ | τ(1 − e^(−k/τ)) |

Multiply that row by the list [c₁, c₂, c₃, rate] and you get c₁(k/20) + c₂(k/20)² + c₃(k/20)³ + rate × τ(1 − e^(−k/τ)).
That is correction plus prior, for all 20 samples and all 3 axes in one operation. That's why the prior costs nothing
extra in the deployed model. Two details from the code: the graph's rate is in radians per second, so the last column
is divided by 100; and in the default graph the table has three more columns, which add a constant offset the network
may put on the nod axis.

**Graph, node, ONNX.** The deployed model is a "graph": a recipe of simple operations such as MatMul, Add or Conv,
each called a node. ONNX is a standard file format for such recipes, so they can run on different hardware. On our
CPU, latency roughly follows the number of nodes (about 0.7 µs each), which is why folding the prior into an existing
MatMul matters.

**State.** Each block of the network keeps a short memory of its past inputs. Each call then processes only the one
new sample instead of all 400 again.

**Conv, kernel 8, dilation.** A convolution ("conv") slides a small pattern detector along the time series. Kernel 8
means it looks at 8 points at a time. Dilation means those points are spaced apart: dilation 47 looks at points 47
samples apart. Three blocks with dilations 1, 8 and 47 together cover 393 of the 400 samples.

**1×1 projection.** A step that mixes the 7 input channels (rate, speed, gravity) into 32 internal channels per
stream, separately at each moment in time.

**LayerNorm trick.** LayerNorm is a standard network operation that rescales a list of numbers. The graph feeds it the
predicted quaternion in a special arrangement so that it ends up normalising it to length 1. It uses an operation the
hardware already runs fast instead of a hand-built division.

## Where each step is in the code

| Step | Training model (`best/model.py`) | Deployed graph (`best/export.py`) |
|---|---|---|
| Rate | `LeanTCNN.features` | `StreamingLean.forward`, from the last two samples |
| Correction curve | `CompactTCNN.readout` (the `basis` buffer) | `StreamingLean.__init__` builds the `basis` table |
| Damped prior | `LeanTCNN.forward` | The extra `elapsed` column of `basis` |
| Cayley map | `map_quaternion` | `StreamingLean.forward`, (2, r) times the current pose |
| Readout set to zero | `GravityMirrorTCNN.__init__`, `CompactTCNN.__init__` | n/a |
| Loss | `rotation_loss` in `tcnn/model.py` | n/a |

The damped baseline on its own (no network) is `DampedVelocity` in `tcnn/damped.py`.

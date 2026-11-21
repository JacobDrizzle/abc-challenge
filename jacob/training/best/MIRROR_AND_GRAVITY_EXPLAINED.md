# Mirror equivariance and gravity, in plain English

k8_combined (the model in `best/`) has two additions that first came in with v6 (26 Oct 2026):
- **Mirror equivariance:** the model treats left and right as equally likely. Show it a mirrored head movement and it
  predicts the mirrored future, exactly.
- **Gravity:** the model is told which way is up, as seen from the head.

This explains both without assuming a maths background. For quaternions, rotation vectors and the rest of the
prediction, see `MATH_EXPLAINED.md`. Written 7 Nov 2026.

## The three head axes

Both additions talk about the head's three axes:
- **x** runs side to side, ear to ear. Turning about it is a **nod** (look up or down).
- **y** runs up through the top of the head. Turning about it is a **turn** (look left or right).
- **z** runs front to back. Turning about it is a **tilt** (ear toward shoulder).

In the BOXRR data, the world's y axis points straight up, away from the ground.

# Part 1: Mirror equivariance

## The word

"Equivariant" means that if you change the input in a certain way, the output changes in the same way. Here the change
is a left-right mirror.

Compare "invariant", where the output doesn't change at all. The model is invariant to which compass direction the
player faces: its inputs are measured from the head's own point of view, so facing north or south makes no difference.
The mirror is different: a mirrored input should give a mirrored output, not the same output.

## What a mirror does to head motion

Picture a player and their reflection:

| Movement | Axis | In the mirror |
|---|---|---|
| Nodding (look up / down) | x | Still a nod, same direction |
| Turning (look left / right) | y | Reversed: a left turn becomes a right turn |
| Tilting (ear toward shoulder) | z | Reversed: a left tilt becomes a right tilt |

A head orientation is stored as four numbers (w, x, y, z). The mirror rule is very simple: keep w and x, flip the signs
of y and z.

The inputs follow the same idea. Turning speed keeps its nod part and flips its turn and tilt parts. Total speed
doesn't change, because a mirror doesn't make you faster. The gravity input changes too (see "How the two fit together"
below).

## The two copies

The network runs as two copies side by side:
- Copy 0 watches the real head movement.
- Copy 1 watches the same movement with those signs flipped, as if in a mirror.

At every layer the copies share notes, and they do it the same way. Each takes "a lot of its own view" (weights A) plus
"some of the other's view" (weights B):
- Copy 0's new state = A × (its own) + B × (the other's)
- Copy 1's new state = A × (its own) + B × (the other's)

Both copies use the same A and B, so neither copy is special. Swap what they watch and they swap their states, nothing
else.

At the end, the mirror copy's answer is flipped back to normal (the sign flip S) and added to copy 0's answer. You can
think of it as two people describing a dancer, one watching directly and one watching in a mirror. At the end the
mirror watcher translates "left" back to "right" and they combine notes.

## Why the prediction comes out exactly mirrored

Feed the model a mirrored movement. Copy 0 now sees what copy 1 used to see, and copy 1 sees the mirror of a mirror,
which is the original. The copies have swapped places, so their final states swap too. Combining them with the
flip-back gives the old answer, flipped. The damped prior is mirrored as well, because it's built from the turning
speed, which was mirrored. So the whole prediction is mirrored, exactly and not approximately. No training is needed
for this. It's built into the wiring.

There's one detail at the end: a constant offset (bias) is only allowed on the nod axis. A constant offset on the turn
axis would mean "always lean a bit to the left", and a mirror would need "always lean a bit to the right". The two
can't both hold, so that offset is removed. A constant nod looks the same in a mirror, so it's allowed.

## What it buys

1. **Left and right teach each other.** Whatever the model learns from a left turn automatically applies to the
   matching right turn. It's a bit like doubling the training data.
2. **No left-right bias from the training players.** If the training users happened to turn left more often, the
   model can't pick that up. That should help on unseen users.
3. **About half the parameters.** An ordinary layer this wide needs a full table of weights. Here the table is
   [[A, B], [B, A]]: four blocks but only two different ones, so only A and B are stored. k8_combined has 7,555
   parameters, and the same model without mirroring has 15,113. The computation per sample stays the same.

The cost is that if real head motion has a genuine left-right lean (some players favour one side, for example), the
model can't use it.

## Does it help?

| Test | Without the mirror | Note |
|---|---:|---|
| v6 suite, 26 Oct, seed 42 | +0.029° | With twice the parameters |
| Leave-one-out run, seed 42 | +0.027° (4.2740° against 4.2471°) | With twice the parameters |

So far it's slightly more accurate and half the size. Both results are single seeds. Seeds 43 and 44 of the
leave-one-out run were still to come on 7 Nov.

The frame-robustness test (27–28 Oct) also checked the guarantee on real data: on mirrored validation data, the mirror
models' predictions are exactly the mirrored predictions.

# Part 2: Gravity

## What it is

Three extra input numbers that say **which way is up, as seen from the head**. The code calls it "gravity" because up
is simply the opposite of the direction gravity pulls. It's the same thing your inner ear senses: you know which way is
up even with your eyes closed.

Some examples:
- Standing straight, looking ahead: up points straight out of the top of your head.
- Looking at the ceiling: up points out of your face.
- Looking at the floor: up points out of the back of your head.
- Ear toward shoulder: up points partly out of the side of your head.

## How it's computed

From the newest head orientation (w, x, y, z) alone:

up as seen from the head = ( 2(xy + wz),  w² − x² + y² − z²,  2(yz − wx) )

You don't need to follow the formula. Three things about it matter:
- **It's only multiplications and additions** of the four numbers, so it's cheap. In the deployed graph it's one
  MatMul with a fixed table.
- **Every term multiplies two of the four numbers.** Flipping all four signs (the q = −q quirk, which describes the
  same orientation) changes nothing, because minus times minus is plus.
- **It uses only the newest sample**, so it adds no memory to the streaming model.

## Why the model needs it

The other inputs, the turning speeds, are measured from the head's own point of view. They say how the head is turning
but not where it's pointing. "Turning left at 50° per second" looks exactly the same whether the player is looking at
the floor, straight ahead or at the ceiling. But what happens next can differ.

The measurements below show gravity helps. These are the likely reasons, which haven't been tested one by one:
1. **Comfort.** People tend to come back toward looking level. If the head is far up, far down or tilted, the next move
   is more likely to be back toward level.
2. **Limits.** The neck can only bend so far, so a head that's already tilted back can't keep tilting much further.
3. **People turn around the vertical.** In a game you mostly swing your head left and right around the room's vertical
   axis. When you're looking down, that same swing appears, from the head's point of view, as a mix of turning and
   tilting. Knowing where up is lets the network recognise it as one simple swing.

The model still ignores which compass direction the player faces. Turning on the spot doesn't change where up is
relative to your head, so the gravity numbers stay the same. What the model now does notice is how far the head is
tilted, which is the point.

## What it costs

Three more input channels: 96 more parameters (7,459 without gravity, 7,555 with it).

## Does it help?

| Test | Without gravity | Note |
|---|---:|---|
| v6 suite, 26 Oct, seed 42 | +0.017° | Later shown to be too low |
| v8 no-gravity test, 28 Oct, 3 seeds | +0.041° (4.3705° against 4.3297°) | Every seed and 14 of 15 users worse |
| Leave-one-out run, seed 42 | +0.047° (4.2942° against 4.2471°) | Seeds 43 and 44 still to come on 7 Nov |

That's about 0.04° for 96 parameters.

## The risks

**1. The test data must use our "up".** Gravity assumes the world's y axis points up, as in BOXRR. If the test data
used a different up axis, the gravity numbers would be wrong. The frame-robustness test measured that: a wrong up axis
costs the gravity models 1–4°, which is worse than the damped prior on its own. On 28 Oct the organisers said they will
convert the test data to whatever frame we document. So the frame description in our submission has to be exact. (A
handedness mix-up costs every learned model 0.1–0.2°, with or without gravity.)

**2. Unusual posture.** Gravity teaches the model what a typical head position looks like. A player who sits unusually
can mislead it. In the v8 test, one validation user (daa4c4dd) held their head about 26° further back than usual, and
gravity made their error 0.94° worse, while it helped 14 of the 15 users.

**Tilt augmentation** softens both risks. During training, half the windows are turned as if the whole room were
tilted, by up to 15° about a random horizontal axis. The history and the future are tilted together, so the right
answer stays right. The turning speeds don't change (they're measured from the head), and only the up direction moves.
That teaches the model that up is a useful hint rather than something to trust exactly. In the v9 screen (28 Oct,
seed 42) it was about neutral for accuracy (−0.010°, 8 of 15 users better). But it halved the cost of a 15° tilted
world: +0.028° for k8_combined, against +0.059° for the earlier model without it. The leave-one-out run's `no_tilt`
variant will measure it on k8_combined itself.

# How the two fit together

The mirror swaps left and right but leaves up as up. So the two additions work together: in the mirror, only the
side-to-side part of the up direction flips.

Directions and turns behave differently in a mirror. A direction like "up, and a bit to the right" becomes "up, and a
bit to the left": only its side-to-side part flips. A turn is named by its axis, and it works the other way round. The
turn about the side-to-side axis (a nod) is the one that stays the same, and the turns about the other two axes reverse.
(Physicists call turn-type arrows "axial vectors" for this reason.)

Here's every number the model reads or writes, and what the mirror does to it:

| Number | In the mirror |
|---|---|
| Turning speed: nod part (x) | Kept |
| Turning speed: turn part (y) | Flipped |
| Turning speed: tilt part (z) | Flipped |
| Total speed | Kept |
| Up direction: side-to-side part (x) | Flipped |
| Up direction: vertical part (y) | Kept |
| Up direction: front-to-back part (z) | Kept |
| Predicted turn: nod, turn, tilt | Kept, flipped, flipped |

These are the `RATE_SIGNS`, `GRAVITY_SIGNS` and `ROTATION_SIGNS` at the top of `model.py`.

# Where each part is in the code

| Part | Where |
|---|---|
| The mirror rule (w, x, −y, −z) | `mirror` in `best/model.py` |
| Sign flips for each input and output | `RATE_SIGNS`, `GRAVITY_SIGNS`, `ROTATION_SIGNS` in `best/model.py` |
| The two copies | `LeanTCNN.forward` and `GravityMirrorTCNN.forward`: the second `projection` call sees the sign-flipped input |
| Sharing notes, [[A, B], [B, A]] | `SeparableBlock.pointwise_weight` (used by `KernelBlock`) |
| Flip back and combine, nod-only bias | `GravityMirrorTCNN.readout`, `head_bias_x` |
| Up as seen from the head | `gravity_in_head_frame` in `best/model.py` |
| Tilt augmentation | `tilts` and `TrainingWindows.batches` in `best/training.py` |
| Deployed graph | `StreamingLean.forward` in `best/export.py`: the two copies are the two rows of one tensor, and gravity is one MatMul with `_gravity_table` |
| Tests | `test_mirror_equivariance_is_exact_for_every_option`, `test_gravity_is_world_up_in_the_head_frame_and_sign_invariant`, `test_yaw_and_sign_invariant_but_tilt_sensitive`, `test_tilt_moves_only_gravity_and_keeps_the_loss` in `best/test_best.py` |
| Results | `jacob/ablations/README.md`: entries 2 (v6 suite), 11 (frame robustness), 12 (no-gravity test) and 15 (v9 screen) |

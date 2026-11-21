window.BEGINNER = {
 "overview": {
  "title": "How the model works",
  "idea": "This compact model, trained on recordings from 5,243 players, predicts where a virtual reality headset will turn a split second ahead.",
  "analogy": "It works like an experienced driver anticipating a bend in the road to steer smoothly.",
  "look": "Look from left to right across the coloured sheets to follow motion travelling through the network.",
  "try": "Click Next to step through each part of the model one piece at a time."
 },
 "input": {
  "title": "Recent headset motion",
  "idea": "The headset records which way it is facing one hundred times every second. Each reading is called a sample.",
  "analogy": "It is like glancing at the second hand of a watch while remembering where the minute hand was.",
  "look": "Look at the four rows w, x, y and z: the oldest pose is deepest into the screen and \"now\" is at the front edge. Each call reads only the two front-most poses.",
  "try": "Press Play to watch the headset turn while new samples arrive at the front."
 },
 "feat": {
  "title": "Instant motion measurements",
  "idea": "A pose says where the head is, not how it is moving. So the model first works out seven measurements: the turning speed (from the newest two poses) and the direction of gravity (from the newest pose).",
  "analogy": "It is like a car dashboard showing how fast you are turning, next to a spirit level showing which way is down.",
  "look": "Look at the seven coloured rows: turning speed around three axes, total speed, and three numbers for which way is down.",
  "try": "Press Next to see these seven measurements mixed into thirty-two channels."
 },
 "proj": {
  "title": "Thirty-two feature channels",
  "idea": "The seven measurements are blended into thirty-two internal signals, called channels. The rest of the model works on these.",
  "analogy": "It is like a mixing desk combining seven input sliders, some turned up and some turned down, into thirty-two new channels.",
  "look": "Look at the fan of lines: each line carries one of the seven measurements into one of the thirty-two channels.",
  "try": "Click any of the thirty-two channels to see only its seven lines light up."
 },
 "mirror": {
  "title": "Left and right symmetry",
  "idea": "The model runs the motion and its left-right mirror image through exactly the same rules, which halves its size.",
  "analogy": "It is like using one instruction manual that applies equally whether turning left or turning right.",
  "look": "The top sheets show the motion as it was recorded. The bottom sheets show its left-right mirror image.",
  "try": "Click Mirror the motion to watch the two sheets smoothly swap places."
 },
 "blk0": {
  "title": "Block 1: short range",
  "idea": "To forecast motion the model has to look back in time, and three blocks do that. This first block looks at the eight newest samples, the last 80 ms: the finest detail the model sees.",
  "analogy": "It works like an alert driver feeling sudden vibrations directly through the steering wheel.",
  "look": "The eight closely spaced columns are eight samples in a row: seven held in memory (violet box) plus the newest one (white box). The strip underneath marks the 8 samples (0.08 s) its output depends on.",
  "try": "Click One call to watch the newest sample enter and flow through the block."
 },
 "blk1": {
  "title": "Block 2: medium range",
  "idea": "The second block works like the first, but spaces its eight checks eight steps apart (80 ms), so it follows a turn that is already under way.",
  "analogy": "It is like a dancer keeping rhythm with the main beat rather than counting every single footstep.",
  "look": "The eight checks now sit further apart, across a wider stretch of history. The strip underneath marks the 64 samples (0.64 s) its output depends on: it reads Block 1’s outputs, and each of those already covers 8.",
  "try": "Click any channel to see how its eight spaced checks are added into a single sum."
 },
 "blk2": {
  "title": "Block 3: long range",
  "idea": "The third block spaces its eight checks 47 steps apart, so on its own it looks over about 3.3 seconds of movement.",
  "analogy": "It is like a distance runner checking split times every kilometre to judge overall pacing.",
  "look": "The eight checks stretch back across the long violet memory box. The strip underneath marks the 393 samples (3.93 s) its output depends on; amber ones are reached twice.",
  "try": "Press One call to watch the oldest sample leave the memory as the newest arrives."
 },
 "reach": {
  "title": "Full history reach",
  "idea": "Stacked together, the three blocks let a single prediction look back over almost four seconds of motion, with no gaps.",
  "analogy": "It is like a family tree branching backward from one person to reach hundreds of ancestors.",
  "look": "Seen from above, the purple tree branches back from one output to every sample it can see.",
  "try": "Drag anywhere to see the tree from another angle."
 },
 "head": {
  "title": "Nine curve dials",
  "idea": "Instead of guessing twenty separate poses, the model works out just nine numbers. Think of them as dials that shape one smooth curve.",
  "analogy": "It is like adjusting three tone dials on a stereo to shape an entire sound wave smoothly.",
  "look": "Look at the three-by-three square of dials: one row of three for each turning axis.",
  "try": "Open the \"basis curve\" tab in the top-right panel, then press Play: the nine dials move as the motion changes."
 },
 "traj": {
  "title": "Momentum guess plus learned correction",
  "idea": "Two parts are added. The momentum guess carries the current turning speed forward and lets it fade. The learned correction, shaped by the nine dials, adjusts it.",
  "analogy": "It is like a curling stone gliding on ice while sweepers brush the surface to steer its finish.",
  "look": "In the top-right panel, the pink momentum guess and the amber learned correction add up to the blue trajectory.",
  "try": "Press Play to watch the momentum guess and the learned correction adjust as the head turns."
 },
 "out": {
  "title": "Twenty-step future forecast",
  "idea": "The model turns the trajectory into twenty future headset poses, covering the next two hundred milliseconds.",
  "analogy": "It is like marking twenty spots on a path, each measured straight from where the person stands now: 10 ms ahead, 20 ms ahead, up to 200 ms.",
  "look": "Look for the white outline around column twenty: it is the only step the challenge scores.",
  "try": "Press Play and look at the top-right panel: the blue dashed predicted turn tracks the green true turn, and the red bars show the error at each step."
 },
 "gaze": {
  "title": "Where the headset points",
  "idea": "The same forecast, shown as a direction. A line from the front of the headset meets a sphere, so you can see where the head will point next. It is the head's direction, not where the eyes look.",
  "analogy": "It is like a laser pointer projecting a dot onto a wall to show where you are aiming.",
  "look": "Compare the blue predicted dots against the green dots showing where the head really pointed; the big dots are step 20.",
  "try": "Drag anywhere to look at the headset and the dots from another angle."
 }
};

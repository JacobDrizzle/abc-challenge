// scene_task.js: everything on deck 1's open stage except the head itself (SPEC sections G4 and H2; ticket 10).
//
// One scene for all eight steps of the deck. Nothing here is in a box: the head stands in open space, the last 4 seconds run
// back from it as four traces, and far behind it stand the dataset's sheets, one behind another, with a line from every
// recording that survives a check to its place on the next sheet. The camera moves through this space as the reader scrolls.
//
//   position s   step      what the camera does
//   0            hook      looks at the head from outside (set by stage_task.js)
//   1            in        sits in the stream of the last 4 s, behind the head, looking along the traces to it
//   2            out       comes beside the head, where the traces end: the model's guess for 200 ms ahead, and beside "now"
//                          the guess it made 200 ms ago against what the head really did
//   3            raw       flies back to the dataset and stops close among the recordings on sheet 1
//   4            rate      follows the survivors' lines to sheet 2
//   5            motion    and on to sheet 3
//   6            split     pulls back: the whole funnel, and the split by player
//   7            window    one 4-second window leaves the training group: the traces that lead back to the head
//
// Everything that depends on scroll is a function of s (setPos), so it reverses exactly. Only the traces follow the clock,
// because they are the recording being replayed (update).
//
// Where each number comes from (they are the page's existing figures; 13 Nov 2026: the split and the window count are the final dataset's):
//   1,000 dots stand for the 394,171 recordings checked in the one pull whose log we have,    data/pipeline/pull_counts.json
//             so one dot is about 394 recordings                                              (the PC pull, snapshot_20261106c)
//   404 dots  = 159,099 / 394,171 of 1,000 (recorded near 100 samples a second)               same file
//   19 dots   = 7,635 / 394,171 of 1,000 (head actually moving: pass every check)             same file
//   2,878 kept, from 1,081 players (at most 5 per player)                                     same file: processed_json.qualified, usable_users
//   5,243 : 786 : 1,049 players (train : tune : test), 7,078 together                         manifest_moredata_final.json "totals"
//   204,603,125 training windows                                                              data/pipeline/examples.js final.splits.train.windows (tools/make_data_final.py)
// WHICH dots survive is drawn from a fixed seed: the picture shows the proportions, not which recordings passed.
// The traces are the clip's own samples (hero/clip.js): q = what the head did, pred = the model's prediction for 200 ms ahead.

const SEED = 42;
const N_ALL = 1000, N_RATE = 404, N_MOVE = 19;         // dots on sheets 1, 2 and 3
const COLS = 50, ROWS = 20;                           // sheet 1's grid
const SPLIT = [5243, 786, 1049];                      // players: train, tune, test (the final dataset)
const HISTORY = 400, AHEAD = 20;                      // samples: 4 s in, 200 ms out
// distances are in "bust heights" (U): the head's bust is 1 U tall
const RIBBON_FROM = 0.8, RIBBON_LEN = 5.0;            // "now" is 0.8 U behind the head's centre; 4 s of samples run back 5 U from there
const ROW_GAP = 0.3, ROW_AMP = 0.11;                  // the four traces: spacing between them, and the height of each one's swing
const SHEET_F = [-16, -13, -10.8, -8.8];              // how far behind the head each sheet stands (sheet 1, 2, 3, the split)
const SHEET_W = 6, SHEET_H = 2.4;                     // a sheet's size
const INK = 0xeaeaea, GREY = 0x8a8a8a, DIM = 0x2e2e2e, AMBER = 0xf0b24a;

// A small seeded random generator (mulberry32): the same seed always gives the same survivors.
function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function pick(rand, from, n) { const a = from.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; } return a.slice(0, n).sort((x, y) => x - y); }
const ramp = (v, a, b) => (v <= a ? 0 : v >= b ? 1 : (v - a) / (b - a));
const ease = t => t * t * (3 - 2 * t);

export function createTaskScene({ THREE, clip, origin, U, F, R, side, accent, labelsEl }) {
  const UP = new THREE.Vector3(0, 1, 0), group = new THREE.Group(); group.name = 'task-scene';
  // a point given as (f, u, r): f bust-heights along the facing direction (negative = behind the head), u up, r to the side
  const P = (f, u, r, out = new THREE.Vector3()) => out.copy(origin).addScaledVector(F, f * U).addScaledVector(UP, u * U).addScaledVector(R, r * U * side);
  const disposables = [];
  const keep = o => { disposables.push(o); return o; };
  const lineMat = (color, opacity = 1) => keep(new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));

  // ------------------------------------------------------------------ the traces: the last 4 s, and the next 200 ms
  // Each of the four numbers of the orientation (w, x, y, z) is one line. A value is drawn against that number's own range over
  // the whole clip, so every line uses its full height (the same choice as the viewer's input panel).
  const lo = [1, 1, 1, 1].map(() => Infinity), hi = lo.map(() => -Infinity);
  for (const q of clip.q) for (let c = 0; c < 4; c++) { if (q[c] < lo[c]) lo[c] = q[c]; if (q[c] > hi[c]) hi[c] = q[c]; }
  const val = (q, c) => ((q[c] - lo[c]) / ((hi[c] - lo[c]) || 1) - 0.5) * 2;          // −1 … 1
  const rowU = c => (1.5 - c) * ROW_GAP;                                             // w on top, z at the bottom
  const past = [];
  const pastMat = lineMat(INK, 0);
  for (let c = 0; c < 4; c++) {
    const g = keep(new THREE.BufferGeometry()); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(HISTORY * 3), 3).setUsage(THREE.DynamicDrawUsage));
    const l = new THREE.Line(g, pastMat); l.frustumCulled = false; past.push(l); group.add(l);
  }
  // Three dots on each trace, all real values from the clip, all on the same scale as the trace (20 samples ahead is 20 samples'
  // length ahead: 200 ms really is that short next to 4 s):
  //   at "now"      amber  = where the head really is: the SCORED sample of the prediction made 200 ms ago
  //   at "now"      accent = what the model predicted for this moment, 200 ms ago (pred[i − 20]); the gap to the amber dot is its error
  //   at "+200 ms"  accent = the model's prediction made now (pred[i]). Nothing is drawn of what the head WILL do: it has not happened yet.
  const dotTex = keep(new THREE.CanvasTexture((() => { const c = document.createElement('canvas'); c.width = c.height = 64; const x = c.getContext('2d'); x.fillStyle = '#fff'; x.beginPath(); x.arc(32, 32, 28, 0, Math.PI * 2); x.fill(); return c; })()));
  const pointsMat = size => keep(new THREE.PointsMaterial({ size: size * U, map: dotTex, transparent: true, alphaTest: 0.35, depthWrite: false, sizeAttenuation: true, vertexColors: true }));   // each dot takes its colour from the geometry
  const endGeo = keep(new THREE.BufferGeometry()); endGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12 * 3), 3).setUsage(THREE.DynamicDrawUsage));
  endGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(12 * 3), 3));
  { const col = endGeo.attributes.color, a = new THREE.Color(AMBER), b = new THREE.Color(accent); for (let c = 0; c < 4; c++) { col.setXYZ(c, a.r, a.g, a.b); col.setXYZ(4 + c, b.r, b.g, b.b); col.setXYZ(8 + c, b.r, b.g, b.b); } }
  const endMat = pointsMat(0.05); endMat.opacity = 0;
  const ends = new THREE.Points(endGeo, endMat); ends.frustumCulled = false; group.add(ends);

  const v = new THREE.Vector3();
  // the point for the sample k steps before "now" (k < 0: after it), value x (−1 … 1), on trace c. One scale for past and future.
  function tracePoint(k, x, c, out) { return P(-(RIBBON_FROM + (k / (HISTORY - 1)) * RIBBON_LEN), rowU(c) + x * ROW_AMP, 0, out); }
  const HIDE = 1e6;
  let lastI = -1;
  function update(frac) {                              // frac = the replay's position in samples (a fraction: between two samples)
    const i = Math.floor(frac);
    if (i === lastI) return; lastI = i;
    const n = Math.min(HISTORY, i + 1);                // just after the loop restarts there are fewer than 400 samples behind "now"
    const e = endGeo.attributes.position;
    for (let c = 0; c < 4; c++) {
      const a = past[c].geometry.attributes.position;
      for (let k = 0; k < n; k++) { tracePoint(k, val(clip.q[i - k], c), c, v); a.setXYZ(k, v.x, v.y, v.z); }
      past[c].geometry.setDrawRange(0, n); a.needsUpdate = true;
      tracePoint(0, val(clip.q[i], c), c, v); e.setXYZ(c, v.x, v.y, v.z);
      if (i >= AHEAD) { tracePoint(0, val(clip.pred[i - AHEAD], c), c, v); e.setXYZ(4 + c, v.x, v.y, v.z); } else e.setXYZ(4 + c, HIDE, HIDE, HIDE);   // the clip's first 200 ms have no earlier prediction
      tracePoint(-AHEAD, val(clip.pred[i], c), c, v); e.setXYZ(8 + c, v.x, v.y, v.z);
    }
    e.needsUpdate = true;
  }

  // ------------------------------------------------------------------ the dataset: sheets of dots, and lines between them
  const data = new THREE.Group(); data.name = 'task-dataset'; group.add(data);
  const sheetPoint = (sheet, x, y, out = new THREE.Vector3()) => P(SHEET_F[sheet], y, x, out);   // x across (in U), y up (in U)
  const rand = rng(SEED);
  const all = Array.from({ length: N_ALL }, (_, i) => i), keptRate = pick(rand, all, N_RATE), keptMove = pick(rand, keptRate, N_MOVE);
  const inRate = new Set(keptRate), inMove = new Set(keptMove);
  // where each dot sits: sheet 1 is the full 50 × 20 grid; sheet 2 packs its 404 survivors into 26 columns; sheet 3 is one column of 19
  const gx = SHEET_W / COLS, gy = SHEET_H / ROWS;
  const p1 = i => [(i % COLS - (COLS - 1) / 2) * gx, ((ROWS - 1) / 2 - Math.floor(i / COLS)) * gy];
  const C2 = 26, R2 = Math.ceil(N_RATE / C2);
  const p2 = j => [(j % C2 - (C2 - 1) / 2) * gx, ((R2 - 1) / 2 - Math.floor(j / C2)) * gy];
  const p3 = j => [0, ((N_MOVE - 1) / 2 - j) * gy];
  const dots = (count, at, sheet) => {
    const g = keep(new THREE.BufferGeometry()), pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) { const [x, y] = at(i); sheetPoint(sheet, x, y, v); pos.set([v.x, v.y, v.z], i * 3); }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3).setUsage(THREE.DynamicDrawUsage));
    const m = pointsMat(0.062); const o = new THREE.Points(g, m); o.frustumCulled = false; data.add(o); return o;
  };
  const d1 = dots(N_ALL, p1, 0), d2 = dots(N_RATE, p2, 1), d3 = dots(N_MOVE, p3, 2);
  const links = (pairs, color) => {                    // one straight line per survivor, from its place on one sheet to its place on the next
    const g = keep(new THREE.BufferGeometry()), pos = new Float32Array(pairs.length * 6);
    pairs.forEach(([a, b], i) => pos.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6));
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const o = new THREE.LineSegments(g, lineMat(color, 0)); o.frustumCulled = false; data.add(o); return o;
  };
  const l12 = links(keptRate.map((i, j) => [sheetPoint(0, ...p1(i)), sheetPoint(1, ...p2(j))]), GREY);
  const l23 = links(keptMove.map((i, j) => [sheetPoint(1, ...p2(keptRate.indexOf(i))), sheetPoint(2, ...p3(j))]), accent);
  // the split by player: three outlined areas whose widths are in the proportion 5,243 : 786 : 1,049
  const total = SPLIT[0] + SPLIT[1] + SPLIT[2], SW = 5, SH = 1.1, GAPX = 0.12;
  const groups = [], groupMid = []; let x0 = -SW / 2;
  SPLIT.forEach((n, k) => {
    const w = (SW - 2 * GAPX) * n / total, a = x0, b = x0 + w; x0 = b + GAPX;
    const g = keep(new THREE.BufferGeometry().setFromPoints([sheetPoint(3, a, SH / 2), sheetPoint(3, b, SH / 2), sheetPoint(3, b, -SH / 2), sheetPoint(3, a, -SH / 2)]));
    const o = new THREE.LineLoop(g, lineMat(k === 0 ? accent : INK, 0)); o.frustumCulled = false; data.add(o); groups.push(o); groupMid.push([(a + b) / 2, a, b]);
  });
  // ONE line from sheet 3 to the split, not one per dot: the recordings that pass give the 1,081 players of this pull, and the split
  // also holds 5,997 players from the team's earlier file and the later pulls, so no dot-to-group picture would be true.
  const l3s = links([[sheetPoint(2, 0, -((N_MOVE - 1) / 2) * gy - 0.15), sheetPoint(3, 0, SH / 2)]], GREY);
  // one survivor, followed: the same recording's dot on each sheet joined by one bright line (which one is seeded, like the rest)
  const loneJ = Math.floor(N_MOVE / 2), loneI = keptMove[loneJ];
  const lone = links([[sheetPoint(0, ...p1(loneI)), sheetPoint(1, ...p2(keptRate.indexOf(loneI)))], [sheetPoint(1, ...p2(keptRate.indexOf(loneI))), sheetPoint(2, ...p3(loneJ))]], INK);
  // one window leaves the training group: a line from it to the far (oldest) end of the traces
  const lwin = links([[sheetPoint(3, groupMid[0][0], -SH / 2), P(-(RIBBON_FROM + RIBBON_LEN), 0, 0)]], accent);
  // each sheet's outline: a hairline rectangle
  const frames = [0, 1, 2].map(sh => { const w = SHEET_W / 2 + 0.25, h = SHEET_H / 2 + 0.25;
    const g = keep(new THREE.BufferGeometry().setFromPoints([sheetPoint(sh, -w, h), sheetPoint(sh, w, h), sheetPoint(sh, w, -h), sheetPoint(sh, -w, -h)]));
    const o = new THREE.LineLoop(g, lineMat(INK, 0)); o.frustumCulled = false; data.add(o); return o; });

  const cInk = new THREE.Color(INK), cDim = new THREE.Color(DIM), cAcc = new THREE.Color(accent), cTmp = new THREE.Color();
  function paint(obj, count, level) {                  // level(i) → 0 (dim) … 1 (lit) … 2 (kept: accent)
    const col = obj.geometry.attributes.color;
    for (let i = 0; i < count; i++) { const l = level(i); (l <= 1 ? cTmp.copy(cDim).lerp(cInk, l) : cTmp.copy(cInk).lerp(cAcc, l - 1)); col.setXYZ(i, cTmp.r, cTmp.g, cTmp.b); }
    col.needsUpdate = true;
  }

  // ------------------------------------------------------------------ small labels in the page's mono face, placed over the scene
  // They are HTML (real text, sharp at any size), positioned every frame at the point they name.
  const labels = [];
  function label(html, at, from, to, until = 99) {
    const e = document.createElement('div'); e.className = 'sc-lab'; e.innerHTML = html;   // html is one of the fixed strings below, never data from outside
    e.setAttribute('aria-hidden', 'true');   // the same figures are in the text beside the stage
    labelsEl.appendChild(e); labels.push({ e, at, from, to, until });
  }
  const top = (sh, up = 0.42) => sheetPoint(sh, -SHEET_W / 2 - 0.25, SHEET_H / 2 + up);   // each sheet's label sits at its own height, so they do not cover each other in the wide shot
  label('<b>394,171</b> recordings, one batch · one dot ≈ 394', top(0, 0.95), 2.6, 3.0);
  label('<b>159,099</b> recorded near 100 a second', top(1, 0.68), 3.5, 4.0);
  label('<b>7,635</b> pass every check', sheetPoint(2, -0.35, SHEET_H / 2 + 0.42), 4.5, 5.0);
  label('<b>2,878</b> kept · from <b>1,081</b> players', sheetPoint(2, 0.3, -SHEET_H / 2 - 0.3), 4.6, 5.0, 5.2);
  label('<b>7,078</b> players · all batches, same checks', sheetPoint(3, -SW / 2, -SH / 2 - 0.62), 5.5, 6.0);
  label('<b>5,243</b> players to learn from', sheetPoint(3, groupMid[0][1], -SH / 2 - 0.22), 5.5, 6.0);
  label('<b>786</b> to tune', sheetPoint(3, groupMid[1][1], SH / 2 + 0.34), 5.5, 6.0);
  label('<b>1,049</b> never seen', sheetPoint(3, groupMid[2][1], -SH / 2 - 0.22), 5.5, 6.0);
  label('one of <b>204,603,125</b> windows · 4 s in', P(-(RIBBON_FROM + RIBBON_LEN) + 0.2, 1.5 * ROW_GAP + 0.3, 0), 6.5, 7.0);
  label('w · x · y · z, each on its own scale', P(-(RIBBON_FROM + RIBBON_LEN * 0.55), 1.5 * ROW_GAP + 0.3, 0), 0.6, 1.0, 2.6);
  label('−4 s', P(-(RIBBON_FROM + RIBBON_LEN), -1.5 * ROW_GAP - 0.28, 0), 0.6, 1.0, 2.6);
  label('now', P(-RIBBON_FROM, -1.5 * ROW_GAP - 0.28, 0), 0.6, 1.0, 2.6);
  label('<i>+200 ms</i>', tracePoint(-AHEAD, 0, 0, new THREE.Vector3()).addScaledVector(UP, 0.3 * U), 1.5, 2.0, 2.6);

  // ------------------------------------------------------------------ what scroll changes
  let s = 0, paintedA = -1, paintedB = -1;             // the dot colours last painted, so they are redone only when they change
  function setPos(pos) {
    s = pos;
    pastMat.opacity = 0.85 * ramp(s, 0.3, 1) * (1 - 0.55 * ramp(s, 2.2, 3));                 // the traces stay, fainter, for the whole deck: they are where the data ends up
    const ahead = ramp(s, 1.4, 2) * (1 - ramp(s, 2.3, 3));
    endMat.opacity = ahead; ends.visible = ahead > 0.01;
    const show = ramp(s, 2.3, 3); data.visible = show > 0.001;
    frames[0].material.opacity = 0.34 * show;
    frames[1].material.opacity = 0.34 * ramp(s, 3.3, 4); frames[2].material.opacity = 0.34 * ramp(s, 4.3, 5);
    const a = ease(ramp(s, 3.2, 3.9)), b = ease(ramp(s, 4.2, 4.9)), c = ease(ramp(s, 5.2, 5.9)), w = ease(ramp(s, 6.3, 7));
    d1.material.opacity = show;
    // Recolouring 1,000 + 404 dots means re-sending their colours to the graphics chip, so it is done only when the two values the
    // colours depend on have changed, and never while the dataset is not shown.
    if (data.visible && (a !== paintedA || b !== paintedB)) {
      paint(d1, N_ALL, i => (inRate.has(i) ? 1 : 1 - 0.86 * a));                              // the recordings that fail the rate check go dark
      paint(d2, N_RATE, j => (inMove.has(keptRate[j]) ? 1 + b : 1 - 0.86 * b));               // then those whose head hardly moves
      if (paintedA < 0) paint(d3, N_MOVE, () => 2);
      paintedA = a; paintedB = b;
    }
    l12.material.opacity = 0.2 * a; d2.material.opacity = a; d2.visible = a > 0.001;
    l23.material.opacity = 0.75 * b; d3.material.opacity = b; d3.visible = b > 0.001;
    lone.material.opacity = 0.95 * Math.max(a, b) * (1 - ramp(s, 5.6, 6.4));               // the one survivor being followed, while the camera is close
    l3s.material.opacity = 0.45 * c; groups.forEach((g, k) => { g.material.opacity = (k === 0 ? 0.95 : 0.55) * c; });
    lwin.material.opacity = 0.9 * w;
  }
  setPos(0);

  // ------------------------------------------------------------------ the camera's path: one place per step, eased between them
  // key 0 (the hook) is set by the stage, which frames the bust. The rest are (position, target) as (f, u, r) points.
  const K = [null,
    [P(-4.6, 0.55, 0.9), P(0, 0.05, 0)],               // in: behind the head, in the stream
    [P(-0.5, 0.25, 3.3), P(-0.45, 0.0, 0)],            // out: beside the head, where the traces end at "now"
    [P(-12.5, 0.3, 1.1), P(-16, 0.0, 0)],              // raw: in front of sheet 1, close enough that single recordings are dots you could count
    [P(-8.9, 0.8, 2.4), P(-13.2, 0.0, 0)],             // rate: back a little, so sheet 2 and the lines arriving on it are both in view
    [P(-7.2, 0.6, 2.0), P(-11.2, 0.0, 0)],             // motion: the 19 that are left, with the two sheets they came through behind them
    [P(0.5, 2.7, 8.4), P(-11.4, -0.1, 0)],             // split: back and to the side: the whole funnel
    [P(-0.6, 1.5, 5.4), P(-5.6, 0.0, 0)],              // window: where the traces leave the dataset
  ];
  const FOVS = [24, 30, 26, 34, 34, 34, 38, 34];       // degrees: narrow for the portrait, wider where there is more to see (FOVS[0] is replaced by setKey0)
  function camera(pos, out) {
    const i = Math.max(0, Math.min(K.length - 2, Math.floor(pos))), t = ease(Math.max(0, Math.min(1, pos - i)));
    out.position.lerpVectors(K[i][0], K[i + 1][0], t); out.target.lerpVectors(K[i][1], K[i + 1][1], t);
    out.fov = FOVS[i] + (FOVS[i + 1] - FOVS[i]) * t;
    out.offset = 1 - ease(ramp(pos, 0.2, 1));          // how much of the hook's sideways shift is left
    return out;
  }

  function placeLabels(cam, w, h) {
    for (const L of labels) {
      const o = ramp(s, L.from, L.to) * (1 - ramp(s, L.until, L.until + 0.4));
      if (o < 0.01) { if (L.e.style.display !== 'none') L.e.style.display = 'none'; continue; }
      v.copy(L.at).project(cam);
      if (v.z > 1) { L.e.style.display = 'none'; continue; }                                 // behind the camera
      L.e.style.display = ''; L.e.style.opacity = o.toFixed(2);
      L.e.style.transform = 'translate(' + ((v.x + 1) / 2 * w).toFixed(1) + 'px,' + ((1 - v.y) / 2 * h).toFixed(1) + 'px)';
    }
  }

  return {
    group, update, setPos, camera, placeLabels,
    setKey0(position, target, fov) { K[0] = [position.clone(), target.clone()]; if (fov) FOVS[0] = fov; },   // the hook's place and field of view come from the stage, which frames the bust
    counts: () => ({ sheet1: N_ALL, sheet2: N_RATE, sheet3: N_MOVE }),
    dispose() { disposables.forEach(o => o.dispose()); labels.forEach(L => L.e.remove()); group.removeFromParent(); group.clear(); },
  };
}

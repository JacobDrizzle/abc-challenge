// P1 head rig: turns ONE head rotation (what a headset reports) into the pose of a small neck, and skins the mesh with it.
// Classic script, no dependencies, so the 3D view (head3d.js) and the Data side's no-WebGL fallback (dataview.js) share it.
//
// The mesh (assets/head.js, window.K8_HEAD) has three bones, each with a pivot: neck_lower (about C7/T1), neck_upper (about C3/C4)
// and head (the joints under the skull). Every skin vertex carries three weights (neck_lower, neck_upper, head); what is left to 1
// belongs to the chest, which does not move. Axes: the face looks along −z, up is +y. Quaternions here are [x, y, z, w].
//
// How one rotation q is shared out (K8HeadPose.solve):
//   1. q is split into a twist about "up" (turning left/right, yaw) and a swing (everything else: nodding and leaning).
//      This is the swing-twist decomposition: twist = the part of q about the up axis, swing = q · twist⁻¹.
//   2. The two neck bones each take a fixed share of the twist and of the swing (SHARE below). Real necks do this: most of a turn
//      happens right under the skull, while nodding and leaning are spread down the neck (Zhou et al., J Biomech 2019).
//   3. The head bone gets whatever is left, so the head's final rotation is exactly q: the headset always shows the recorded pose.
// Each bone turns about its own pivot and carries the bones above it. Skin vertices blend the bones that own them
// (linear blend skinning), which is what bends the neck instead of shearing it.
(function () {
  'use strict';
  // share of each kind of motion taken by [neck_lower, neck_upper]; the head takes the rest.
  // yaw: anatomy puts about 77% under the skull; a little more is given to the neck here so the skin twist is spread out.
  var SHARE = { yaw: [0.20, 0.20], pitch: [0.40, 0.30], roll: [0.45, 0.35] };

  function mul(a, b) { return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0], a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]]; }
  function conj(a) { return [-a[0], -a[1], -a[2], a[3]]; }
  function norm(a) { var n = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]) || 1; return [a[0] / n, a[1] / n, a[2] / n, a[3] / n]; }
  function rot(q, v) { var x = q[0], y = q[1], z = q[2], w = q[3], tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)]; }
  // a fraction f of rotation q (same axis, f times the angle)
  function part(q, f) { var w = Math.max(-1, Math.min(1, q[3])), half = Math.acos(w), s = Math.sin(half); if (s < 1e-6) return [0, 0, 0, 1];
    var h2 = half * f, k = Math.sin(h2) / s; return [q[0] * k, q[1] * k, q[2] * k, Math.cos(h2)]; }

  // q -> the three bones' world rotations and where each pivot ends up. Returns { rot: [qLower, qUpper, qHead], piv: [pL, pU, pH] }.
  function wrap(a) { a = a % (2 * Math.PI); return a > Math.PI ? a - 2 * Math.PI : a <= -Math.PI ? a + 2 * Math.PI : a; }
  // how far round "up" a rotation faces (its heading), in radians
  function heading(q) { return wrap(2 * Math.atan2(q[1], q[3])); }
  var NECK_TURN = 1.2;   // radians (about 69°): the most the neck and head turn relative to the body; beyond that the body turns
  // ref = the heading the body faces when the player looks "straight ahead" for this recording (radians; default 0).
  // A neck cannot turn all the way round, so the turn is split first: up to about 69° stays in the neck and head (a soft limit,
  // tanh), and the rest turns the whole body about a vertical line through the base of the neck. Without this, a player facing
  // backwards would have a neck twisted by 70° that flips to the other side the moment they pass 180°.
  function solve(q, piv, ref) {
    q = norm(q);
    var d = wrap(heading(q) - (ref || 0)), dn = NECK_TURN * tanh(d / NECK_TURN), B = (ref || 0) + d - dn;
    var qB = [0, Math.sin(B / 2), 0, Math.cos(B / 2)];                           // the body's turn
    q = mul(conj(qB), q); if (q[3] < 0) q = [-q[0], -q[1], -q[2], -q[3]];        // the head relative to the body
    var twist = norm([0, q[1], 0, q[3]]), swing = mul(q, conj(twist));
    // how much of the swing is nodding (axis across the head, x) and how much is leaning (axis front to back, z)
    var ax = swing[0] * swing[0], az = swing[2] * swing[2], t = ax + az, nod = t > 1e-12 ? ax / t : 1;
    var fL = SHARE.pitch[0] * nod + SHARE.roll[0] * (1 - nod), fU = SHARE.pitch[1] * nod + SHARE.roll[1] * (1 - nod);
    var qL = mul(qB, mul(part(swing, fL), part(twist, SHARE.yaw[0])));           // neck_lower, in the world (carried by the body)
    var qU = mul(qL, mul(part(swing, fU), part(twist, SHARE.yaw[1])));           // neck_upper = lower, then its own share
    var qH = mul(qB, q);                                                         // the head: exactly the rotation asked for
    var pL = piv[0], pU = add(pL, rot(qL, sub(piv[1], pL))), pH = add(pU, rot(qU, sub(piv[2], pU)));   // neck_lower's pivot is on the body's axis, so it stays put
    return { rot: [qL, qU, qH], piv: [pL, pU, pH], body: qB };
  }
  function tanh(x) { var e = Math.exp(2 * x); return (e - 1) / (e + 1); }
  // the average heading of a list of headings (radians), going round the circle properly
  function meanHeading(list) { var sx = 0, cx = 0, i; for (i = 0; i < list.length; i++) { sx += Math.sin(list[i]); cx += Math.cos(list[i]); } return Math.atan2(sx, cx); }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }

  // Build a poser for the loaded mesh. make() returns null if assets/head.js is missing or has no bones.
  function make() {
    var H = window.K8_HEAD; if (!H || !H.bones) return null;
    var skinPart = null, i; for (i = 0; i < H.parts.length; i++) if (H.parts[i].group === 'skin') skinPart = H.parts[i];
    if (!skinPart) return null;
    var u = H.unit, restPiv = H.bones.map(function (b) { return [b.pivot[0] * u, b.pivot[1] * u, b.pivot[2] * u]; });
    var n = skinPart.p.length / 3, rest = new Float32Array(n * 3), W = new Float32Array(n * 3);
    for (i = 0; i < n * 3; i++) { rest[i] = skinPart.p[i] * u; W[i] = skinPart.w[i] / 255; }
    return {
      count: n, rest: rest, part: skinPart, pivots: restPiv,
      // Fill P (positions) and, if given, N (normals, from rest normals restN) for head rotation q = [x, y, z, w].
      // Returns the rigid transform of everything fixed to the head (the headset): { q, t } with x' = q·x + t.
      apply: function (q, P, restN, N, ref) {
        var s = solve(q, restPiv, ref), R = s.rot, qB = s.body, ax = restPiv[0], pv = s.piv, k, b, j, v = [0, 0, 0], o, w, acc = [0, 0, 0], accN = [0, 0, 0], r, m;
        // each bone as a matrix-free transform: x' = R_b · (x − restPivot_b) + posedPivot_b
        for (k = 0, j = 0; k < n; k++, j += 3) {
          v[0] = rest[j]; v[1] = rest[j + 1]; v[2] = rest[j + 2];
          var w0 = W[j], w1 = W[j + 1], w2 = W[j + 2], wc = 1 - w0 - w1 - w2; if (wc < 0) wc = 0;
          // the chest's share: turned with the body about the vertical line through the base of the neck
          r = rot(qB, [v[0] - ax[0], v[1], v[2] - ax[2]]); acc[0] = wc * (r[0] + ax[0]); acc[1] = wc * r[1]; acc[2] = wc * (r[2] + ax[2]);
          if (N) { r = rot(qB, [restN[j], restN[j + 1], restN[j + 2]]); accN[0] = wc * r[0]; accN[1] = wc * r[1]; accN[2] = wc * r[2]; }
          for (b = 0; b < 3; b++) { w = b === 0 ? w0 : b === 1 ? w1 : w2; if (w <= 0) continue;
            o = restPiv[b]; r = rot(R[b], [v[0] - o[0], v[1] - o[1], v[2] - o[2]]); m = pv[b];
            acc[0] += w * (r[0] + m[0]); acc[1] += w * (r[1] + m[1]); acc[2] += w * (r[2] + m[2]);
            if (N) { r = rot(R[b], [restN[j], restN[j + 1], restN[j + 2]]); accN[0] += w * r[0]; accN[1] += w * r[1]; accN[2] += w * r[2]; } }
          P[j] = acc[0]; P[j + 1] = acc[1]; P[j + 2] = acc[2];
          if (N) { m = Math.sqrt(accN[0] * accN[0] + accN[1] * accN[1] + accN[2] * accN[2]) || 1; N[j] = accN[0] / m; N[j + 1] = accN[1] / m; N[j + 2] = accN[2] / m; }
        }
        var hq = R[2], ht = sub(pv[2], rot(hq, restPiv[2]));
        return { q: hq, t: ht };
      },
    };
  }
  window.K8HeadPose = { make: make, solve: solve, SHARE: SHARE, rot: rot, heading: heading, meanHeading: meanHeading };
})();

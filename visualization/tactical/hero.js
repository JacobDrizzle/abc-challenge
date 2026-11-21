// hero.js: the hero as a PART (SPEC B3, B5, B6, G4 "hook"; ticket 04a). A bust wearing the headset that replays the clip.
//
//   import { createHero } from './hero.js';
//   const hero = createHero({ THREE, clip: window.K8_HERO_CLIP, head: window.K8_HERO_HEAD, pose: window.K8HeroPose });
//   scene.add(hero.group);  hero.update(seconds);  renderer.render(scene, camera);
//
// It knows nothing about the page, the deck, the canvas or the renderer: its caller owns the clock and the drawing.
// Returned: { group, update(clockSeconds), setPaused(bool), sampleIndex(), dispose() }, plus, read only, for the caller and the
// checks: period (seconds of clip time before the loop restarts), skin (the skinned Mesh), headset (the Group fixed to the head
// bone), anchors (skin vertex indices that sit under the headset), mmPerUnit (millimetres per mesh unit), bodyRef (radians).
//
// Axes are the head mesh's own (hero/head.js, made in Blender): up +y, the face looks along −z, units = H.unit metres·per_metre.
// The group itself is never turned or moved: all turning is done by the neck rig (hero/headpose.js), exactly as the viewer does.
//
// Ticket 04b adds, all inside the same group:
//   ghost     a second, translucent copy of the headset only, posed at the model's prediction pred[i] (200 ms ahead);
//   pointers  a thin forward line on each headset, 1.5 head-lengths long from the headset's front;
//   arc       a great-circle arc between the two pointer tips, centred on the head (its angle is the angle between the two
//             facing directions; see poseAt).
// Extra methods: poseAt(clockSeconds) (poses even while paused: for checks), measure() (the numbers the checks compare).

// The page's "prediction" accent (index.html :root --blue). The ghost, its pointer and the arc take their colour from here only.
export const PREDICTION_ACCENT = '#dff140';   // the page's one accent (tactical.css --blue: acid yellow, 16 Nov 2026; story/ keeps #84b2ff)
// The look (ticket 05, SPEC B6 Materials). Physically based materials: `roughness` is how matte a surface is, `clearcoat` a thin
// glossy layer on top (the visor, the shell), `sheen` a soft glow at grazing angles that keeps skin and cloth from looking like
// plastic. Colours are restrained, and the skin tone is ONE constant so the team can change it.
export const SKIN_TONE = '#c8a590';
const LOOK = {
  skin:    { color: SKIN_TONE, roughness: 0.58, sheen: 0.3, sheenColor: '#ff9d80', sheenRoughness: 0.6 },
  cloth:   { color: '#474c57', roughness: 0.92, sheen: 0.5, sheenColor: '#8f97a6', sheenRoughness: 0.8 },
  body:    { color: '#ecebe7', roughness: 0.34, clearcoat: 0.5, clearcoatRoughness: 0.25 },          // the headset's shell
  plate:   { color: '#08090b', roughness: 0.1, clearcoat: 1, clearcoatRoughness: 0.04 },             // the visor: reflects the lights
  pad:     { color: '#2b2c2f', roughness: 0.85 },
  strap:   { color: '#33363b', roughness: 0.8 }, top: { color: '#33363b', roughness: 0.8 }, backpad: { color: '#33363b', roughness: 0.8 },
};
const TRUE_POINTER = '#dde4ee';           // the true headset's pointer: a neutral light grey
// tactical/ (16 Nov 2026, Nic: "outline only"): with the detailed headset a filled ghost showed as yellow bands wherever it sat a few
// mm outside the real shell. The ghost is now almost only its outline: a near-clear fill, and a rim where the surface turns edge-on.
const GHOST_OPACITY = 0.03;               // the fill: barely there
const GHOST_EDGE = 0.95;                  // the rim's opacity, where the surface is edge-on to the viewer
const GHOST_RIM = [0.5, 0.82];            // the rim: 1 − |n·v| from 0.5 (starts) to 0.82 (full), a thin line rather than a soft glow
const POINTER_HEADS = 1.5;                // pointer length in head-lengths (SPEC B4)
const ARC_STEPS = 48;                     // line segments in the arc: smooth at any size up to 90°

// ---- the clip's quaternions in three.js space
// The clip stores (w, x, y, z) in Unity's left-handed axes (y up, z forward). three.js is right-handed with the face along −z,
// so mirroring the z axis (Unity z → three −z) turns the rotation (w, x, y, z) into (w, −x, −y, z): THREE.Quaternion(−x, −y, z, w).
// This is the viewer's own mapping (viewer.mjs: headM.quaternion.set(-q[1], -q[2], q[3], q[0])), so both headsets agree.
// Used for both q (the truth) and pred (the prediction); each list is kept in one hemisphere on its own.
function clipToThree(THREE, list) {
  const out = [];
  for (let i = 0; i < list.length; i++) {
    const [w, x, y, z] = list[i], q = new THREE.Quaternion(-x, -y, z, w).normalize();
    // q and −q are the same rotation; keep neighbours in the same hemisphere so each 10 ms step is the short way round
    if (i && q.dot(out[i - 1]) < 0) q.set(-q.x, -q.y, -q.z, -q.w);
    out.push(q);
  }
  return out;
}

export function createHero({ THREE, clip, head, pose, accent = PREDICTION_ACCENT }) {
  if (!THREE || !clip || !head || !pose) throw new Error('createHero needs THREE, clip, head and pose');
  const rigPose = pose.make();                 // the snapshot's poser, bound to window.K8_HERO_HEAD
  if (!rigPose) throw new Error('createHero: the head has no skin or no bones');
  const rate = clip.rate_hz || 100, Q = clipToThree(THREE, clip.q), PRED = clipToThree(THREE, clip.pred), N = Q.length;
  const horizon = clip.horizon || 20;
  // The loop: samples 0 … N−1 are 10 ms apart, so the recorded motion spans (N − 1) / rate seconds. Clip time t ∈ [0, period]
  // shows the recording at fractional sample t·rate; there is no interval from the last sample back to the first (no blending
  // across the loop: the caller jumps from t = period to t = 0, see stage_task.js).
  const period = (N - 1) / rate;
  // The body's reference heading: the circular mean of the clip's headings (how far round "up" each pose faces). The neck rig
  // turns the body only when the head turns more than about 69° away from this, so for this clip the chest stays still.
  const bodyRef = pose.meanHeading(Q.map(q => pose.heading([q.x, q.y, q.z, q.w])));

  // ---- the meshes, built as incoming/head/head3d.js builds them, with a plain lit look (a later ticket does the real look)
  const u = head.unit, group = new THREE.Group(), headset = new THREE.Group(), materials = [], geometries = [];
  group.name = 'hero'; headset.name = 'hero-headset';
  const mat = (color, basic, look) => { const m = basic ? new THREE.MeshBasicMaterial({ color }) : new THREE.MeshPhysicalMaterial({ color, roughness: 0.72, metalness: 0, ...(LOOK[look] || {}) }); materials.push(m); return m; };
  let skin = null, restN = null;
  for (const p of head.parts) {
    const g = new THREE.BufferGeometry(), a = new Float32Array(p.p.length);
    for (let i = 0; i < a.length; i++) a[i] = p.p[i] * u;
    g.setAttribute('position', new THREE.BufferAttribute(a, 3)); geometries.push(g);
    if (p.group === 'skin') {
      // one continuous mesh: index list i = skin (head, neck), i2 = clothing; two material groups, one colour each
      g.setIndex(p.i.concat(p.i2)); g.addGroup(0, p.i.length, 0); g.addGroup(p.i.length, p.i2.length, 1);
      g.computeVertexNormals();                // smooth normals, computed ONCE on the rest pose; the rig turns them with the bones
      restN = g.attributes.normal.array.slice();
      g.attributes.position.setUsage(THREE.DynamicDrawUsage); g.attributes.normal.setUsage(THREE.DynamicDrawUsage);
      skin = new THREE.Mesh(g, [mat(p.color, false, 'skin'), mat(p.color2, false, 'cloth')]);
      skin.name = 'hero-skin';
      skin.frustumCulled = false;              // its vertices move every frame, so stored bounds would go stale
    } else {
      g.setIndex(p.i); g.computeVertexNormals();
      const m = new THREE.Mesh(g, mat(p.color, p.name === 'led', p.name)); m.name = 'hero-headset-' + p.name; headset.add(m);
    }
  }
  if (!skin) throw new Error('createHero: no skin part in the head');
  // BOTH go in one group that does not turn; pose() below moves the skin's vertices and places the headset.
  group.add(skin, headset);

  // Skin vertices that sit under the headset (within 20 mm of any headset vertex at rest). They belong wholly to the head bone,
  // so the headset must carry them exactly: the checks measure how far they slide from the headset over the whole clip.
  const mmPerUnit = 1000 / head.per_metre, anchors = [];
  {
    const hv = []; headset.children.forEach(m => { const a = m.geometry.attributes.position.array; for (let i = 0; i < a.length; i += 3) hv.push(a[i], a[i + 1], a[i + 2]); });
    const sp = skin.geometry.attributes.position.array, lim = (20 / mmPerUnit) ** 2;
    for (let i = 0, k = 0; i < sp.length; i += 3, k++) {
      for (let j = 0; j < hv.length; j += 3) { const dx = sp[i] - hv[j], dy = sp[i + 1] - hv[j + 1], dz = sp[i + 2] - hv[j + 2]; if (dx * dx + dy * dy + dz * dz < lim) { anchors.push(k); break; } }
    }
  }

  // ---- the ghost: the same headset meshes (shared geometry), drawn as ONE clean translucent layer.
  // A translucent object made of several overlapping parts (shell, plate, strap…) normally shows its inner surfaces through its
  // outer ones and flickers as three.js re-sorts the parts. So it is drawn twice, after everything opaque:
  //   1. a depth pre-pass: every ghost mesh writes only depth (no colour), so the depth buffer holds the ghost's nearest surface;
  //   2. the colour pass: the same meshes again at ~35% opacity, drawn only where their depth EQUALS that nearest surface, so each
  //      pixel gets exactly one layer of ghost colour, whatever the order of the parts.
  // The real headset and the bust are opaque and drawn before both passes, so where the ghost is behind them it is simply hidden,
  // and where it is in front they show through it once.
  const ghost = new THREE.Group(); ghost.name = 'hero-ghost';
  const ghostColor = new THREE.Color(accent);
  // Both passes are pushed very slightly back (polygon offset, the same for both so they still match each other exactly): where the
  // ghost and the real headset coincide (a near-perfect prediction) the real one wins cleanly instead of the two flickering in stripes.
  const OFFSET = { polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 4 };
  const depthMat = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true, transparent: true, ...OFFSET }); materials.push(depthMat);
  // The colour pass: a small shader. a = base opacity + an edge term that grows where the surface turns away from the viewer
  // (1 − |n·v|, the usual "Fresnel-like" rim), so the ghost's outline reads clearly at any size without a thick fill.
  const ghostMat = new THREE.ShaderMaterial({
    uniforms: { uColor: { value: ghostColor }, uOpacity: { value: GHOST_OPACITY }, uEdge: { value: GHOST_EDGE }, uRim: { value: new THREE.Vector2(...GHOST_RIM) } },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = -mv.xyz; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uOpacity; uniform float uEdge; uniform vec2 uRim; varying vec3 vN; varying vec3 vV;
      void main() { float e = smoothstep(uRim.x, uRim.y, 1.0 - abs(dot(normalize(vN), normalize(vV))));
        gl_FragColor = vec4(mix(uColor, vec3(1.0), 0.25 * e), min(1.0, uOpacity + uEdge * e));
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, depthFunc: THREE.LessEqualDepth, side: THREE.FrontSide, ...OFFSET,
  });
  materials.push(ghostMat);
  for (const m of headset.children) {
    const d = new THREE.Mesh(m.geometry, depthMat); d.renderOrder = 1; d.name = m.name.replace('hero-headset', 'hero-ghost-depth');
    const c = new THREE.Mesh(m.geometry, ghostMat); c.renderOrder = 2; c.name = m.name.replace('hero-headset', 'hero-ghost');
    ghost.add(d, c);
  }

  // ---- the pointers and the arc, in the head's own (rest) axes, where the headset meshes are defined
  // head length: front-to-back extent of the skin that belongs (mostly) to the head bone; front: the headset's foremost point,
  // on the centre line (x = 0) at the height of the shell's middle. Facing direction = −z.
  let zLo = Infinity, zHi = -Infinity;
  { const sk = head.parts.find(p => p.group === 'skin');
    for (let k = 0; k < sk.w.length / 3; k++) if (sk.w[3 * k + 2] > 200) { const z = sk.p[3 * k + 2] * u; if (z < zLo) zLo = z; if (z > zHi) zHi = z; } }
  const headLength = zHi - zLo;
  const hsBox = new THREE.Box3(); headset.children.forEach(m => { m.geometry.computeBoundingBox(); hsBox.union(m.geometry.boundingBox); });
  const shell = headset.children.find(m => /body$/.test(m.name)) || headset.children[0];
  const front = new THREE.Vector3(0, (shell.geometry.boundingBox.min.y + shell.geometry.boundingBox.max.y) / 2, hsBox.min.z);
  const tipLocal = front.clone().add(new THREE.Vector3(0, 0, -POINTER_HEADS * headLength));
  // The arc's centre, in head axes: the point of the pointer's own line level with the head bone's pivot (the joint under the
  // skull the head turns about). It lies ON the pointer line, so the pointer runs straight out from it, and the arc from the true
  // tip, at radius |tip − centre|, round to the ghost's facing direction sweeps exactly the angle between the two directions.
  const headPivot = rigPose.pivots[2];
  const centreLocal = new THREE.Vector3(front.x, front.y, headPivot[2]), radius = centreLocal.z - tipLocal.z;
  const lineGeo = (a, b) => { const g = new THREE.BufferGeometry().setFromPoints([a, b]); geometries.push(g); return g; };
  const truePtr = new THREE.Line(lineGeo(front, tipLocal), new THREE.LineBasicMaterial({ color: TRUE_POINTER }));
  const ghostPtr = new THREE.Line(lineGeo(front, tipLocal), new THREE.LineBasicMaterial({ color: ghostColor, transparent: true, opacity: 0.95, depthWrite: false }));
  materials.push(truePtr.material, ghostPtr.material);
  truePtr.name = 'hero-pointer'; ghostPtr.name = 'hero-ghost-pointer'; truePtr.renderOrder = 3; ghostPtr.renderOrder = 3;
  headset.add(truePtr); ghost.add(ghostPtr);
  // arc buffer: ARC_STEPS + 1 points on the great circle, then one more point: the ghost's tip itself (see poseAt)
  const arcPos = new Float32Array((ARC_STEPS + 2) * 3), arcGeo = new THREE.BufferGeometry();
  arcGeo.setAttribute('position', new THREE.BufferAttribute(arcPos, 3).setUsage(THREE.DynamicDrawUsage)); geometries.push(arcGeo);
  const arc = new THREE.Line(arcGeo, new THREE.LineBasicMaterial({ color: ghostColor, transparent: true, opacity: 0.8, depthWrite: false }));
  materials.push(arc.material); arc.name = 'hero-arc'; arc.renderOrder = 3; arc.frustumCulled = false;
  group.add(ghost, arc);

  // ---- posing
  // The skinning (hero/headpose.js, the head session's rule, unchanged):
  //  * twist/swing split: the head's rotation is split into a twist about "up" (turning left/right) and a swing (nodding and
  //    leaning); the two neck bones take fixed shares of each and the head bone takes the rest, so the head's own rotation is
  //    exactly the recorded one, and the neck bends instead of the bust turning as one block;
  //  * linear blend skinning: every skin vertex is moved by each bone that owns it (rotate about that bone's pivot, then follow
  //    the pivot), and the results are averaged with the vertex's weights; whatever weight is left belongs to the chest, which
  //    turns only with the body, and the body turns only past the neck's limit relative to bodyRef.
  // apply() writes the skinned positions and normals into the geometry and returns the head bone's rigid transform, which is
  // exactly where the headset goes.
  const qa = new THREE.Quaternion(), qp = new THREE.Quaternion(), arr = [0, 0, 0, 1];
  const c1 = new THREE.Vector3(), d1 = new THREE.Vector3(), d2 = new THREE.Vector3(), tip1 = new THREE.Vector3(), tip2 = new THREE.Vector3();
  const qTurn = new THREE.Quaternion(), qPart = new THREE.Quaternion(), qId = new THREE.Quaternion(), dk = new THREE.Vector3();
  let paused = false, f = 0, lastF = NaN, arcDeg = 0;
  function poseAt(fs) {
    if (fs === lastF) return; lastF = fs;
    const i = Math.min(N - 2, Math.floor(fs)), t = fs - i;
    // Spherical linear interpolation (slerp) between the two neighbouring 10 ms samples: it moves along the shortest great-circle
    // arc between them at a constant angular rate, so every pose shown lies on the arc between two recorded samples (three.js's
    // slerp takes the short way round, and the samples were put in one hemisphere above).
    qa.slerpQuaternions(Q[i], Q[i + 1], t);
    arr[0] = qa.x; arr[1] = qa.y; arr[2] = qa.z; arr[3] = qa.w;
    const pos = skin.geometry.attributes.position, nor = skin.geometry.attributes.normal;
    const hb = rigPose.apply(arr, pos.array, restN, nor.array, bodyRef);
    pos.needsUpdate = true; nor.needsUpdate = true;
    headset.quaternion.set(hb.q[0], hb.q[1], hb.q[2], hb.q[3]); headset.position.set(hb.t[0], hb.t[1], hb.t[2]);

    // The ghost: the prediction made at this moment for 200 ms ahead, interpolated between neighbouring samples exactly like the
    // truth, and placed through the SAME neck rule (pose.solve: the neck bones take their shares, the head bone the rest), so it
    // sits where the real headset will be if the prediction comes true. Only the head bone's transform is needed (no skin).
    qp.slerpQuaternions(PRED[i], PRED[i + 1], t);
    const s = pose.solve([qp.x, qp.y, qp.z, qp.w], rigPose.pivots, bodyRef), gq = s.rot[2], gp = s.piv[2], r = pose.rot(gq, headPivot);
    ghost.quaternion.set(gq[0], gq[1], gq[2], gq[3]); ghost.position.set(gp[0] - r[0], gp[1] - r[1], gp[2] - r[2]);

    // The arc (a great circle): centre c1 = the true head's arc centre; it starts at the true pointer's tip (c1 + radius·d1) and
    // turns, in ARC_STEPS equal steps of one rotation, to c1 + radius·d2, where d1 and d2 are the two facing directions. So the
    // angle it sweeps is exactly the angle between where the head faces now and where the model says it will face.
    // The neck carries the head a little as it turns, so the ghost's own centre sits a few millimetres from c1 (median 0.005,
    // at most 0.023 mesh units over this clip): one last short straight piece closes the line onto the ghost's real tip.
    c1.copy(centreLocal).applyQuaternion(headset.quaternion).add(headset.position);
    d1.set(0, 0, -1).applyQuaternion(headset.quaternion); d2.set(0, 0, -1).applyQuaternion(ghost.quaternion);
    tip1.copy(tipLocal).applyQuaternion(headset.quaternion).add(headset.position);
    tip2.copy(tipLocal).applyQuaternion(ghost.quaternion).add(ghost.position);
    qTurn.setFromUnitVectors(d1, d2);
    for (let k = 0; k <= ARC_STEPS; k++) {
      qPart.slerpQuaternions(qId, qTurn, k / ARC_STEPS);
      dk.copy(d1).applyQuaternion(qPart).multiplyScalar(radius).add(c1);
      arcPos[3 * k] = dk.x; arcPos[3 * k + 1] = dk.y; arcPos[3 * k + 2] = dk.z;
    }
    arcPos.set([tip2.x, tip2.y, tip2.z], 3 * (ARC_STEPS + 1));
    arcGeo.attributes.position.needsUpdate = true;
    arcDeg = d1.angleTo(d2) * 180 / Math.PI;
  }
  function poseClock(clockSeconds) {
    let t = clockSeconds;                                     // t = period itself shows the last sample
    if (!(t >= 0 && t <= period)) { t %= period; if (t < 0) t += period; if (!(t >= 0)) t = 0; }
    f = Math.min(N - 1, Math.max(0, t * rate));
    poseAt(f);
  }
  poseAt(0);

  const wq = new THREE.Quaternion(), wv = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3();
  const q4 = q => [q.x, q.y, q.z, q.w];
  return {
    group, skin, headset, ghost, arc, anchors, period, mmPerUnit, bodyRef, samples: N, rate, horizon, headLength, accent,
    // Pose the bust for clip time clockSeconds (any value: it is wrapped into [0, period]). Ignored while paused.
    update(clockSeconds) { if (!paused) poseClock(clockSeconds); },
    // The same, also while paused (for checks and test hooks).
    poseAt: poseClock,
    setPaused(p) { paused = !!p; },
    sampleIndex() { return Math.floor(f); },
    fraction() { return f; },
    // What the checks compare, measured on the drawn objects (world space, after the group's own world matrix):
    //   headset, ghost  world quaternions [x, y, z, w];   tips  world positions of the two pointer tips (the true one, the ghost's)
    //   arcDeg          the angle the arc's first and last great-circle points make at its centre, read back from the line buffer
    //   centre          the arc's centre (world);  gap  the closing piece's length in mesh units (ghost centre vs true centre)
    measure() {
      group.updateMatrixWorld(true);
      const M = group.matrixWorld, at = k => new THREE.Vector3(arcPos[3 * k], arcPos[3 * k + 1], arcPos[3 * k + 2]);
      const cw = c1.clone().applyMatrix4(M);
      a.copy(at(0)).applyMatrix4(M).sub(cw); b.copy(at(ARC_STEPS)).applyMatrix4(M).sub(cw);
      return {
        f, headset: q4(headset.getWorldQuaternion(wq)), ghost: q4(ghost.getWorldQuaternion(wq)),
        tips: [tip1.clone().applyMatrix4(M), tip2.clone().applyMatrix4(M)], centre: cw,
        arcDeg: a.angleTo(b) * 180 / Math.PI, arcDegPosed: arcDeg, gap: at(ARC_STEPS).distanceTo(tip2),
        radius, headLength, pointerLength: tipLocal.distanceTo(front),
      };
    },
    // tactical/ (16 Nov 2026): swap the modelled box headset for a real one (hero_model.js: Elin's CC BY "Meta Quest 3", logo
    // removed). `model` is an Object3D in the head's rest space, in page units, like the box parts. The box meshes are hidden,
    // not removed: the pointer, the shell the journey reads its anchors from, and the skin anchors keep their geometry.
    // The ghost is rebuilt from a clone of the model, drawn with the same depth pre-pass + one translucent layer as before.
    useModel(model) {
      for (const m of headset.children) if (m.isMesh) m.visible = false;
      headset.add(model);
      for (const c of [...ghost.children]) if (c.isMesh) ghost.remove(c);
      const g = model.clone(true), meshes = [];
      g.traverse(o => { if (o.isMesh) meshes.push(o); });
      for (const o of meshes) {
        o.material = ghostMat; o.renderOrder = 2;
        const d = new THREE.Mesh(o.geometry, depthMat); d.renderOrder = 1;
        d.position.copy(o.position); d.quaternion.copy(o.quaternion); d.scale.copy(o.scale); o.parent.add(d);
      }
      ghost.add(g);
      model.traverse(o => { if (o.isMesh) { geometries.push(o.geometry); materials.push(...(Array.isArray(o.material) ? o.material : [o.material])); } });
    },
    dispose() { geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose()); group.removeFromParent(); group.clear(); },
  };
}

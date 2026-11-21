// board_textures.js: surface detail for the replica board (board_q3.glb), drawn in code on 2D canvases. No image files.
// (16 Nov 2026, Nic: "improve textures to make them more realistic")
//
// Blender gave every part a flat colour. Real boards read as real because of small surface detail, so three textures are made here:
//   1. the solder mask: the copper under the mask (signal traces in bundles, ground pours, vias) shows through as slightly raised,
//      slightly glossier, slightly lighter shapes, plus a fine grain. One canvas covers the whole board, mapped from straight above.
//   2. the shield frames: stamped tin with a fine dimpled pattern (the photo's frames look knurled), as a small repeating tile.
//   3. a fine grain for chip tops and connector plastic, so they are not perfectly uniform.
// Each texture is used three ways: as the colour (map), as a bump map (fake relief from brightness) and as a roughness map
// (three.js reads roughness from the green channel). UVs are not in the glb, so they are made here: a top-down projection of each
// vertex's x and z (the board's plane, in the glb's millimetres).
//
// The patterns are seeded, so the board looks the same on every load. They are plausible PCB detail, not the real board's copper.

export function applyBoardTextures(THREE, root, renderer) {
  let seed = 23; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const aniso = renderer.capabilities.getMaxAnisotropy();
  const tex = (canvas, srgb, repeat) => {
    const t = new THREE.CanvasTexture(canvas); t.anisotropy = aniso;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
    return t;
  };
  // top-down UVs: u from x, v from z, over the box [x0, x0 + w] x [z0, z0 + h] (millimetres); RepeatWrapping does the tiling
  const planarUV = (geo, x0, z0, w, h) => {
    const p = geo.attributes.position, uv = new Float32Array(p.count * 2);
    for (let i = 0; i < p.count; i++) { uv[2 * i] = (p.getX(i) - x0) / w; uv[2 * i + 1] = 1 - (p.getZ(i) - z0) / h; }
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  };

  // ---------------------------------------------------------------- 1. the solder mask
  const BX = -56, BZ = -31.5, BW = 112, BH = 63, PX = 18;            // board area (mm) and canvas pixels per mm
  const W = Math.round(BW * PX), H = Math.round(BH * PX);
  const hc = document.createElement('canvas'); hc.width = W; hc.height = H; const g = hc.getContext('2d');
  const P = (x, z) => [(x - BX) * PX, (z - BZ) * PX];
  g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  // ground pours: big copper areas with a clearance gap, slightly raised (mid grey)
  g.fillStyle = '#6a6a6a';
  for (let k = 0; k < 14; k++) {
    const [x, z] = P(-52 + rnd() * 98, -28 + rnd() * 52), w = (6 + rnd() * 18) * PX, h = (4 + rnd() * 12) * PX;
    g.fillRect(x, z, w, h);
  }
  // signal traces: bundles of parallel lines that run straight, then turn 45 degrees, like routed copper
  g.lineCap = 'round'; g.lineJoin = 'round';
  for (let b = 0; b < 70; b++) {
    let x = -52 + rnd() * 104, z = -28 + rnd() * 56;
    const n = 2 + Math.floor(rnd() * 7), pitch = .32 + rnd() * .1, wide = .1 + rnd() * .1;
    let dir = Math.floor(rnd() * 4) * Math.PI / 2;
    const path = [[x, z]];
    for (let s = 0; s < 3 + Math.floor(rnd() * 3); s++) {
      const len = 2 + rnd() * 12; x += Math.cos(dir) * len; z += Math.sin(dir) * len; path.push([x, z]);
      dir += (rnd() < .5 ? -1 : 1) * Math.PI / 4;
    }
    // clearance first (black, wider) so the bundle cuts through a pour like a real routed trace, then the copper
    for (const [col, extra] of [['#000', .5], ['#ffffff', 0]]) {
      g.strokeStyle = col;
      for (let i = 0; i < n; i++) {
        const off = (i - (n - 1) / 2) * pitch; g.lineWidth = (wide + extra) * PX; g.beginPath();
        path.forEach(([px_, pz], j) => {
          const [a, bb] = path[Math.min(j + 1, path.length - 1)], [c, d] = path[Math.max(j - 1, 0)];
          const dx = a - c, dz = bb - d, L = Math.hypot(dx, dz) || 1;
          const [u, v] = P(px_ - dz / L * off, pz + dx / L * off);
          j ? g.lineTo(u, v) : g.moveTo(u, v);
        });
        g.stroke();
      }
    }
  }
  // vias (tented): small bright dots, a few in a row at bundle ends
  g.fillStyle = '#fff';
  for (let k = 0; k < 900; k++) { const [x, z] = P(-54 + rnd() * 108, -30 + rnd() * 60); g.beginPath(); g.arc(x, z, .17 * PX, 0, Math.PI * 2); g.fill(); }

  // colour and roughness from the height: copper under the mask is a shade lighter and glossier; plus a fine grain everywhere
  const hd = g.getImageData(0, 0, W, H).data;
  const cc = document.createElement('canvas'); cc.width = W; cc.height = H; const cg = cc.getContext('2d'); const cd = cg.createImageData(W, H);
  const rc = document.createElement('canvas'); rc.width = W; rc.height = H; const rg = rc.getContext('2d'); const rd = rg.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const h = hd[4 * i] / 255, n = (rnd() - .5);
    cd.data[4 * i] = 21 + h * 9 + n * 3; cd.data[4 * i + 1] = 20 + h * 8 + n * 3; cd.data[4 * i + 2] = 15 + h * 5 + n * 3; cd.data[4 * i + 3] = 255;
    const r = (.5 - h * .16 + n * .06) * 255; rd.data[4 * i] = rd.data[4 * i + 1] = rd.data[4 * i + 2] = r; rd.data[4 * i + 3] = 255;
  }
  cg.putImageData(cd, 0, 0); rg.putImageData(rd, 0, 0);
  const maskMap = tex(cc, true), maskBump = tex(hc, false), maskRough = tex(rc, false);

  // ---------------------------------------------------------------- 2. stamped tin (shield frames): a dimple tile, 0.6 mm square
  const T = 64, tc = document.createElement('canvas'); tc.width = tc.height = T; const tg = tc.getContext('2d');
  tg.fillStyle = '#808080'; tg.fillRect(0, 0, T, T);
  for (const [cx, cy] of [[16, 16], [48, 48], [48, 16], [16, 48]]) {
    const gr = tg.createRadialGradient(cx, cy, 1, cx, cy, 13); gr.addColorStop(0, '#2a2a2a'); gr.addColorStop(1, '#808080');
    tg.fillStyle = gr; tg.beginPath(); tg.arc(cx, cy, 13, 0, Math.PI * 2); tg.fill();
  }
  const tinBump = tex(tc, false, true);
  const trc = document.createElement('canvas'); trc.width = trc.height = T; const trg = trc.getContext('2d');
  trg.drawImage(tc, 0, 0); trg.globalCompositeOperation = 'lighter'; trg.fillStyle = 'rgba(40,40,40,1)'; trg.fillRect(0, 0, T, T);
  const tinRough = tex(trc, false, true);                            // dimples a little rougher than the flats

  // ---------------------------------------------------------------- 3. fine grain tile (chip tops, plastic), 3 mm square
  const G = 256, gc = document.createElement('canvas'); gc.width = gc.height = G; const gg = gc.getContext('2d'); const gd = gg.createImageData(G, G);
  for (let i = 0; i < G * G; i++) { const v = 128 + (rnd() - .5) * 46; gd.data[4 * i] = gd.data[4 * i + 1] = gd.data[4 * i + 2] = v; gd.data[4 * i + 3] = 255; }
  gg.putImageData(gd, 0, 0); const grain = tex(gc, false, true);

  root.traverse(o => {
    if (!o.isMesh) return; const m = o.material, n = m.name, geo = o.geometry;
    if (n === 'solder_mask') {
      planarUV(geo, BX, BZ, BW, BH);
      m.color.set(0xffffff); m.map = maskMap; m.bumpMap = maskBump; m.bumpScale = .8; m.roughnessMap = maskRough; m.roughness = 1;
    } else if (n === 'shield_tin') {
      planarUV(geo, 0, 0, .6, .6);
      m.bumpMap = tinBump; m.bumpScale = 1.2; m.roughnessMap = tinRough; m.roughness = .6;   // map mean ~0.66: ~0.4 overall
    } else if (n === 'chip_epoxy' || n === 'connector_plastic' || n === 'soc_top') {
      planarUV(geo, 0, 0, 3, 3);
      m.roughnessMap = grain; m.roughness = n === 'soc_top' ? .36 : 1.5;    // the map averages 0.5, so this keeps the old mean
      m.bumpMap = grain; m.bumpScale = n === 'soc_top' ? .15 : .4;
    }
    m.needsUpdate = true;
  });

  // ---------------------------------------------------------------- 4. flicker fixes (Nic, 16 Nov: "aspects are flashing")
  // (a) Thin layers (chip markings, silkscreen, label ink, the gold rings, the copper-over-mask dots) sit only a few thousandths of a
  //     mm above the surface under them. At the board's distance the depth buffer cannot always tell which is in front, so they
  //     flickered as the board swayed ("z-fighting"). A polygon offset pulls them a little toward the camera in depth only, so they
  //     always win, without moving them on screen.
  const DECALS = new Set(['laser_marking', 'silkscreen', 'label_ink', 'gold_plating', 'mask_over_copper', 'led_phosphor', 'cap_tops']);
  root.traverse(o => {
    if (o.isMesh && DECALS.has(o.material.name)) { const m = o.material; m.polygonOffset = true; m.polygonOffsetFactor = -2; m.polygonOffsetUnits = -4; m.needsUpdate = true; }
  });
  // (b) The two 2D codes (the serial label's and the memory chip's) were hundreds of tiny raised squares, each smaller than a screen
  //     pixel, so they shimmered. They become flat textures instead: a texture is filtered (mipmapped) when it is small on screen,
  //     geometry is not. The square meshes are hidden and a flat quad with the drawn code takes their place, a hair above.
  const codeDecal = (name, cells, ink, paper) => {
    const src = root.getObjectByName(name); if (!src || !src.isMesh) return;
    src.geometry.computeBoundingBox(); const bb = src.geometry.boundingBox;
    const S = 512, c = document.createElement('canvas'); c.width = c.height = S; const q = c.getContext('2d');
    q.fillStyle = paper; q.fillRect(0, 0, S, S); q.fillStyle = ink; const cs = S / cells;
    for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) {
      const finder = (i < 7 && j < 7) || (i >= cells - 7 && j < 7) || (i < 7 && j >= cells - 7);
      let on;
      if (finder) {                                                 // the three corner squares of a QR-style code
        const ii = i >= cells - 7 ? i - (cells - 7) : i, jj = j >= cells - 7 ? j - (cells - 7) : j;
        on = ii === 0 || ii === 6 || jj === 0 || jj === 6 || (ii >= 2 && ii <= 4 && jj >= 2 && jj <= 4); }
      else on = rnd() < .5;
      if (on) q.fillRect(i * cs, j * cs, cs + .5, cs + .5);
    }
    const t = tex(c, true); t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter;
    const w = bb.max.x - bb.min.x, d = bb.max.z - bb.min.z;
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshStandardMaterial({ map: t, roughness: .6, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 }));
    quad.rotation.x = -Math.PI / 2; quad.position.set((bb.min.x + bb.max.x) / 2, bb.max.y + .002, (bb.min.z + bb.max.z) / 2);
    src.parent.add(quad); src.visible = false;
  };
  codeDecal('serial_code', 21, '#0a0a0a', '#ecece9');
  codeDecal('mem_code', 14, '#7c7c7c', '#161618');
}

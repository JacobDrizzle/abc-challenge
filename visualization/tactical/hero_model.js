// hero_model.js (tactical/, 16 Nov 2026): the real headset for the hero, replacing the modelled box headset.
// Model: "Meta Quest 3" by Elin (sketchfab.com/ElinHohler), CC BY 4.0, https://sketchfab.com/3d-models/meta-quest-3-65a813833dc04eeeb7d33bdca58c184c
// Modified: Meta logo removed from the textures, fitted to this head; the full 76k-triangle mesh, Draco-compressed, 1024 px JPEG
// textures (exported headless from models/headset.blend; credits in models/CREDITS.md).
// The .glb is exported in the head's rest space in page units. Blender's glTF export turns its (x, y, z) into (x, z, −y); the
// page's head axes are Blender's (x, z, y), so the page frame is the glTF frame with z mirrored: the wrapper's scale.z = −1.
// (three.js draws mirrored objects correctly: the renderer flips the front face when an object's matrix has a negative determinant.)
import { GLTFLoader } from './vendor_gltf/loaders/GLTFLoader.js';
import { DRACOLoader } from './vendor_gltf/loaders/DRACOLoader.js';

export const HEADSET_URL = new URL('./assets/headset_q3.glb?r=q4', import.meta.url).href;

// One glTF loader for the page's .glb files (16 Nov, smaller files): both are Draco-compressed (Google's mesh compression), so the
// loader gets three.js's Draco decoder (vendor_gltf/draco/, ~250 KB, fetched once on first use). One decoder is shared by every load.
let DRACO = null;
export function makeGLTFLoader() {
  if (!DRACO) { DRACO = new DRACOLoader(); DRACO.setDecoderPath(new URL('./vendor_gltf/draco/', import.meta.url).href); }
  return new GLTFLoader().setDRACOLoader(DRACO);
}

export async function loadHeadsetModel(THREE, url = HEADSET_URL) {
  const gltf = await makeGLTFLoader().loadAsync(url);
  const wrap = new THREE.Group(); wrap.name = 'hero-headset-model'; wrap.scale.z = -1;
  wrap.add(gltf.scene);
  return wrap;
}

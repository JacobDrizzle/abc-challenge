GLTFLoader.js and BufferGeometryUtils.js copied from three.js r180 (MIT, see LICENSE), 16 Nov 2026.
The page runs three r185 (../vendor/three); the loader uses only public three API. Swap for r185 copies if anything breaks.
DRACOLoader.js (loaders/) and the Draco decoder (draco/draco_wasm_wrapper.js, draco/draco_decoder.wasm) copied from the same three.js
r180 (examples/jsm/libs/draco/gltf/), 16 Nov 2026, so the two glbs can be Draco-compressed (hero_model.js makeGLTFLoader). The decoder
is Google's Draco (Apache 2.0, see draco/README.md); three.js ships it.

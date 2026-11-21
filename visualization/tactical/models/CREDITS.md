# Third-party 3D models (candidates, 16 Nov 2026)

Both are **Creative Commons Attribution 4.0** (CC BY 4.0): free to use, also commercially, as long as the author is credited.
Licence read from the Sketchfab API (`api.sketchfab.com/v3/models/<uid>`, field `license`) on 16 Nov 2026.

| Model | Author | Link | Licence | Size |
|---|---|---|---|---|
| Meta Quest 3 | Elin (ElinHohler) | https://sketchfab.com/3d-models/meta-quest-3-65a813833dc04eeeb7d33bdca58c184c | CC BY 4.0 | 76,260 faces, 8.2 MB .glb |
| Man with Meta Quest 3 headset | onirix | https://sketchfab.com/3d-models/man-with-meta-quest-3-headset-eabf1099363a48ce9a4192539aa499b0 | CC BY 4.0 | 24,405 faces, 9.5 MB .glb (Mixamo rig: Neck, Head) |

Credit line to show on the page if used (CC BY asks for title, author, source and licence):

> "Meta Quest 3" by Elin (sketchfab.com/ElinHohler), "Man with Meta Quest 3 headset" by onirix (sketchfab.com/onirix),
> licensed under CC BY 4.0 (creativecommons.org/licenses/by/4.0). Modified (resized, decimated, textures compressed).

Note: the Quest 3 model shows Meta's logo. CC BY covers the model, not Meta's trademark. Decide whether to keep the logo
(product shown as-is) or blank it in the texture before publishing. "Meta Quest" is Meta's trademark; the page does not
claim any link with Meta.

Source files (not copied into the repo or the page yet): a local downloads folder, a local downloads folder.
Opened for review in `quest_ref.blend`; renders `previews/ref_*.png`.

## Status (16 Nov 2026, evening)

- **In use (Blender only so far):** Elin's "Meta Quest 3", collection QUEST in `headset.blend`, object `quest3`.
  Scale 0.01 × 3.4831 (cm → page units), location (0, -0.23, 0.105). Straps exactly as modelled (not stretched).
- **Head adjusted instead (Nic's call):** `ref_skin` scaled 0.96 and moved y -0.129 so the face meets the cushion; at that
  size no headset vertex is inside the head (checked, every 2nd vertex). The page's head (hero/head.js) needs the same
  0.96 scale + offset when the headset goes in.
- **Meta logo blanked** in base colour and normal map (box u .421-.456, v .690-.720): `quest_tex/quest3_basecolor_nologo.png`,
  `quest_tex/quest3_normal_nologo.png`; `quest_tex/quest3_orm.png` unchanged copy. Originals in a local downloads folder untouched.
- onirix's avatar: not used (reviewed only, in `quest_ref.blend`).

## Live in the page (16 Nov 2026, late)

- `assets/headset_q3.glb` (1.28 MB, 26,690 triangles, 1024 px JPEG textures, logo-free), exported from `headset.blend`
  collection WEB in the head's rest frame (the head's 0.96 fit undone on the headset instead, so hero/head.js is unchanged).
- Loaded by `hero_model.js` (GLTFLoader from three r180 in `vendor_gltf/`), swapped in by `hero.useModel()`; used by the story
  hero (`stage_task.js`) and the journey (`journey/journey.js`). The ghost is a translucent clone of the model.
- Journey processor dot / dive target moved to the middle front sensor pill: (0, -0.115, -0.556) in the head frame.
- Credit shown in the page's Credits section and under the journey's headset panel.

## Fix (16 Nov 2026, 21:10): full mesh, outline ghost

- The 35% decimation smeared the shading (lumpy shell, smeared pill edges). `assets/headset_q3.glb` is now the **full** original
  mesh (76,260 faces, its own custom normals), same placement and logo-free textures: 3.7 MB. The decimated one is kept as
  `assets/headset_q3_dec35.glb` (deleted later that night: nothing used it; it is in git history). Exported headless by a one-off script (Blender GUI crashed on export); `headset.blend` unchanged,
  so the full copy is not saved in it. Cache tag in `hero_model.js` bumped to `?r=q2`.
- The 200 ms ghost is outline-only (hero.js `GHOST_OPACITY/GHOST_EDGE/GHOST_RIM`): a filled copy showed as yellow bands where it
  sat a few mm outside the real shell.

## Also in the 3D spaces (16 Nov 2026, night)

The same headset (assets/headset_q3.glb) now replaces the box headset in the model viewer (`?explore=1`) and on the Data side
(`?explore=1&side=data`): head3d.js loads it once and gives each head a copy (same head mesh and rest frame as the hero, so no refit).
The credit line in the page's Credits section covers these uses too.

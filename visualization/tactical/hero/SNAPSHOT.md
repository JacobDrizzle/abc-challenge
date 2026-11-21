# hero/: snapshot of the head session's head

The hero (the head on deck 1's stage) uses its own copy of the head, so the viewer's older head (`assets/head.js`,
`headpose.js`, `head3d.js`, globals `K8_HEAD`, `K8HeadPose`, `K8HeadGL`) can never collide with it or be replaced by it.

- **Source:** `incoming/head/`, delivered by the head session from `../proposal/` on 10 Nov 2026 at about 01:05
  (see `incoming/head/README.md` and `incoming/head/head_rig.md`).
- **Copied:** 10 Nov 2026, ticket 04a.

| File here | Copied from | What was changed |
|---|---|---|
| `head.js` | `incoming/head/head.js` | Line 2 only: `window.K8_HEAD =` became `window.K8_HERO_HEAD =`. |
| `headpose.js` | `incoming/head/headpose.js` | Line 63: reads `window.K8_HERO_HEAD` instead of `window.K8_HEAD`. Line 94: defines `window.K8HeroPose` instead of `window.K8HeadPose`. |

sha256 of the sources as copied: `head.js` `346688bcc23f5d566128a7691859b0c6548f9f8027798924ecf7e9323cbd0d77`,
`headpose.js` `e7fbf49965c56df212284d9f14838ecd693ddca9d4540eb542e1ad8c0ae1a62c`.

Nothing else was edited: the geometry, weights, bone pivots, the twist/swing shares and the skinning code are byte for byte the
delivered ones. The comments in `headpose.js` (lines 4 and 8) still name `K8_HEAD` / `K8HeadPose`; they describe the original
and were left alone on purpose, so a `diff` against the source shows only the two code lines.

`head3d.js` was not copied: `hero.js` builds the bust itself, using `incoming/head/head3d.js` only as the reference for how the
meshes, the two skin material groups and the headset's attachment to the head bone are made.

Check the snapshot against its source:

```
diff incoming/head/headpose.js hero/headpose.js
diff <(sed 's/^window\.K8_HERO_HEAD = /window.K8_HEAD = /' hero/head.js) incoming/head/head.js
```

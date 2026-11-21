# Quest 3 main board: what is on it, and what we can use to build it (16 Nov 2026)

Research for the journey's board and die levels. Facts below are from public teardowns; nothing here is decided yet.

## 1. The chip

**Qualcomm Snapdragon XR2 Gen 2** (part SXR2230P-100-AB in iFixit's Quest 3 chip ID). Same chip in the Quest 3S.
- 4 nm (Samsung 4LPX reported), 6-core Kryo CPU, Adreno 740 GPU, Hexagon NPU (Qualcomm: about 8x the AI performance
  per watt of XR2 Gen 1, no TOPS figure published).
- 8 GB LPDDR5 next to it (SK hynix, "likely", per iFixit).
- Package size, ball count and die photo: **not public** in what I found. A die-level level stays illustrative.
- Where k8 would run: the Hexagon NPU (or the DSP). Our page can say "a model this size would fit on the headset's NPU"; we
  have not run k8 on an XR2, so the page must not claim it does.

## 2. The board (iFixit "Meta Quest 3 Chip ID", 13 Oct 2023)

Side 1 (the SoC side):
| Part | Role |
|---|---|
| Qualcomm SXR2230P Snapdragon XR2 Gen 2 | applications processor (CPU, GPU, NPU) |
| SK hynix H58G66BK8BX105 (likely) | 8 GB LPDDR5 memory |
| Qualcomm WCN6856 | Wi-Fi 6E |
| Qorvo QM42391 / QM45392 | 2.4 / 5 GHz Wi-Fi front ends |
| Nordic nRF52833 | Bluetooth LE (controllers) |
| AKM AK4333 | audio DAC |
| Triad TS5231 (probably) | IR analog front end |

Side 2:
| Part | Role |
|---|---|
| SK hynix HN8T05DEHKX073 | 128 GB NAND storage |
| Qualcomm PMXR2230 (x4 variants) | power management |
| **STMicroelectronics STM32G071** | Cortex-M0+ microcontroller (nice touch: the challenge sponsor's chip is on the board) |
| Infineon CYPD3177 | USB-C port controller |
| TI TPS65132L, SG Micro SGM62110 | display power, buck-boost |
| Analog Devices MAX98388 | speaker amp |

Not on the main board's chip list: the **headset IMU** (the head-tracking sensor that produces our quaternions). iFixit lists a
TDK ICM-42686-P 6-axis IMU in the *controllers*; the headset's own IMU part is not identified there. Our "Motion sensor" hotspot
should not name a part.

## 3. Free 3D resources (checked licences on 16 Nov)

None of these is a Quest 3 board; no free Quest 3 board model was found on Sketchfab (searched: pcb, motherboard, circuit board,
smartphone motherboard, snapdragon, soc chip, bga, processor chip, vr headset teardown).

| Resource | Licence | Useful for | Notes |
|---|---|---|---|
| KiCad packages3D (gitlab.com/kicad/libraries/kicad-packages3D) | CC BY-SA 4.0 + KiCad exception | real-dimension parts: BGA, QFN, LGA packages, 0201/0402 passives, connectors, shields | the exception covers designs that *use* the parts; redistributing the parts as a collection is SA. A rendered board using a handful is fine with credit. Best source for detail |
| ambientCG (ambientcg.com) | CC0 | metal, solder-mask-like plastics, brushed shield metal | no credit needed (we credit anyway) |
| Sketchfab "Circuit Board TI" (vanart) | CC BY 4.0 | look reference / small board | 19k faces, Substance textures |
| Sketchfab "Drone Control PCB v3" (Plasmabot) | CC BY 4.0 | a clean KiCad-style render of a small board | 89k faces, no textures |
| Sketchfab "Snapdragon 855+ 8K PBR" (R3DHILLS) | CC BY 4.0 | chip shape only | carries Qualcomm's logo and an older chip: would need the logo removed, so little value |
| Sketchfab "Unmarked processor (CPU) + Socket" | CC BY 4.0 | desktop CPU, wrong form factor | skip |

Avoid: many Sketchfab results for "soc chip", "smartphone motherboard" are spam listings ("... tool free download") posing as
models. Do not download those.

iFixit's photos are CC BY-NC-SA: fine to look at as a layout reference, not to put in the page.

## 4. Proposed approach (for Nic to decide)

Build the board ourselves in Blender, laid out like the real Quest 3 board (SoC with LPDDR5 beside it, Wi-Fi block with its RF
shield and front ends, the STM32, PMICs, connectors to the display and camera flex cables), using KiCad package models for
the parts and our own board, traces, vias and silkscreen. Chips get plain dark tops with our own neutral labels, no company logos.
The side panel can still name the real parts as facts, with iFixit as the source.

# Sources behind the experiments

> Working log, kept as written. "Pending" and "team call" describe the state at the time. The submitted model and its recipe are in [`model_card.md`](../../Submission/Motion_Pending_headpose/model_card.md).

This lists every paper (and the few non-paper sources) that was read or named while deciding what to run, from the 14 Oct reading guide to the 6 Nov endpoint refit. Each source is linked to the experiment it fed into. The experiments use the same numbering and sections as [`MODEL_HISTORY.md`](MODEL_HISTORY.md), which has the full reasoning and results.

Compiled on 6 Nov 2026 from the team's development sessions with AI assistants (Claude Code and Codex), the approved plans, and the Group B reading guide (`ABC_local_test/ABC_local_test/research/`). None of these are public. The paper's own reference list is the final one.

## How to read this

**Role**, meaning what the source did for the experiment:
- **Adopted**: the idea is in the model or the recipe.
- **Informed**: it shaped a decision (for example "keep the prior", "use the full 4 s"), without adding a new component.
- **Prior art**: named after the design was already built, as related work. Cite it as related work, not as the inspiration.
- **Considered**: weighed and not adopted. The reason is given.
- **Background**: read at the start, with no direct link to a run.

**Check**, meaning how far the bibliographic details have been verified:
- **✓**: title, authors and venue seen in a primary record: a page fetched in the sessions, the checked bibliography of 14 Oct (`research/references.bib`), or a search on 6 Nov.
- **~**: the title or arXiv ID was seen in the sessions, but some details (authors, venue, pages) were written from memory.
- **?**: named in the sessions without a lookup, or only the abstract was read.
- **✓ ?**: the metadata is checked, but only the abstract was read.

These checks are as of 6 Nov. Every entry needed reading before it was cited.

## 1. Sources by experiment

### 0. Before any training: the Group B reading guide (14 Oct)
- **What it is:** `ABC_local_test/research/Group_B_reading_guide.md` and `references.bib` (16 papers, metadata checked 14 Oct). This review came before the first TCNN.
- **Its recommendation:** start with a small rotation-aware predictor; compare a GRU, a causal TCN and a temporal MLP; test "a learned correction to angular-velocity extrapolation".
- **Sources:** [2] Illahi, [5] Zhong 2026, [3] Yoon, [12] Pavllo, [15] Zhou and siMLPe (§3) as the six to read first; [7] TRACK, [6] Zhong 2025 and DLow on evaluation; [4] HAP, SkeletonDiffusion, SMamDiff and Robo-Saber as newer work; [18] Solà for geometry; [37] Hinton and [41] Jacob for compression.
- **What carried forward:** the learned correction to a motion forecast ([3], [4]) became the damped-velocity prior in v3. The relative-rotation advice ([5], [15]) carried into the input design.

### 1. First TCNN, 23 Oct (MODEL_HISTORY §1.1)
- **Sources:**
  - [12] Pavllo et al. is cited in `TRAINING_DETAILS.md` "for quaternion motion forecasting context". **Background.**
  - While reviewing the results, Codex fetched [11] Martinez et al. and [13] QuaterNet. It cited [11] when suggesting the network correct a damped-velocity forecast. **Informed.**
- **Outcome:** test 5.593° against hold-last 6.741° (7 users). This led to the five recommendations.

### 2. `tcnn_next` and `tcnn_expanded`, 23–24 Oct (§1.2–1.3)
- No new sources. These runs tested the five recommendations and the larger dataset [1].

### 3. v3: damped prior and EMA, 24 Oct (§1.4)
- **Sources:**
  - The prior traces to [11] (Codex, 23 Oct) and to the reading guide's hybrid suggestion from [3]. **Informed.**
  - The v3 session itself cites no paper. EMA was added with no source.
  - The v6 plan (25 Oct) later names [3] Yoon and [4] HAP as prior art for "a residual on a damped-velocity prior". **Prior art.**
- **Outcome:** test 5.10° against damped velocity's 6.13° (13/13 users). The prior alone was worth only −0.003° in v4.

### 4. v4: more training users, 24 Oct (§1.5)
- No sources besides the data [1].

### 5. Local v5: streaming body-frame inputs, 24–25 Oct (§1.6)
- **Why it was built:** the latency weight (30%) and the challenge's permission for a stateful graph (Submitting Artifacts page). No paper was read at the time.
- **Sources found afterwards** (v6 plan, 25 Oct): [26] Fast WaveNet and [27] Coucke et al. for per-layer streaming caches. **Prior art.**
- **Related, not cited in the decision:** [5] Zhong 2026 (body-frame targets reduce calibration sensitivity).
- **Outcome:** 60× fewer MACs and 3× faster, at +0.160°.

### 6. Local v6: gravity and mirror equivariance, 25 Oct (§1.7)
This was the main literature step for the architecture: 18 web searches and 5 page reads on the night of 24–25 Oct (the v6 plan).
- **Adopted:**
  - [19] Chirality Nets: odd/even weight sharing for left-right reflection. This is the tied [[A, B], [B, A]] mirror block.
  - [20] EqNIO: "subequivariance" to gravity-preserving roto-reflections. This frames gravity and the mirror together: build in the symmetry the motion has, and expose the one it lacks.
- **Informed the output maths** (Feature 2: rate from δ_w·δ_v, Gibbs/Cayley residual, safe operators only):
  - [15] Zhou et al. and [16] Brégier on rotation representations for regression.
  - [17] Liu et al. was listed as "Gibbs vector and Cayley transform", but it is a robust rotation-estimation paper. See §5.
  - The Submitting Artifacts operator list.
- **Considered, not adopted:**
  - Quaternion or PHM layers ([21] QTCN thesis, [22], [23] Parcollet, [24] PHM, [25] PHNN): up to 4× fewer parameters, but "compression only, with no physical reason behind it", and prior art exists.
  - A minimum-jerk output head ([10]): low expected gain at this size.
- **Listed as related work:** [6] Zhong 2025 (ESKF for edge XR), [12] Pavllo, [26] and [27].
- **Outcome:** −0.124° against v5 (13/13 users), with half the parameters.

### 7. Shared bench and group-a-v2 data, 25–26 Oct (§2)
- **Sources:** [1] BOXRR-23 and the XROR reference implementation (§4). No papers.

### 8. v7 compression, 26–27 Oct (§3)
- **Sources:** no papers were consulted. The suite came from our own measurements: M1 (latency follows node count), M2 and E1. The ST Edge AI and Qualcomm AI Hub docs supported the hardware plan (§4).
- [21] and [24] were listed again as rejected prior art.
- The int8 check (E3) has [41] as background from the reading guide, but the session did not cite it.
- **For the paper:** separable blocks have no cited source. MobileNets is the obvious candidate (§3).

### 9. v8 recipe and angular loss, 27 Oct (§4.1–4.2)
- **Sources:** none. The changes came from our own diagnostics: squared-angle loss against a mean-angle score, and A1 memorisation.
- **For the paper:** the angular (geodesic) loss needs a citation. Huynh 2009 and Mahendran et al. 2017 are the candidates (§3).

### 10. Frame robustness and the no-gravity hedge, 27–28 Oct (§4.3)
- **Sources:** no papers. The deciding inputs were the organisers' 28 Oct answers (§4).
- **Related work:** [20] (the gravity-symmetry framing) and [5] (calibration robustness).

### 11. Lean graph (Track A), 28 Oct (§5.2)
- **Sources:**
  - The challenge page's hint: "if your head computes an exponential map or similar, consider where that work belongs". **Informed.**
  - The Cayley output continues v6's Feature 2 ([15], [16]).
  - [26] for the state graph.
- **Outcome:** 130 → 84 nodes, 87.6 → 70.5 µs, with no retraining.

### 12. v9 → k8_combined, 28–29 Oct (§5.1, 5.3)
This was the literature review Jacob asked for, run on 28 Oct (the v9 plan, 20 searches). Each source below is matched to the screen item it produced.

| Screen item | Source | Role | Result |
| --- | --- | --- | --- |
| Receptive field to the full 4 s (B1, B2, B4) | [14] Mao et al.: motion repeats, so matching against long history helps. Nic's ablations agreed. | Adopted | k8 (B4), receptive field 393: −0.048° at seed 42 |
| Fewer, larger-kernel blocks (B3, B4) | [28] ModernTCN: large depthwise kernels are cheap | Adopted | k8 has 3 blocks with kernel 8, and fewer nodes |
| State cost of a longer receptive field (L0 timing check) | [26] Fast WaveNet: the cached state grows with the receptive field | Informed | Each variant was timed before the screen |
| Window sampling (B5, B6) | [30] Idrissi et al.: subsampling and reweighting are strong balancing baselines | Adopted | −0.029°; fixes the A1 memorisation |
| Tilt augmentation only (B8) | [31] Wen et al. survey, [32] FrAug: forecasting gains from time-domain augmentation are mixed. Plus [20] EqNIO. | Adopted | 15° tilt, kept for robustness |
| Keep motion-only inputs and the damped prior | [7] TRACK, [2] Illahi, [11] Martinez, [9] LaValle, [8] Gül | Informed | No change: confirms the existing design |
| Polynomial head | [29] N-BEATS (basis-expansion heads) | Prior art | The head already existed (v7 M2) |
| Non-neural reference | [6] Zhong 2025 | Background | Fetched, not used |
| Parked or rejected | [33] C-Mixup and [34] SAMformer (generic, unproven here); [35] RevIN (needs Div or Sqrt) with [36] the 2026 critique; motion attention from [14] (needs Softmax) | Considered | Not run |

- **Outcome:** k8_combined at −0.080° against sep + angular (15/15 users), 55 µs, promoted to `best/`.

### 13. Distillation (Track C), 28–29 Oct (§5.3)
- **Sources:**
  - [37] Hinton et al., [38] Furlanello et al. (born-again networks), [39] Saputra et al. (distilling a pose regressor) and [40] Pareek et al. (repeated self-distillation).
  - What the plan took from them: in regression the gain comes from denoised targets weighted by how reliable the teacher is, so the track was gated on an ensemble-headroom check.
- **Role:** Informed. The challenge page also lists distillation as encouraged.
- **Outcome:** the 9-checkpoint teacher was only 0.02° better than k8, and β 0.5 cost +0.021°. **Not adopted.**

### 14. Jitter and timing robustness, 29 Oct (§5.4)
- **Sources:** no papers. The organisers' "some jitter, but mostly 100 Hz" answer started it.
- The v9 plan's caution from [31] and [32] about time-domain augmentation is the reason the jitter variant had to pass an exact-grid rule.
- **Outcome:** 1.5 ms jitter costs +0.009°. Not adopted in the submitted model.

### 15. Grokking question → v10_users, 31 Oct (§7.1)
- **Sources:**
  - [42] Power et al. (grokking) and [43] Grokfast were named in the answer, which said grokking needs a memorisation phase that k8 doesn't have.
  - **No search was run.** The description came from the model's own knowledge, so both are marked "?". Check them before citing.
- **External-data candidates:** [44] Wu et al. 2017 and [45] David et al. 2018 (MMSys 360° head-movement datasets) were named, with licences and rates unverified. None were used.
- The FAQ's rule on external data was fetched, and it allows external data if disclosed.
- **Outcome:** data-limited. 50 → 100 users gained −0.029°.

### 16. 800-user dataset and k8 retrain, 31 Oct–2 Nov (§7.2)
- **Sources:** [1] only.

### 17. Nic's models as comparison points (§6)
- **Sources:** GADS [48], the inspiration for Nic's GADS-T. It is not peer reviewed, and GADS-T is Nic's own design (`nic/FINDINGS.md` G01).
- No sources for the 2 Nov compression read-out.

### 18. Weight-edit pilot, 4–5 Nov (§8.1)
- **Sources:** [46] "Sparks of Weight-Space Programming", the blog post Jacob linked. **Adopted as the method:** edit trained weights directly and evaluate, with no retraining.
- The results README frames it as "gradient-free post-training calibration inspired by" the post, not evidence for the post's broader claims.
- **Outcome:** −0.0023°, CI crosses zero. It failed its gate.

### 19. Endpoint refit, 5–6 Nov (§8.2)
- **Sources:** the plan and the README cite none.
  - Before proposing the refit, Codex searched for last-layer retraining on frozen features and for model soups (5 Nov 18:00). The results included [47] Wortsman et al. and the related items in §2.12.
  - The plan does not say whether any of these shaped it.
- **For the paper:** refitting a frozen network's final layer by regularised least squares has prior art in last-layer retraining. Kirichenko et al. (ICLR 2023, "Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations") is a candidate. That suggestion was added by the AI assistant while compiling this file, not taken from the sessions, so check it.
- **Outcome:** −0.0059° on 40 reused users. On test20 the interval includes zero.

### 20. Generalisation report, 5 Nov (§8.3)
- No sources.

## 2. Reference list

"Used in" refers to the experiment numbers in §1.

### 2.1 Data
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 1 | V. Nair, W. Guo, R. Wang, J. F. O'Brien, L. Rosenberg and D. Song, "Berkeley Open Extended Reality Recordings 2023 (BOXRR-23): 4.7 Million Motion Capture Recordings from 105,852 Extended Reality Device Users," *IEEE TVCG*, 30(5), 2239–2246, 2024. Dataset DOI 10.25350/B5NP4V. | [arXiv:2310.00430](https://arxiv.org/abs/2310.00430) | ✓ | Every experiment |

### 2.2 Head-motion prediction
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 2 | G. K. Illahi, A. Vaishnav, T. Kämäräinen, M. Siekkinen and M. Di Francesco, "Learning to Predict Head Pose in Remotely-Rendered Virtual Reality," *Proc. 14th ACM MMSys*, 27–38, 2023. | [doi:10.1145/3587819.3590972](https://doi.org/10.1145/3587819.3590972) | ✓ | 0, 12 |
| 3 | S. Yoon, H. J. Lim, J. H. Kim, Y.-T. Kim, H.-S. Lee and S. Sull, "Deep 6-DOF Head Motion Prediction for Latency in Lightweight Augmented Reality Glasses," *IEEE ICCE*, 2022. Abstract only read. | [doi:10.1109/ICCE53296.2022.9730327](https://doi.org/10.1109/ICCE53296.2022.9730327) | ✓ ? | 0, 3 |
| 4 | Y. Feng, J. Ma, G. Sun, C. Xu and H. Wang, "HAP: A Hand-Driven Active Perception Framework for Egocentric Head Motion Prediction," arXiv preprint, Sep 2026. | [arXiv:2609.18548](https://arxiv.org/abs/2609.18548) | ✓ | 0, 3 |
| 5 | Z. Zhong, B. Landfeldt, G. Alce and H. Caltenco, "Towards Calibration Robust: 6DoF Multi-Horizon Head Motion Prediction for EdgeXR," *IEEE AIxVR*, 2026. Abstract only read. | [doi:10.1109/AIxVR67263.2026.00010](https://doi.org/10.1109/AIxVR67263.2026.00010) | ✓ ? | 0, 5, 10 |
| 6 | Z. Zhong, H. A. Caltenco, B. Landfeldt and G. Alce, "Predictability-Aware Motion Prediction for Edge XR via High-Order Error-State Kalman Filtering," arXiv preprint, 2025. | [arXiv:2507.13179](https://arxiv.org/abs/2507.13179) | ✓ | 0, 6, 12 |
| 7 | M. F. Romero Rondón, L. Sassatelli, R. Aparicio Pardo and F. Precioso, "TRACK: A New Method from a Re-examination of Deep Architectures for Head Motion Prediction in 360 Degree Videos," *IEEE TPAMI*, 44(9), 5681–5699, 2022. | [arXiv:1911.11702](https://arxiv.org/abs/1911.11702) | ✓ | 0, 12 |
| 8 | S. Gül, S. Bosse, D. Podborski, T. Schierl and C. Hellge, "Kalman Filter-based Head Motion Prediction for Cloud-based Mixed Reality," *ACM Multimedia*, 2020. | [arXiv:2007.14084](https://arxiv.org/abs/2007.14084) | ~ | 12 |
| 9 | S. M. LaValle, "The Latent Power of Prediction," Oculus developer blog (the link the v9 plan used). The citable paper is S. M. LaValle, A. Yershova, M. Katsev and M. Antonov, "Head Tracking for the Oculus Rift," *IEEE ICRA*, 2014; only a synopsis was read. | [blog](https://developers.meta.com/horizon/blog/the-latent-power-of-prediction/) | ~ | 12 |
| 10 | N. Boulanger, F. Buisseret, J. Burny, F. Dierick, W. Estievenart, A. Teregulov and O. White, "A higher-derivative model predicts a smoothness vs duration relationship in head-pointing movements," arXiv preprint, Aug 2026. Abstract only read. | [arXiv:2608.17904](https://arxiv.org/abs/2608.17904) | ✓ ? | 6 (considered) |

### 2.3 Human motion prediction and quaternion models
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 11 | J. Martinez, M. J. Black and J. Romero, "On Human Motion Prediction Using Recurrent Neural Networks," *CVPR*, 4674–4683, 2017. | [arXiv:1705.02445](https://arxiv.org/abs/1705.02445) | ✓ | 1, 3, 12 |
| 12 | D. Pavllo, C. Feichtenhofer, M. Auli and D. Grangier, "Modeling Human Motion with Quaternion-based Neural Networks," *IJCV*, published online 2019. doi:10.1007/s11263-019-01245-6. | [arXiv:1901.07677](https://arxiv.org/abs/1901.07677) | ✓ | 0, 1, 6 |
| 13 | D. Pavllo, D. Grangier and M. Auli, "QuaterNet: A Quaternion-based Recurrent Model for Human Motion," *BMVC*, 2018. | [arXiv:1805.06485](https://arxiv.org/abs/1805.06485) | ~ | 1 |
| 14 | W. Mao, M. Liu and M. Salzmann, "History Repeats Itself: Human Motion Prediction via Motion Attention," *ECCV*, 2020. | [arXiv:2007.11755](https://arxiv.org/abs/2007.11755) | ~ | 12 |

### 2.4 Rotation representations
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 15 | Y. Zhou, C. Barnes, J. Lu, J. Yang and H. Li, "On the Continuity of Rotation Representations in Neural Networks," *CVPR*, 5745–5753, 2019. | [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html) | ✓ | 0, 6, 11 |
| 16 | R. Brégier, "Deep Regression on Manifolds: A 3D Rotation Case Study," *3DV*, 2021. | [arXiv:2103.16317](https://arxiv.org/abs/2103.16317) | ~ | 6, 11 |
| 17 | Y. Liu, T. Huang and Z.-X. Yang, "Linearly Solving Robust Rotation Estimation," arXiv preprint, 2025. | [arXiv:2506.11547](https://arxiv.org/abs/2506.11547) | ✓ | 6 (weak; see §5) |
| 18 | J. Solà, "Quaternion Kinematics for the Error-State Kalman Filter," arXiv, 2017. | [arXiv:1711.02508](https://arxiv.org/abs/1711.02508) | ✓ | 0 |

### 2.5 Symmetry and equivariance
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 19 | R. Yeh, Y.-T. Hu and A. Schwing, "Chirality Nets for Human Pose Regression," *NeurIPS* 32, 8163–8173, 2019. | [NeurIPS](https://papers.nips.cc/paper_files/paper/2019/file/1f88c7c5d7d94ae08bd752aa3d82108b-Paper.pdf) | ✓ | 6 (adopted) |
| 20 | R. K. Jayanth, Y. Xu, Z. Wang, E. Chatzipantazis, D. Gehrig and K. Daniilidis, "EqNIO: Subequivariant Neural Inertial Odometry," *ICLR*, 2025. | [arXiv:2408.06321](https://arxiv.org/abs/2408.06321) | ~ | 6 (adopted), 10, 12 |

### 2.6 Quaternion and hypercomplex layers (considered, not adopted)
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 21 | C. E. Long, "Quaternion Temporal Convolutional Neural Networks," M.S. thesis, University of Dayton, 2019. | [eCommons](https://ecommons.udayton.edu/graduate_theses/6857) | ✓ | 6, 8 |
| 22 | T. Parcollet, M. Ravanelli, M. Morchid, G. Linarès, C. Trabelsi, R. De Mori and Y. Bengio, "Quaternion Recurrent Neural Networks," *ICLR*, 2019. | [arXiv:1806.04418](https://arxiv.org/abs/1806.04418) | ~ | 6 |
| 23 | T. Parcollet, Y. Zhang, M. Morchid, C. Trabelsi, G. Linarès, R. De Mori and Y. Bengio, "Quaternion Convolutional Neural Networks for End-to-End Automatic Speech Recognition," *Interspeech*, 2018. | [arXiv:1806.07789](https://arxiv.org/abs/1806.07789) | ~ | 6 |
| 24 | A. Zhang, Y. Tay, S. Zhang, A. Chan, A. T. Luu, S. C. Hui and J. Fu, "Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters," *ICLR*, 2021. | [arXiv:2102.08597](https://arxiv.org/abs/2102.08597) | ~ | 6, 8 |
| 25 | E. Grassucci, A. Zhang and D. Comminiello, "PHNNs: Lightweight Neural Networks via Parameterized Hypercomplex Convolutions." Venue to confirm. | [arXiv:2110.04176](https://arxiv.org/abs/2110.04176) | ~ | 6 |

### 2.7 Streaming convolutions and architecture
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 26 | T. L. Paine, P. Khorrami, S. Chang, Y. Zhang, P. Ramachandran, M. A. Hasegawa-Johnson and T. S. Huang, "Fast Wavenet Generation Algorithm," arXiv, 2016. | [arXiv:1611.09482](https://arxiv.org/abs/1611.09482) | ~ | 5 (prior art), 6, 11, 12 |
| 27 | A. Coucke, M. Chlieh, T. Gisselbrecht, D. Leroy, M. Poumeyrol and T. Lavril, "Efficient Keyword Spotting Using Dilated Convolutions and Gating," *ICASSP*, 2019. | [arXiv:1811.07684](https://arxiv.org/abs/1811.07684) | ~ | 5 (prior art), 6 |
| 28 | D. Luo and X. Wang, "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis," *ICLR*, 2024. | [OpenReview](https://openreview.net/forum?id=vpJMJerXHU) | ✓ | 12 (adopted) |
| 29 | B. N. Oreshkin, D. Carpov, N. Chapados and Y. Bengio, "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting," *ICLR*, 2020. | — | ? | 12 (prior art) |

### 2.8 Training recipe: sampling, augmentation, regularisation
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 30 | B. Y. Idrissi, M. Arjovsky, M. Pezeshki and D. Lopez-Paz, "Simple Data Balancing Achieves Competitive Worst-Group-Accuracy," *CLeaR*, 2022. | [arXiv:2110.14503](https://arxiv.org/abs/2110.14503) | ~ | 12 (adopted) |
| 31 | Q. Wen, L. Sun, F. Yang, X. Song, J. Gao, X. Wang and H. Xu, "Time Series Data Augmentation for Deep Learning: A Survey," *IJCAI*, 2021. | [arXiv:2002.12478](https://arxiv.org/abs/2002.12478) | ~ | 12, 14 |
| 32 | M. Chen, Z. Xu, A. Zeng and Q. Xu, "FrAug: Frequency Domain Augmentation for Time Series Forecasting," arXiv, 2023. | [arXiv:2302.09292](https://arxiv.org/abs/2302.09292) | ~ | 12, 14 |
| 33 | H. Yao, Y. Wang, L. Zhang, J. Zou and C. Finn, "C-Mixup: Improving Generalization in Regression," *NeurIPS*, 2022. | [arXiv:2210.05775](https://arxiv.org/abs/2210.05775) | ~ | 12 (considered) |
| 34 | R. Ilbert et al., "SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention," *ICML*, 2024. | [arXiv:2402.10198](https://arxiv.org/abs/2402.10198) | ~ | 12 (considered) |
| 35 | T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi and J. Choo, "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift," *ICLR*, 2022. | — | ~ | 12 (considered) |
| 36 | "On the Role of Reversible Instance Normalization," arXiv preprint, Mar 2026. The author list differs between sources; take it from the arXiv page. | [arXiv:2603.11869](https://arxiv.org/abs/2603.11869) | ~ | 12 (considered) |

### 2.9 Distillation
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 37 | G. Hinton, O. Vinyals and J. Dean, "Distilling the Knowledge in a Neural Network," arXiv, 2015. | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) | ✓ | 0, 13 |
| 38 | T. Furlanello, Z. C. Lipton, M. Tschannen, L. Itti and A. Anandkumar, "Born-Again Neural Networks," *ICML*, 2018. | — | ? | 13 |
| 39 | M. R. U. Saputra, P. P. B. de Gusmão, Y. Almalioglu, A. Markham and N. Trigoni, "Distilling Knowledge From a Deep Pose Regressor Network," *ICCV*, 2019. | [arXiv:1908.00858](https://arxiv.org/abs/1908.00858) | ~ | 13 |
| 40 | D. Pareek, S. S. Du and S. Oh, "Understanding the Gains from Repeated Self-Distillation," 2024. Venue to confirm. | [arXiv:2407.04600](https://arxiv.org/abs/2407.04600) | ~ | 13 |

### 2.10 Quantisation
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 41 | B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam and D. Kalenichenko, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," *CVPR*, 2018. | [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) | ✓ | 0, 8 (background) |

### 2.11 Data scaling (v10)
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 42 | A. Power, Y. Burda, H. Edwards, I. Babuschkin and V. Misra, "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets," arXiv, 2022. | — | ? | 15 |
| 43 | J. Lee, B. G. Kang, K. Kim and K. M. Lee, "Grokfast: Accelerated Grokking by Amplifying Slow Gradients," arXiv, 2024. | — | ? | 15 |
| 44 | C. Wu, Z. Tan, Z. Wang and S. Yang, "A Dataset for Exploring User Behaviors in VR Spherical Video Streaming," *ACM MMSys*, 2017. | — | ? | 15 (not used) |
| 45 | E. J. David, J. Gutiérrez, A. Coutrot, M. P. Da Silva and P. Le Callet, "A Dataset of Head and Eye Movements for 360° Videos," *ACM MMSys*, 2018. | — | ? | 15 (not used) |

### 2.12 Weight editing (v10)
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 46 | R. Takehi, J. Teoh, A. Zhang and O. Khattab, "Sparks of Weight-Space Programming," blog post, Nov 2026. The author list comes from secondary coverage; confirm it on the post. | [rikiyat.github.io](https://rikiyat.github.io/blog/weight-space-programming/) | ~ | 18 (adopted) |
| 47 | M. Wortsman et al., "Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy without Increasing Inference Time," *ICML*, 2022. | [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) | ✓ | 19 (searched) |

Other results of the same 5 Nov search, not cited anywhere: Rewarded soups ([arXiv:2306.04488](https://arxiv.org/abs/2306.04488)), stochastic weight averaging (Izmailov et al. 2018, named in a snippet), Dodge et al. ([arXiv:2002.06305](https://arxiv.org/abs/2002.06305)) and Barycentric Anchored Weight Search ([arXiv:2610.14017](https://arxiv.org/abs/2610.14017)).

### 2.13 Nic's comparison models
| # | Reference | Link | Check | Used in |
| --- | --- | --- | :-: | --- |
| 48 | GADS: face-landmark head-pose estimation from one frame. Not peer reviewed, no code. The title isn't in our notes; Nic's `research/modelling/gads_verified.md` should have it. | [arXiv:2504.15751](https://arxiv.org/abs/2504.15751) | ~ | 17 |

## 3. Candidates for related work that weren't used during the experiments

**From the paper plan (5 Nov).** None of these came up while the experiments were being decided. Each would back a design choice that was made on other grounds.

| Candidate | Would support |
| --- | --- |
| R. Azuma and G. Bishop, SIGGRAPH 1995; J. J. LaViola Jr., EGVE 2003; LaValle et al., ICRA 2014 | The hold-last, constant-velocity and damped-velocity baselines, and the damped prior (3) |
| D. Q. Huynh, *J. Math. Imaging Vis.*, 2009; S. Mahendran et al., ICCV Workshops 2017 | The geodesic metric and the angular loss (9) |
| S. Bai, J. Z. Kolter and V. Koltun, arXiv:1803.01271, 2018; A. van den Oord et al., WaveNet, arXiv:1609.03499, 2016 | Dilated causal TCNs (every version) |
| A. G. Howard et al., MobileNets, arXiv:1704.04861, 2017 | Separable blocks (8, 12) |
| T. Cohen and M. Welling, ICML 2016 | Weight tying for symmetry, alongside [19] and [20] (6) |
| A. Zeng et al., AAAI 2023 | The case that simple predictors are strong at short horizons (12) |

**Reading-guide background (14 Oct), never linked to a run:**
- W. Guo et al., "Back to MLP," WACV 2023 (siMLPe)
- Y. Yuan and K. Kitani, "DLow," ECCV 2020
- C. Curreli et al., SkeletonDiffusion, CVPR 2025
- J. Fan et al., SMamDiff, arXiv:2512.00355, 2025
- N. H. Kim et al., "Robo-Saber," *Computer Graphics Forum* 45(2), 2026, which is trained on BOXRR-23

Full entries are in `ABC_local_test/research/references.bib`.

## 4. Non-paper sources

| Source | Used for |
| --- | --- |
| 9th ABC Challenge pages: [The Challenge](https://sites.google.com/view/9th-abc-challenge/home/the-challenge), [The Data](https://sites.google.com/view/9th-abc-challenge/home/the-data), [FAQs](https://sites.google.com/view/9th-abc-challenge/home/faqs), [Submitting Artifacts](https://sites.google.com/view/9th-abc-challenge/home/submitting-artifacts), read many times; the 14 Oct snapshots are in `ABC_local_test/research/sources/` | The task, horizon and scoring; the safe-operator list (v6 Feature 2, Track A); the "mid-field" innovation remark (v6); the exponential-map hint (Track A); distillation listed as encouraged (13); external data allowed if disclosed (15) |
| The organisers' answers of 28 Oct (by email) | The 200 ms horizon, test data in our documented frame (gravity kept, 10), "some jitter" (14) |
| [BOXRR-23 dataset page](https://rdi.berkeley.edu/metaverse/boxrr-23/) and its data-use agreement; the [Hugging Face mirror](https://huggingface.co/datasets/cschell/boxrr-23) | Data access and licence terms (7, 16) |
| [XROR reference implementation](https://github.com/MetaGuard/xror) | Decoding the recordings (group-a-v2 pipeline, 7) |
| [ST Edge AI Developer Cloud](https://stm32ai.st.com/st-edge-ai-dc/) and [Qualcomm AI Hub](https://workbench.aihub.qualcomm.com/) docs | The hardware latency plan (8) |

## 5. Before citing

The Check column records what had been verified by 6 Nov. Before submission, the paper's reference list was checked against the publishers' records, so prefer the paper's entries over the ones here. Some sources are prior art rather than inspiration: the streaming state graph and the polynomial head were built before [26], [27] and [29] were found, and the damped prior was built after [11]'s suggestion and before [3] and [4] were named.

## How this was compiled

- **Scan:** a read-only script searched the team's AI-assistant sessions for this project (29 Claude Code, 7 Codex). It looked for web searches, page fetches, academic links, arXiv IDs and "et al." citations. Its output isn't public.
- **Rule for "used":** a source counts as used only where a plan or reply states it as the basis for a run or a decision. Search results that were never cited are left out, apart from the two lists of "searched, not used" items above.
- **Verification on 6 Nov:** four web searches checked [17], [21], [36] and [46].
- **Source lists:** the v6 and v9 plans (their Sources and Literature review sections) and the reading guide (`ABC_local_test/research/`). None of these are public.

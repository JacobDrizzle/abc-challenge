window.EXPERT = {
 "overview": {
  "title": "Streaming mirror-equivariant temporal predictor",
  "prose": "k8_combined is a streaming temporal convolutional network with 7,555 learned parameters, trained on 5,243 BOXRR-23 players. A shared 7 → 32 projection feeds two mirror streams through three depthwise-separable dilated causal convolution blocks (kernel 8, dilations 1, 8 and 47) with skip connections, giving a 393-step receptive field. A shared readout predicts cubic polynomial residuals that are added to a damped-velocity prior (the native body rate ω, exponentially damped), and a Cayley map turns each of the 20 local rotation vectors into a unit quaternion, all horizons in parallel. The exported ONNX graph runs one new sample per call, with a streaming state cache (the block's memory of its past inputs) per block instead of re-reading the whole 4 s window.",
  "summary": "A shared projection and three separable causal blocks with skip connections predict polynomial residuals to a damped-velocity prior.",
  "math": [
   "q[t−1], q[t] → f_t → shared projection → blocks(d=1,8,47) → coefficients → r[1:20] → q̂[1:20]",
   "K=32 per mirror stream; 2 streams; kernel=8 in every temporal block",
   "Nθ = (32×7+32) + 3×(32×8+32×32×2+32) + 9×32+3 = 7555",
   "ONNX initializer elements = 7552; the 3 learned x biases reside in /Constant_53"
  ],
  "shapes": [
   "context: [1,400,4]",
   "hidden: [1,32,2,1]",
   "prediction: [1,20,4]"
  ],
  "onnx": [
   "/Conv (Conv)",
   "/Conv_1 (Conv)",
   "/Conv_3 (Conv)",
   "/Conv_5 (Conv)",
   "/MatMul_6 (MatMul)",
   "/MatMul_8 (MatMul)",
   "/Slice_10 (Slice)"
  ],
  "params": "7555 learned; 7552 initializer elements",
  "refs": [
   "jacob/training/best/model.py:28",
   "jacob/training/best/model.py:140",
   "jacob/training/best/model.py:252",
   "jacob/training/best/model.py:255",
   "jacob/training/best/export.py:89",
   "jacob/training/best/export.py:103",
   "jacob/training/best/export.py:157"
  ],
  "caveat": "Shape convention for hidden tensors: [batch,channel,mirror row,time]; the graph runs one new sample per call."
 },
 "input": {
  "title": "Latest pair and persistent history",
  "prose": "The streaming graph reads only the two newest unit, scalar-first quaternions, q[t−1] and q[t]. Everything older lives in each block's streaming state cache, which holds that block's past input activations, not raw poses. The graph assumes unit-norm inputs; normalisation happens in PyTorch (model.features), not in the exported graph.",
  "summary": "The streaming graph reads only the newest pair of unit scalar-first quaternions while each block retains its own past input activations.",
  "math": [
   "previous = context[:,−2:−1]; now = context[:,−1:]",
   "q=(w,x,y,z); ‖q‖₂=1 is an input precondition of this graph",
   "state_in_i[t] = x_i[t−7d_i:t]; state_out_i = cat(state_in_i,x_i[t])[...,1:]"
  ],
  "shapes": [
   "context: [1,400,4]",
   "previous: [1,1,4]",
   "now: [1,1,4]",
   "state_in_0: [1,32,2,7]",
   "state_in_1: [1,32,2,56]",
   "state_in_2: [1,32,2,329]"
  ],
  "onnx": [
   "/Slice (Slice)",
   "/Slice_1 (Slice)",
   "/Concat_1 (Concat)",
   "/Slice_4 (Slice)",
   "/Concat_3 (Concat)",
   "/Slice_6 (Slice)",
   "/Concat_5 (Concat)",
   "/Slice_8 (Slice)"
  ],
  "params": "0 learned",
  "refs": [
   "jacob/training/best/export.py:57",
   "jacob/training/best/export.py:72",
   "jacob/training/best/export.py:124",
   "jacob/training/best/export.py:140",
   "jacob/training/best/export.py:141"
  ],
  "caveat": "The exporter assumes unit inputs; model.features normalises them in PyTorch; caches contain block inputs, not raw poses."
 },
 "feat": {
  "title": "Native body rate, L1 speed and gravity",
  "prose": "Seven features are computed per step with no learned weights. The relative rotation d = conj(q[t−1]) ⊗ q[t] gives a sign-invariant native body-motion rate ω = 200 w_d v_d (which uses sin θ, not the exact logarithmic map), its L1 magnitude, and world up expressed in the current head frame, g = R(q[t])ᵀ(0, 1, 0). No global frame canonicalisation is applied.",
  "summary": "Seven features combine a sign-invariant native body-motion rate with its L1 magnitude and world up expressed in the current head frame.",
  "math": [
   "d=conj(q[t−1])⊗q[t]=(w_d,v_d)",
   "ω=200 w_d v_d = 100 sin(θ)n; speed=|ω_x|+|ω_y|+|ω_z|",
   "g=R(q[t])ᵀ(0,1,0)=(2(xy+wz), w²−x²+y²−z², 2(yz−wx))",
   "f_t=(ω_x,ω_y,ω_z,speed,g_x,g_y,g_z)",
   "ONNX: δ=√200 d; ω=δ_w δ_xyz; gravity=flatten(q qᵀ) G"
  ],
  "shapes": [
   "native rate: [1,1,3]",
   "speed: [1,1,1]",
   "gravity: [1,1,3]",
   "f: [1,1,7]"
  ],
  "onnx": [
   "/MatMul (MatMul)",
   "/MatMul_1 (MatMul)",
   "/Slice_2 (Slice)",
   "/Slice_3 (Slice)",
   "/Mul (Mul)",
   "/MatMul_2 (MatMul)",
   "/Relu (Relu)",
   "/MatMul_3 (MatMul)",
   "/MatMul_4 (MatMul)",
   "/MatMul_5 (MatMul)",
   "/Concat (Concat)"
  ],
  "params": "0 learned",
  "refs": [
   "jacob/training/best/model.py:59",
   "jacob/training/best/model.py:333",
   "jacob/training/best/model.py:335",
   "jacob/training/best/model.py:338",
   "jacob/training/best/model.py:339",
   "jacob/training/best/export.py:75",
   "jacob/training/best/export.py:125",
   "jacob/training/best/export.py:129",
   "jacob/training/best/export.py:132"
  ],
  "caveat": "Native ω uses sin(θ), not the exact logarithmic angular displacement θ; no global frame canonicalisation is applied."
 },
 "proj": {
  "title": "Shared feature projection",
  "prose": "A 1 × 1 convolution (an affine map, 32 × 7 weights plus 32 biases, then ReLU) projects each feature row into a 32-channel hidden stream. The same 256 parameters are applied to the observed row and to its mirrored copy (weight tying), against 512 for an untied 7 → 64 projection.",
  "summary": "The same affine map projects observed and mirrored feature rows into the two hidden streams before ReLU.",
  "math": [
   "f_0=f; f_1=D_f f; D_f=diag(1,−1,−1,1,−1,1,1)",
   "h_s=ReLU(W_p f_s+b_p), s∈{0,1}",
   "W_p: 32×7; b_p: 32; Conv kernel=(1,1), stride=(1,1)"
  ],
  "shapes": [
   "mirrored features: [1,7,2,1]",
   "projection weight: [32,7,1,1]",
   "projection bias: [32]",
   "h: [1,32,2,1]"
  ],
  "onnx": [
   "/Reshape_3 (Reshape)",
   "/Mul_1 (Mul)",
   "/Conv (Conv)",
   "/Relu_1 (Relu)"
  ],
  "params": "256 = 224 weights + 32 biases, shared across rows",
  "refs": [
   "jacob/training/best/model.py:50",
   "jacob/training/best/model.py:143",
   "jacob/training/best/export.py:86",
   "jacob/training/best/export.py:89",
   "jacob/training/best/export.py:134",
   "jacob/training/best/export.py:135"
  ]
 },
 "mirror": {
  "title": "Equivariance by tied mirror streams",
  "prose": "The model is mirror (reflection) equivariant by construction, through weight tying. A fixed sign flip D_f on the features builds the reflected input; because every later layer uses tied weights, of the form [[A, B], [B, A]], reflecting the input swaps the two hidden streams. The tying adds no parameters; the untied version of the same model has 15,113 against 7,555.",
  "summary": "Feature sign flips construct the reflected input and tied block weights make reflection swap the two hidden streams.",
  "math": [
   "M(w,x,y,z)=(w,x,−y,−z); f_1=D_f f_0, D_f=diag(1,−1,−1,1,−1,1,1)",
   "h_0=ReLU(W_p f_0+b_p); h_1=ReLU(W_p D_f f_0+b_p)",
   "PW = [[A,B],[B,A]]; u_0=A h_0+B h_1+b; u_1=B h_0+A h_1+b",
   "ONNX: store [A|B] once; Conv(kernel=(2,1)) over rows [h_0,h_1,h_0]",
   "M: (h_0,h_1) ↔ (h_1,h_0); output axis signs S=diag(1,−1,−1)"
  ],
  "shapes": [
   "feature sign mask: [1,7,2,1]",
   "hidden rows: [1,32,2,1]",
   "stored pointwise weight per block: [32,32,2,1]"
  ],
  "onnx": [
   "/Mul_1 (Mul)",
   "/Slice_5 (Slice)",
   "/Concat_2 (Concat)",
   "/Conv_2 (Conv)",
   "/Slice_7 (Slice)",
   "/Concat_4 (Concat)",
   "/Conv_4 (Conv)",
   "/Slice_9 (Slice)",
   "/Concat_6 (Concat)",
   "/Conv_6 (Conv)",
   "/Mul_2 (Mul)"
  ],
  "params": "0 additional; shared projection/block/head parameters counted in their stages",
  "refs": [
   "jacob/training/best/model.py:12",
   "jacob/training/best/model.py:50",
   "jacob/training/best/model.py:56",
   "jacob/training/best/model.py:207",
   "jacob/training/best/export.py:86",
   "jacob/training/best/export.py:92",
   "jacob/training/best/export.py:150"
  ],
  "caveat": "Sign flips apply to input features; h_1 is not obtained by negating selected learned channels of h_0."
 },
 "blk0": {
  "title": "Block 1: kernel 8, dilation 1",
  "prose": "Block 1 is a depthwise-separable dilated causal convolution: a depthwise temporal filter (kernel 8, dilation 1, one filter per channel), then a tied pointwise mirror mixing, then ReLU, added to the identity path (a skip connection). It costs 2,336 parameters against 16,416 for a dense tied convolution with the same kernel. With the layers before it, one output has a receptive field of 8 feature steps; its streaming state cache holds the block's inputs from t−7 to t−1.",
  "summary": "A shared depthwise temporal filter and tied mirror mixing add a rectified update to the identity stream, reaching 8 feature steps.",
  "math": [
   "DW: kernel=(1,8), dilation=(1,1), groups=32, stride=(1,1)",
   "m=cat(state_in,x); state_out=m[...,1:]",
   "h[c,s]=Σⱼ₌₀⁷ W_D[c,j] m[c,s,j×1]",
   "u_0=A h_0+B h_1+b; u_1=B h_0+A h_1+b",
   "x_out=ReLU(x+ReLU(u))"
  ],
  "shapes": [
   "state_in_0: [1,32,2,7]",
   "m: [1,32,2,8]",
   "depthwise weight: [32,1,1,8]",
   "pointwise weight: [32,32,2,1]",
   "pointwise bias: [32]",
   "x_out: [1,32,2,1]"
  ],
  "onnx": [
   "/Concat_1 (Concat)",
   "/Slice_4 (Slice)",
   "/Conv_1 (Conv)",
   "/Slice_5 (Slice)",
   "/Concat_2 (Concat)",
   "/Conv_2 (Conv)",
   "/Relu_2 (Relu)",
   "/Add (Add)",
   "/Relu_3 (Relu)"
  ],
  "params": "2336 = 256 depthwise + 2048 pointwise + 32 bias",
  "refs": [
   "jacob/training/best/model.py:283",
   "jacob/training/best/model.py:288",
   "jacob/training/best/model.py:290",
   "jacob/training/best/model.py:295",
   "jacob/training/best/model.py:297",
   "jacob/training/best/model.py:299",
   "jacob/training/best/export.py:140",
   "jacob/training/best/export.py:142",
   "jacob/training/best/export.py:150",
   "jacob/training/best/export.py:152"
  ],
  "caveat": "Cache holds block inputs t−7 through t−1; there is no activation between DW and PW, and dropout is disabled at inference. With the layers before it, one output depends on 8 feature steps (context strip)."
 },
 "blk1": {
  "title": "Block 2: kernel 8, dilation 8",
  "prose": "Block 2 repeats the same depthwise-separable dilated causal convolution with dilation 8: the same 8 taps are spaced 8 samples apart. Stacked on Block 1, the receptive field grows to 64 feature steps at the same 2,336 parameters. Its streaming state cache holds the block's inputs from t−56 to t−1.",
  "summary": "A shared depthwise temporal filter and tied mirror mixing add a rectified update to the identity stream, reaching 64 feature steps.",
  "math": [
   "DW: kernel=(1,8), dilation=(1,8), groups=32, stride=(1,1)",
   "m=cat(state_in,x); state_out=m[...,1:]",
   "h[c,s]=Σⱼ₌₀⁷ W_D[c,j] m[c,s,j×8]",
   "u_0=A h_0+B h_1+b; u_1=B h_0+A h_1+b",
   "x_out=ReLU(x+ReLU(u))"
  ],
  "shapes": [
   "state_in_1: [1,32,2,56]",
   "m: [1,32,2,57]",
   "depthwise weight: [32,1,1,8]",
   "pointwise weight: [32,32,2,1]",
   "pointwise bias: [32]",
   "x_out: [1,32,2,1]"
  ],
  "onnx": [
   "/Concat_3 (Concat)",
   "/Slice_6 (Slice)",
   "/Conv_3 (Conv)",
   "/Slice_7 (Slice)",
   "/Concat_4 (Concat)",
   "/Conv_4 (Conv)",
   "/Relu_4 (Relu)",
   "/Add_1 (Add)",
   "/Relu_5 (Relu)"
  ],
  "params": "2336 = 256 depthwise + 2048 pointwise + 32 bias",
  "refs": [
   "jacob/training/best/model.py:283",
   "jacob/training/best/model.py:288",
   "jacob/training/best/model.py:290",
   "jacob/training/best/model.py:295",
   "jacob/training/best/model.py:297",
   "jacob/training/best/model.py:299",
   "jacob/training/best/export.py:140",
   "jacob/training/best/export.py:142",
   "jacob/training/best/export.py:150",
   "jacob/training/best/export.py:152"
  ],
  "caveat": "Cache holds block inputs t−56 through t−1; there is no activation between DW and PW, and dropout is disabled at inference. With the layers before it, one output depends on 64 feature steps (context strip)."
 },
 "blk2": {
  "title": "Block 3: kernel 8, dilation 47",
  "prose": "Block 3 uses dilation 47, so its 8 taps span 329 samples. Stacked on Blocks 1 and 2, the receptive field reaches 393 feature steps (3.93 s), still with 2,336 parameters and a skip connection. Its streaming state cache holds the block's inputs from t−329 to t−1, the longest of the three.",
  "summary": "A shared depthwise temporal filter and tied mirror mixing add a rectified update to the identity stream, reaching 393 feature steps.",
  "math": [
   "DW: kernel=(1,8), dilation=(1,47), groups=32, stride=(1,1)",
   "m=cat(state_in,x); state_out=m[...,1:]",
   "h[c,s]=Σⱼ₌₀⁷ W_D[c,j] m[c,s,j×47]",
   "u_0=A h_0+B h_1+b; u_1=B h_0+A h_1+b",
   "x_out=ReLU(x+ReLU(u))"
  ],
  "shapes": [
   "state_in_2: [1,32,2,329]",
   "m: [1,32,2,330]",
   "depthwise weight: [32,1,1,8]",
   "pointwise weight: [32,32,2,1]",
   "pointwise bias: [32]",
   "x_out: [1,32,2,1]"
  ],
  "onnx": [
   "/Concat_5 (Concat)",
   "/Slice_8 (Slice)",
   "/Conv_5 (Conv)",
   "/Slice_9 (Slice)",
   "/Concat_6 (Concat)",
   "/Conv_6 (Conv)",
   "/Relu_6 (Relu)",
   "/Add_2 (Add)",
   "/Relu_7 (Relu)"
  ],
  "params": "2336 = 256 depthwise + 2048 pointwise + 32 bias",
  "refs": [
   "jacob/training/best/model.py:283",
   "jacob/training/best/model.py:288",
   "jacob/training/best/model.py:290",
   "jacob/training/best/model.py:295",
   "jacob/training/best/model.py:297",
   "jacob/training/best/model.py:299",
   "jacob/training/best/export.py:140",
   "jacob/training/best/export.py:142",
   "jacob/training/best/export.py:150",
   "jacob/training/best/export.py:152"
  ],
  "caveat": "Cache holds block inputs t−329 through t−1; there is no activation between DW and PW, and dropout is disabled at inference. With the layers before it, one output depends on 393 feature steps (context strip)."
 },
 "reach": {
  "title": "Finite causal receptive field",
  "prose": "The receptive field of the stack is 1 + Σ(k − 1)·d = 1 + 7 × (1 + 8 + 47) = 393 feature steps; dilation buys this reach at the same 8-tap cost per block. Because each feature step uses an adjacent pose pair, 393 steps span 394 raw poses. The price is memory, not parameters: a 392-column streaming state cache (25,088 floats), and a zero-state start needs warm-up.",
  "summary": "The three dilations extend the hidden feature history to 393 steps with the same 8-tap cost per block; the price is a 392-column cache (25,088 floats).",
  "math": [
   "RF_features=1+Σᵢ(k_i−1)d_i=1+7(1+8+47)=393",
   "RF after blocks 0,1,2: 8 → 64 → 393",
   "f[t−392] uses q[t−393] and q[t−392] ⇒ RF_raw=394 poses",
   "cache floats=32×2×(7+56+329)=25088"
  ],
  "shapes": [
   "feature support: [393,7]",
   "raw pose support: [394,4]",
   "state_in_0: [1,32,2,7]",
   "state_in_1: [1,32,2,56]",
   "state_in_2: [1,32,2,329]"
  ],
  "onnx": [
   "/Conv_1 (Conv)",
   "/Conv_3 (Conv)",
   "/Conv_5 (Conv)"
  ],
  "params": "0 additional; receptive field is a property of the three blocks",
  "refs": [
   "jacob/training/best/model.py:28",
   "jacob/training/best/model.py:82",
   "jacob/training/best/model.py:333",
   "jacob/training/best/export.py:72",
   "jacob/training/best/export.py:140",
   "jacob/training/best/export.py:142"
  ],
  "caveat": "393 feature steps span 394 raw poses because each rate consumes an adjacent pose pair; zero-state starts require warm-up."
 },
 "head": {
  "title": "Shared polynomial coefficients with x-only bias",
  "prose": "The readout is a shared linear head (9 × 32 weights) applied to both hidden streams; the mirrored stream's output is sign-flipped before the two are summed, which keeps the prediction mirror-equivariant. It produces three polynomial coefficients per rotation axis, plus one learned x-only bias per term: 291 parameters against 1,940 for a direct 60-output readout.",
  "summary": "The head reads both hidden streams through the same matrix, applies mirror output signs and adds one x-only bias for each polynomial term.",
  "math": [
   "W_h: [9,32]; S=diag(1,−1,−1) repeated for each polynomial term",
   "C=reshape(W_h h_0 + S W_h h_1, [1,3,3])",
   "C̃[j,:]=C[j,:]+(b_x[j],0,0), j=1,2,3",
   "Nθ=9×32+3=291"
  ],
  "shapes": [
   "hidden: [1,32,2,1]",
   "W_h: [9,32]",
   "both readouts: [9,2]",
   "C: [1,3,3]",
   "b_x: [3]",
   "C̃: [1,3,3]"
  ],
  "onnx": [
   "/Reshape_4 (Reshape)",
   "/MatMul_6 (MatMul)",
   "/Mul_2 (Mul)",
   "/MatMul_7 (MatMul)",
   "/Reshape_5 (Reshape)",
   "/Constant_53 (Constant)",
   "/Concat_7 (Concat)"
  ],
  "params": "291 = 288 shared weights + 3 learned x biases",
  "refs": [
   "jacob/training/best/model.py:164",
   "jacob/training/best/model.py:168",
   "jacob/training/best/model.py:252",
   "jacob/training/best/model.py:255",
   "jacob/training/best/model.py:259",
   "jacob/training/best/export.py:103",
   "jacob/training/best/export.py:115",
   "jacob/training/best/export.py:157"
  ],
  "caveat": "ONNX /Reshape_5 is C before bias; /Constant_53 supplies the three learned x biases to the trajectory MatMul."
 },
 "traj": {
  "title": "Polynomial residual plus damped-velocity prior",
  "prose": "The 20 future rotation vectors are a cubic polynomial in normalised horizon u = k/20 (fixed basis, learned coefficients) plus the damped-velocity prior: the latest native rate ω_t extrapolated with exponential damping, time constant 10 samples (100 ms). The basis and damping weights are precomputed constants. All 20 horizons are predicted in parallel relative to now, not by autoregressive integration.",
  "summary": "A fixed basis expands the learned coefficients into all future local rotation parameters and adds the latest rate with exponential damping.",
  "math": [
   "k∈{1,…,20}; u_k=k/20; B[k,j]=u_kʲ, j∈{1,2,3}",
   "τ=100 ms / 10 ms=10 samples; E_k=−τ expm1(−k/τ)/100",
   "r_k=Σⱼ₌₁³ C̃[j,:] u_kʲ + E_k ω_t",
   "ONNX basis=[B,E,B]: [20,7]; rows=cat(C,ω_t,bias): [1,7,3]",
   "r=basis @ rows: [1,20,3]"
  ],
  "shapes": [
   "C̃: [1,3,3]",
   "native rate: [1,1,3]",
   "basis: [20,7]",
   "rows: [1,7,3]",
   "r: [1,20,3]"
  ],
  "onnx": [
   "/Constant_53 (Constant)",
   "/Concat_7 (Concat)",
   "/Constant_54 (Constant)",
   "/MatMul_8 (MatMul)"
  ],
  "params": "0 additional; coefficient biases counted in head",
  "refs": [
   "jacob/training/best/model.py:260",
   "jacob/training/best/model.py:261",
   "jacob/training/best/model.py:338",
   "jacob/training/best/model.py:362",
   "jacob/training/best/model.py:364",
   "jacob/training/best/export.py:104",
   "jacob/training/best/export.py:106",
   "jacob/training/best/export.py:116",
   "jacob/training/best/export.py:160"
  ],
  "caveat": "The prior exponentials and polynomial basis are precomputed constants; horizons are parallel predictions relative to now, not autoregressive integration."
 },
 "out": {
  "title": "Cayley map, world composition and unit output",
  "prose": "Each local rotation vector r_k is mapped to a rotation by the Cayley map, a_k = (2, r_k), composed with the current pose, p_k = q_t ⊗ a_k, and normalised to a unit quaternion. The normalisation is expressed as a LayerNorm with fixed constants over the signed pair [p_k, −p_k], so it adds no learned parameters. The viewer's error row is the geodesic error: the rotation angle between the predicted and the true orientation.",
  "summary": "Each local parameter vector becomes a native Cayley quaternion direction, composes with the current pose and is normalised through a signed-pair LayerNorm.",
  "math": [
   "a_k=(2,r_k); p_k=q_t⊗a_k; paired=[p_k,−p_k]",
   "mean(paired)=0; var(paired)=‖p_k‖²/4",
   "LN: γ=1/2, β=0, ε=9.999999960041972e−13 (stored ONNX float)",
   "q̂_k=LN(paired)[0:4]=p_k / √(‖p_k‖²+4ε)",
   "native local angle=2 atan(‖r_k‖/2); output steps k=1,…,20 at 100 Hz"
  ],
  "shapes": [
   "r: [1,20,3]",
   "a: [1,20,4]",
   "composition table: [1,4,8]",
   "paired: [1,20,8]",
   "prediction: [1,20,4]"
  ],
  "onnx": [
   "/Constant_55 (Constant)",
   "/Concat_8 (Concat)",
   "/MatMul_9 (MatMul)",
   "/Reshape_6 (Reshape)",
   "/MatMul_10 (MatMul)",
   "/Constant_58 (Constant)",
   "/Constant_59 (Constant)",
   "/LayerNormalization (LayerNormalization)",
   "/Slice_10 (Slice)"
  ],
  "params": "0 learned; LayerNorm scale and bias are fixed constants",
  "refs": [
   "jacob/training/best/model.py:37",
   "jacob/training/best/model.py:65",
   "jacob/training/best/model.py:69",
   "jacob/training/best/model.py:365",
   "jacob/training/best/export.py:77",
   "jacob/training/best/export.py:119",
   "jacob/training/best/export.py:120",
   "jacob/training/best/export.py:161",
   "jacob/training/best/export.py:162",
   "jacob/training/best/export.py:163"
  ],
  "caveat": "With ε the norm is numerically, rather than algebraically, one; this is output quaternion normalisation, not a learned feature normalisation layer."
 },
 "gaze": {
  "title": "Head direction derived from the pose",
  "prose": "The model outputs head orientation only. The viewer derives a head-direction proxy by rotating a chosen local forward axis by each predicted quaternion; this is display-side and has no ONNX operator. It is not measured eye gaze, and the model does not prescribe which local axis is forward.",
  "summary": "The graph outputs head orientation, from which a viewer can derive a head-direction proxy using an explicitly chosen local forward axis.",
  "math": [
   "graph output: q̂_k=(w,x,y,z), k=1,…,20",
   "display-only: g_world,k=R(q̂_k) u_head, with chosen unit local viewing axis u_head",
   "R(q)ᵀ y expresses world up in the head frame; R(q) maps a head-frame direction into world coordinates"
  ],
  "shapes": [
   "prediction: [1,20,4]",
   "display-only local direction: [3]",
   "display-only world directions: [20,3]"
  ],
  "onnx": [
   "/Slice_10 (Slice)"
  ],
  "params": "0 learned; display transformation has no ONNX operator",
  "refs": [
   "jacob/training/best/model.py:59",
   "jacob/training/best/model.py:60",
   "jacob/training/best/model.py:365",
   "jacob/training/best/export.py:163",
   "jacob/training/best/export.py:164"
  ],
  "caveat": "This is a display-side head-direction proxy, not measured eye gaze; the model code does not prescribe a local forward axis."
 }
};

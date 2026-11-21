# board_build.py: the journey's circuit board, laid out like the Meta Quest 3 main board (side 1, the SoC side).
#
# Run headless (no GUI needed):
#   /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 -P board_build.py -- [--render]
# Writes models/board.blend, assets/board_q3.glb and (with --render) models/previews/board_*.png.
#
# Reference: iFixit "Meta Quest 3 Chip ID" (13 Oct 2023), photo of side 1, used for placement only (nothing copied from it).
# Decided (Nic, 16 Nov): neutral chip tops, no company logos; real part names only in the page's side panel.
# Everything is built in millimetres, then the root is scaled by 0.1, so 1 page unit = 10 mm (the journey's board level, U = 10).
# Blender is Z-up; the glTF export turns it Y-up, so the board's top face ends up at y = 0 facing +y, like the old placeholder.
# Positions come from the reference photo in pixels (board ≈ 337 px wide), converted with px(): 337 px ≈ 100 mm.
import bpy, bmesh, math, random, sys, os
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, '..', 'assets')
PREV = os.path.join(HERE, 'previews')
RENDER = '--render' in sys.argv
rnd = random.Random(7)                      # seeded: the same board every run

MM_PER_PX = 100 / 337
CX, CY = 186, 133                           # photo pixel at the board's centre
def px(x, y):                               # photo pixel -> board mm (x right, y up)
    return ((x - CX) * MM_PER_PX, (CY - y) * MM_PER_PX)

T = 1.0                                     # board thickness (mm); the top face is z = 0
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
root = bpy.data.objects.new('board_q3', None); scene.collection.objects.link(root)

# ---------------------------------------------------------------- materials
def mat(name, color, rough=0.5, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*color, 1); b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    if coat: b.inputs['Coat Weight'].default_value = coat
    return m
M = {
    'mask':   mat('solder_mask', (0.018, 0.022, 0.020), 0.38, coat=0.5),
    'trace':  mat('mask_over_copper', (0.032, 0.036, 0.030), 0.30, coat=0.5),   # copper under the mask: a shade lighter, glossier
    'core':   mat('fr4_edge', (0.30, 0.27, 0.18), 0.7),
    'epoxy':  mat('chip_epoxy', (0.028, 0.028, 0.030), 0.55),
    'die':    mat('chip_top_die', (0.020, 0.022, 0.026), 0.18, metal=0.3),  # the SoC's flip-chip die: dark, a little mirror-like
    'tin':    mat('shield_tin', (0.78, 0.77, 0.73), 0.28, metal=1.0),
    'gold':   mat('gold_pads', (1.0, 0.77, 0.42), 0.22, metal=1.0),
    'silver': mat('solder_silver', (0.82, 0.82, 0.80), 0.25, metal=1.0),
    'mlcc':   mat('mlcc_ceramic', (0.50, 0.40, 0.27), 0.6),
    'res':    mat('resistor_black', (0.03, 0.03, 0.03), 0.6),
    'plastic':mat('connector_plastic', (0.04, 0.04, 0.045), 0.5),
    'silk':   mat('silkscreen', (0.85, 0.85, 0.82), 0.7),
    'label':  mat('qr_label', (0.92, 0.92, 0.90), 0.6),
    'can':    mat('capacitor_can', (0.70, 0.71, 0.72), 0.35, metal=1.0),
}

def obj_from_bm(bm, name, material, parent=root):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); scene.collection.objects.link(o); o.parent = parent
    me.materials.append(material)
    for p in me.polygons: p.use_smooth = False
    return o

def add_box(bm, cx, cy, z0, w, d, h, rot=0.0):
    """An axis box (w x d x h mm) standing on z0, centred on (cx, cy), turned by rot radians about z."""
    r = bmesh.ops.create_cube(bm, size=1.0)
    vs = r['verts']
    bmesh.ops.scale(bm, vec=(w, d, h), verts=vs)
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(rot, 3, 'Z'), verts=vs)
    bmesh.ops.translate(bm, vec=(cx, cy, z0 + h / 2), verts=vs)

def add_cyl(bm, cx, cy, z0, r, h, seg=12):
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r, depth=h)
    bmesh.ops.translate(bm, vec=(cx, cy, z0 + h / 2), verts=res['verts'])

# ---------------------------------------------------------------- 1. the board outline
OUT_PX = [(18, 80), (92, 80), (97, 53), (262, 53), (267, 80), (355, 80), (355, 212), (268, 212), (258, 172),
          (118, 172), (108, 212), (18, 212)]
outline = [px(*p) for p in OUT_PX]
bm = bmesh.new()
vs = [bm.verts.new((x, y, 0)) for x, y in outline]
face = bm.faces.new(vs)
bmesh.ops.bevel(bm, geom=list(bm.verts), offset=2.2, segments=6, affect='VERTICES')   # rounded corners, 2.2 mm
ext = bmesh.ops.extrude_face_region(bm, geom=list(bm.faces))
bmesh.ops.translate(bm, vec=(0, 0, -T), verts=[v for v in ext['geom'] if isinstance(v, bmesh.types.BMVert)])
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
board = obj_from_bm(bm, 'pcb', M['mask'])
board.data.materials.append(M['core'])
for p in board.data.polygons:                 # the cut edge shows the FR4 core
    if abs(p.normal.z) < 0.5: p.material_index = 1

# point-in-board test (for scattering), against the un-bevelled outline shrunk a little
def inside(x, y, margin=1.8):
    n, c = len(outline), False
    for i in range(n):
        x1, y1 = outline[i]; x2, y2 = outline[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1: c = not c
    if not c: return False
    for i in range(n):                         # distance to every edge >= margin
        x1, y1 = outline[i]; x2, y2 = outline[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1; L = dx * dx + dy * dy
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / L))
        if math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)) < margin: return False
    return True

# keep-out rectangles (mm, x0, y0, x1, y1): parts placed so far, so the scatter does not land on them
keep = []
def reserve(cx, cy, w, d, pad=0.6): keep.append((cx - w / 2 - pad, cy - d / 2 - pad, cx + w / 2 + pad, cy + d / 2 + pad))
def free(x, y, r=0.0): return all(not (a - r < x < c + r and b - r < y < d + r) for a, b, c, d in keep)

# ---------------------------------------------------------------- 2. the big parts (positions from the photo)
anchors = {}
def anchor(name, x, y, z):
    e = bpy.data.objects.new(name, None); scene.collection.objects.link(e); e.parent = root
    e.location = (x, y, z); e.empty_display_size = 2; anchors[name] = e

chips, tops = bmesh.new(), bmesh.new()
def chip(name, cx_px, cy_px, w, d, h, top_bm=None):
    cx, cy = px(cx_px, cy_px)
    add_box(chips, cx, cy, 0, w, d, h); reserve(cx, cy, w, d)
    return cx, cy

# memory (LPDDR5, left) and the SoC (right), inside one shield frame
mx, my = chip('mem', 164, 96, 13.0, 13.0, 1.0)
sx, sy = px(222, 97); SOC = 15.0
add_box(chips, sx, sy, 0, SOC, SOC, 0.55)                  # SoC substrate (the package)
add_box(tops, sx, sy, 0.55, SOC * 0.62, SOC * 0.62, 0.25)  # the flip-chip die on top: a neutral dark mirror, no marking
reserve(sx, sy, SOC, SOC)
anchor('anchor_mem', mx, my, 1.0)
anchor('anchor_soc', sx, sy, 0.8)

# Wi-Fi SoC, its two front-end modules and the Bluetooth chip, inside the right shield frame
wx, wy = chip('wifi', 292, 125, 7.5, 7.5, 0.8)
chip('fem24', 318, 112, 3.0, 3.0, 0.7); chip('fem5', 318, 138, 3.0, 3.0, 0.7); chip('bt', 284, 150, 3.5, 3.5, 0.6)
anchor('anchor_wifi', wx, wy, 0.8)
# small parts on the left: audio DAC, IR front end, a few more ICs
chip('dac', 52, 180, 3.0, 3.0, 0.6); chip('irafe', 45, 128, 4.0, 3.0, 0.6)
for p in [(120, 115, 4, 4), (105, 140, 3, 5), (60, 150, 3, 3), (300, 190, 4, 3), (250, 160, 3, 3), (140, 160, 2.5, 2.5)]:
    chip('ic', p[0], p[1], p[2], p[3], 0.6)
chips_o = obj_from_bm(chips, 'chips', M['epoxy'])
tops_o = obj_from_bm(tops, 'soc_die', M['die'])

# pin-1 dots on every chip (a tiny silver dimple): from the chip boxes' top faces
# shield frames: a 0.25 mm tin wall, 1.3 mm high, with a top lip; the lids are off (as in the teardown photo)
shields = bmesh.new()
def frame(x0p, y0p, x1p, y1p, h=1.3, wall=0.25, lip=1.0):
    (x0, y1), (x1, y0) = px(x0p, y0p), px(x1p, y1p)
    w, d, cx, cy = x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2
    for (bx, by, bw, bd) in [(cx, y1 - wall / 2, w, wall), (cx, y0 + wall / 2, w, wall),
                             (x0 + wall / 2, cy, wall, d), (x1 - wall / 2, cy, wall, d)]:
        add_box(shields, bx, by, 0, bw, bd, h)
    for (bx, by, bw, bd) in [(cx, y1 - lip / 2, w, lip), (cx, y0 + lip / 2, w, lip),
                             (x0 + lip / 2, cy, lip, d), (x1 - lip / 2, cy, lip, d)]:
        add_box(shields, bx, by, h - 0.12, bw, bd, 0.12)
    reserve(cx, cy, w, d, pad=0.3)
frame(132, 64, 255, 132)
frame(268, 95, 335, 162)
obj_from_bm(shields, 'shield_frames', M['tin'])

# capacitor cans (tall aluminium polymer capacitors) with the pressure-vent cross on top
cans = bmesh.new()
for (xp, yp) in [(75, 110), (318, 172)]:
    x, y = px(xp, yp); add_cyl(cans, x, y, 0, 3.0, 4.6, seg=32); reserve(x, y, 6, 6)
obj_from_bm(cans, 'cap_cans', M['can'])
vent = bmesh.new()
for (xp, yp) in [(75, 110), (318, 172)]:
    x, y = px(xp, yp)
    add_box(vent, x, y, 4.6, 4.2, 0.25, 0.04); add_box(vent, x, y, 4.6, 0.25, 4.2, 0.04)
obj_from_bm(vent, 'cap_vents', M['res'])

# board-to-board flex connectors along the edges: black body, two rows of gold contacts, metal hold-downs at the ends
conn, pins, holds = bmesh.new(), bmesh.new(), bmesh.new()
CONNS = [(32, 150, 90), (32, 186, 90), (65, 205, 0), (128, 64, 0), (240, 64, 0), (340, 120, 90), (340, 195, 90),
         (300, 205, 0), (205, 160, 0), (95, 92, 90)]
for (xp, yp, deg) in CONNS:
    x, y = px(xp, yp); rot = math.radians(deg); L, Wd = 8.0, 2.6
    add_box(conn, x, y, 0, L, Wd, 0.8, rot)
    c, s = math.cos(rot), math.sin(rot)
    n = 14
    for row in (-1, 1):
        for i in range(n):
            u = (i - (n - 1) / 2) * 0.4; v = row * (Wd / 2 + 0.2)
            add_box(pins, x + u * c - v * s, y + u * s + v * c, 0, 0.18, 0.5, 0.12, rot)
    for end in (-1, 1):
        u = end * (L / 2 + 0.35)
        add_box(holds, x + u * c, y + u * s, 0, 0.5, Wd + 0.4, 0.5, rot)
    reserve(x, y, L + 1.2 if deg == 0 else Wd + 1.4, Wd + 1.4 if deg == 0 else L + 1.2)
obj_from_bm(conn, 'connectors', M['plastic']); obj_from_bm(pins, 'connector_pins', M['gold']); obj_from_bm(holds, 'connector_holds', M['tin'])

# mounting holes: a plated gold ring with a dark hole
rings = bmesh.new(); holes = bmesh.new()
for (xp, yp) in [(26, 88), (346, 88), (26, 204), (346, 204), (104, 62), (255, 62), (130, 164), (246, 164)]:
    x, y = px(xp, yp)
    if not inside(x, y, 0.5): continue
    add_cyl(rings, x, y, 0, 1.9, 0.03, seg=28); add_cyl(holes, x, y, -T - 0.01, 1.05, T + 0.06, seg=24); reserve(x, y, 4, 4)
obj_from_bm(rings, 'mount_rings', M['gold']); obj_from_bm(holes, 'mount_holes', M['res'])

# the QR label (a white sticker); the code itself is a random grid drawn as raised dark squares
qx, qy = px(90, 160); Q = 5.5
lab = bmesh.new(); add_box(lab, qx, qy, 0, Q, Q, 0.05); obj_from_bm(lab, 'qr_label', M['label']); reserve(qx, qy, Q, Q)
qr = bmesh.new(); cell = Q * 0.84 / 21
for i in range(21):
    for j in range(21):
        finder = any(i0 <= i < i0 + 7 and j0 <= j < j0 + 7 for i0, j0 in [(0, 0), (14, 0), (0, 14)])
        on = (finder and not (1 <= (i % 14 if i >= 14 else i) <= 5 and 1 <= (j % 14 if j >= 14 else j) <= 5)) or \
             (finder and 2 <= (i % 14 if i >= 14 else i) <= 4 and 2 <= (j % 14 if j >= 14 else j) <= 4) or \
             (not finder and rnd.random() < 0.5)
        if on: add_box(qr, qx - Q * 0.42 + (i + .5) * cell, qy - Q * 0.42 + (j + .5) * cell, 0.05, cell, cell, 0.01)
obj_from_bm(qr, 'qr_code', M['res'])

# ---------------------------------------------------------------- 3. passives: 0402 and 0201 resistors and capacitors
bodies_c, bodies_r, caps = bmesh.new(), bmesh.new(), bmesh.new()
def passive(x, y, rot, size):
    L, Wd, H = (1.0, 0.5, 0.35) if size == '0402' else (0.6, 0.3, 0.3)
    e = L * 0.2; c, s = math.cos(rot), math.sin(rot)
    is_cap = rnd.random() < 0.65
    add_box(bodies_c if is_cap else bodies_r, x, y, 0, L - 2 * e, Wd, H, rot)
    for end in (-1, 1):
        u = end * (L / 2 - e / 2)
        add_box(caps, x + u * c, y + u * s, 0, e, Wd, H, rot)
placed = 0
# Real boards are not evenly covered: parts sit in tight, aligned groups next to the chip they serve (decoupling caps on the
# power pins, resistors on the signal lines), with open solder mask between the groups (Nic, 16 Nov: "a lot of capacitors,
# looks good but not right"). So: one short row of caps along each side of the big chips, then small aligned clusters next to
# the small ICs and connectors, and nothing anywhere else.
# rows of caps along the OUTSIDE of the two shield frames (inside, the frame hugs the chips), in runs with gaps
for (x0p, y0p, x1p, y1p) in [(132, 64, 255, 132), (268, 95, 335, 162)]:
    (x0, y1), (x1, y0) = px(x0p, y0p), px(x1p, y1p)
    for (ax, ay, bx, by, rot) in [(x0, y0 - 1.0, x1, y0 - 1.0, 0), (x0, y1 + 1.0, x1, y1 + 1.0, 0),
                                  (x0 - 1.0, y0, x0 - 1.0, y1, math.pi / 2), (x1 + 1.0, y0, x1 + 1.0, y1, math.pi / 2)]:
        L = math.hypot(bx - ax, by - ay); n = int(L / 1.0); run = 0
        for i in range(n):
            if run == 0 and rnd.random() < 0.35: run = -rnd.randint(2, 5)       # a gap
            if run < 0: run += 1; continue
            run = run + 1 if run < rnd.randint(4, 9) else 0
            f = (i + .5) / n; x, y = ax + (bx - ax) * f, ay + (by - ay) * f
            if inside(x, y) and free(x, y, 0.1):
                passive(x, y, rot, '0201'); reserve(x, y, 0.7, 0.7, pad=0.02); placed += 1
cluster_at = []                                     # (x, y, how many): beside each small IC, connector and can
for (xp, yp) in [(52, 168), (58, 128), (120, 100), (118, 140), (60, 162), (300, 178), (250, 172), (140, 150), (92, 118)]:
    cluster_at.append((*px(xp, yp), rnd.randint(12, 22)))
for (xp, yp, deg) in CONNS:
    x, y = px(xp, yp); out = (-math.sin(math.radians(deg)), math.cos(math.radians(deg)))
    side = 1 if y < 0 else -1                        # toward the board's middle
    cluster_at.append((x + out[0] * 3.2 * side, y + out[1] * 3.2 * side, rnd.randint(6, 12)))
for (cx, cy, n) in cluster_at:                       # a small aligned grid: rows 1.2 mm apart, parts 0.9 mm apart
    rot = 0 if rnd.random() < 0.5 else math.pi / 2; cols = rnd.choice([2, 3, 4])
    for k in range(n):
        u, v = (k % cols - (cols - 1) / 2) * 1.3, (k // cols) * 0.9
        x, y = (cx + u, cy + v) if rot == 0 else (cx + v, cy + u)
        size = '0402' if rnd.random() < 0.3 else '0201'
        if inside(x, y) and free(x, y, 0.15):
            passive(x, y, rot, size); reserve(x, y, 1.0, 0.6, pad=0.05); placed += 1
obj_from_bm(bodies_c, 'passives_caps', M['mlcc']); obj_from_bm(bodies_r, 'passives_res', M['res']); obj_from_bm(caps, 'passive_ends', M['silver'])

# ---------------------------------------------------------------- 4. vias and surface traces (under the mask)
via_bm = bmesh.new(); n_via = 0
for (cx, cy, n) in cluster_at + [(sx, sy, 30), (mx, my, 20), (wx, wy, 12)]:   # vias gather where the parts are, too
    for k in range(n):
        a = rnd.uniform(0, 2 * math.pi); r = rnd.uniform(2.0, 6.0)
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        if inside(x, y, 1.0) and free(x, y, 0.15):
            add_cyl(via_bm, x, y, 0, 0.16, 0.015, seg=10); reserve(x, y, 0.35, 0.35, pad=0.02); n_via += 1
obj_from_bm(via_bm, 'vias', M['trace'])   # tented vias: covered by the mask, so they read as faint dimples, not bare metal
tr = bmesh.new()
def trace(pts, w=0.12):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-3: continue
        add_box(tr, (x0 + x1) / 2, (y0 + y1) / 2, 0, L + w, w, 0.012, math.atan2(y1 - y0, x1 - x0))
for (cx, cy, half) in [(sx, sy, SOC / 2), (mx, my, 6.5), (wx, wy, 3.75)]:   # fan-out: out from each chip edge, a 45° jog, then away
    for k in range(48):
        side = k % 4; t = (k // 4) / 11 - 0.5
        out = [(0, 1), (0, -1), (1, 0), (-1, 0)][side]; along = (out[1], out[0])
        x0, y0 = cx + out[0] * (half + 0.2) + along[0] * t * 2 * half * 0.9, cy + out[1] * (half + 0.2) + along[1] * t * 2 * half * 0.9
        a = 1.5 + rnd.random() * 2.5; j = (1 if t > 0 else -1) * (1 + rnd.random() * 2)
        x1, y1 = x0 + out[0] * a, y0 + out[1] * a
        x2, y2 = x1 + out[0] * abs(j) + along[0] * j, y1 + out[1] * abs(j) + along[1] * j
        b = 4 + rnd.random() * 10; x3, y3 = x2 + out[0] * b, y2 + out[1] * b
        pts = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        if all(inside(x, y, 1.2) for x, y in pts): trace(pts)
obj_from_bm(tr, 'traces', M['trace'])

# ---------------------------------------------------------------- 5. silkscreen: reference designators next to the parts
def silk_text(s, x, y, size=0.9, rot=0.0):
    cu = bpy.data.curves.new('silk', 'FONT'); cu.body = s; cu.size = size; cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
    o = bpy.data.objects.new('silk_' + s, cu); scene.collection.objects.link(o)
    o.location = (x, y, 0.005); o.rotation_euler = (0, 0, rot)
    return o
silk = []
for i, (xp, yp, s) in enumerate([(222, 136, 'U1'), (164, 136, 'U2'), (292, 166, 'U10'), (52, 192, 'U21'), (45, 140, 'U22'),
                                 (75, 128, 'C1'), (318, 190, 'C2'), (120, 128, 'U31'), (105, 155, 'U32'), (300, 200, 'U40')]):
    x, y = px(xp, yp); silk.append(silk_text(s, x, y))
for k, (cx, cy, n) in enumerate(cluster_at):         # one small designator beside each passive cluster
    for dx, dy in [(0, -1.6), (2.8, 0), (-2.8, 0), (0, 4.5)]:
        x, y = cx + dx, cy + dy
        if inside(x, y, 2) and free(x, y, 0.6):
            silk.append(silk_text(('C' if k % 3 else 'R') + str(100 + k * 7), x, y, 0.55)); break
x, y = px(150, 150); silk.append(silk_text('MP-K8 · REV C', x, y, 1.1))   # our own board name, not the real one
for o in silk:
    o.select_set(True)
bpy.context.view_layer.objects.active = silk[0]
bpy.ops.object.convert(target='MESH')
bpy.ops.object.join()
silk_o = bpy.context.view_layer.objects.active; silk_o.name = 'silkscreen'; silk_o.parent = root
silk_o.data.materials.clear(); silk_o.data.materials.append(M['silk'])

# ---------------------------------------------------------------- 6. scale to page units, save, export
root.scale = (0.1, 0.1, 0.1)
bpy.context.view_layer.update()
os.makedirs(ASSETS, exist_ok=True); os.makedirs(PREV, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, 'board.blend'))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(filepath=os.path.join(ASSETS, 'board_q3.glb'), export_format='GLB', use_selection=True,
                          export_apply=True, export_animations=False, export_yup=True)
tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in scene.objects if o.type == 'MESH')
print('BOARD', 'passives', placed, 'vias', n_via, 'tris', tris, 'glb', os.path.getsize(os.path.join(ASSETS, 'board_q3.glb')))

# ---------------------------------------------------------------- 7. previews
if RENDER:
    scene.render.engine = 'CYCLES'; scene.cycles.samples = 64; scene.cycles.device = 'CPU'
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1000
    world = bpy.data.worlds.new('w'); scene.world = world; world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.004, 0.004, 0.005, 1)
    def light(name, loc, energy, size=3.0, color=(1, 1, 1)):
        l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.size = size; l.color = color
        o = bpy.data.objects.new(name, l); scene.collection.objects.link(o); o.location = loc
        o.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    light('key', (-6, 4, 5), 90, 4, (1, .95, .88)); light('rim', (6, -4, 3), 40, 3, (.85, .9, 1)); light('top', (0, 0, 8), 20, 6)
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); scene.collection.objects.link(cam); scene.camera = cam
    for name, loc, tgt, lens in [('top', (0, -0.01, 16), (0, 0, 0), 50), ('wide', (2.5, -7.5, 6.5), (0, 0, 0), 50), ('soc', (2.6, -2.0, 2.2), (0.55, 0.35, 0), 70),
                                 ('wifi', (4.6, -1.8, 1.9), (3.2, -0.3, 0), 75)]:
        cam.location = loc; cam.data.lens = lens
        cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = os.path.join(PREV, f'board_{name}.png')
        bpy.ops.render.render(write_still=True)
    print('RENDERED')

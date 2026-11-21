# board_build.py: a replica of the Meta Quest 3 main board, side 1 (the SoC side), for the journey's board level.
#
# Run headless (no GUI needed):
#   /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 -P board_build.py -- [--render]
# Writes models/board.blend, assets/board_q3.glb and (with --render) models/previews/board_*.png.
#
# Reference: iFixit "Meta Quest 3 Chip ID" (13 Oct 2023), step 1 photo of side 1. Every part's position and size is measured off
# that photo in pixels and converted with px(); nothing from the photo itself (no image, no texture) goes into the model.
# Decided (Nic, 16 Nov, late): replicate the board exactly, chips carry their real markings (as printed on the parts). Meta's logo
# on the silkscreen is left out (same call as the headset's logo); the board's number 844-02465-08 is kept.
# The old generic version is kept as board_build_v1_generic.py.
#
# Units: built in millimetres, then the root is scaled by 0.1, so 1 page unit = 10 mm (the journey's board level, U = 10).
# Blender is Z-up; the glTF export turns it Y-up, so the board's top face ends up at y = 0 facing +y.
import bpy, bmesh, math, random, sys, os
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, '..', 'assets')
PREV = os.path.join(HERE, 'previews')
RENDER = '--render' in sys.argv
rnd = random.Random(7)                      # seeded: the same board every run

# Photo scale: the board is ~1627 px wide in the reference; the SoC measures ~200 px. 0.0615 mm/px makes the board 100 mm wide
# and the SoC 12.3 mm, which is the size class of a phone/XR SoC package. Absolute size is an estimate; proportions are measured.
MM_PER_PX = 0.0615
CX, CY = 921, 677                           # photo pixel at the board's centre
def px(x, y):                               # photo pixel -> board mm (x right, y up)
    return ((x - CX) * MM_PER_PX, (CY - y) * MM_PER_PX)
def L(p): return p * MM_PER_PX              # a photo length in px -> mm

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
    'mask':   mat('solder_mask', (0.022, 0.018, 0.012), 0.40, coat=0.5),        # the photo's mask: black with a warm brown cast
    'trace':  mat('mask_over_copper', (0.036, 0.030, 0.020), 0.32, coat=0.5),
    'core':   mat('fr4_edge', (0.30, 0.27, 0.18), 0.7),
    'epoxy':  mat('chip_epoxy', (0.030, 0.030, 0.032), 0.55),
    'die':    mat('soc_top', (0.012, 0.014, 0.020), 0.12, metal=0.4),            # the SoC's top: dark and glossy (reflects in the photo)
    'mark':   mat('laser_marking', (0.20, 0.20, 0.20), 0.7),                     # laser-etched text on chip tops: a lighter grey
    'tin':    mat('shield_tin', (0.62, 0.60, 0.55), 0.55, metal=1.0),            # the shield frames: matt, slightly brown tin
    'steel':  mat('standoff_steel', (0.80, 0.80, 0.80), 0.30, metal=1.0),
    'gold':   mat('gold_plating', (1.0, 0.77, 0.42), 0.25, metal=1.0),
    'silver': mat('solder_silver', (0.82, 0.82, 0.80), 0.25, metal=1.0),
    'mlcc':   mat('mlcc_ceramic', (0.50, 0.40, 0.27), 0.6),
    'res':    mat('resistor_black', (0.03, 0.03, 0.03), 0.6),
    'plastic':mat('connector_plastic', (0.035, 0.035, 0.04), 0.5),
    'silk':   mat('silkscreen', (0.72, 0.60, 0.36), 0.7),                        # the photo's silkscreen is a pale gold
    'label':  mat('qr_label', (0.93, 0.93, 0.91), 0.6),
    'ink':    mat('label_ink', (0.03, 0.03, 0.03), 0.7),
    'can':    mat('capacitor_can', (0.72, 0.73, 0.74), 0.35, metal=1.0),
    'blue':   mat('wlcsp_blue', (0.05, 0.10, 0.28), 0.25, metal=0.2),            # the small blue-grey part left of the left can
    'led':    mat('led_white', (0.95, 0.93, 0.80), 0.3),
    'phos':   mat('led_phosphor', (0.95, 0.80, 0.20), 0.4),
    'tan':    mat('inductor_tan', (0.62, 0.50, 0.34), 0.6),
}

def obj_from_bm(bm, name, material, parent=root):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); scene.collection.objects.link(o); o.parent = parent
    me.materials.append(material)
    for p in me.polygons: p.use_smooth = False
    return o

def add_box(bm, cx, cy, z0, w, d, h, rot=0.0):
    """A box (w x d x h mm) standing on z0, centred on (cx, cy), turned by rot radians about z."""
    r = bmesh.ops.create_cube(bm, size=1.0); vs = r['verts']
    bmesh.ops.scale(bm, vec=(w, d, h), verts=vs)
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(rot, 3, 'Z'), verts=vs)
    bmesh.ops.translate(bm, vec=(cx, cy, z0 + h / 2), verts=vs)

def add_cyl(bm, cx, cy, z0, r, h, seg=12, r2=None):
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r if r2 is None else r2, depth=h)
    bmesh.ops.translate(bm, vec=(cx, cy, z0 + h / 2), verts=res['verts'])

def add_ring(bm, cx, cy, z0, r_out, r_in, h, seg=32):
    """A flat washer: outer radius r_out, hole r_in."""
    for k in range(seg):
        a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
        pts = [(r_in, a0), (r_out, a0), (r_out, a1), (r_in, a1)]
        bot = [bm.verts.new((cx + r * math.cos(a), cy + r * math.sin(a), z0)) for r, a in pts]
        top = [bm.verts.new((cx + r * math.cos(a), cy + r * math.sin(a), z0 + h)) for r, a in pts]
        bm.faces.new(top); bm.faces.new(bot[::-1])
        for i in range(4):
            j = (i + 1) % 4; bm.faces.new((bot[i], bot[j], top[j], top[i]))

def pbox(bm, x0p, y0p, x1p, y1p, z0, h):
    """A box given by its photo-pixel corners."""
    (ax, ay), (bx, by) = px(x0p, y0p), px(x1p, y1p)
    add_box(bm, (ax + bx) / 2, (ay + by) / 2, z0, abs(bx - ax), abs(by - ay), h)

# text: Blender's built-in font, turned into a flat mesh just above the surface it sits on
text_objs = {'mark': [], 'silk': [], 'ink': []}
def text(s, xp, yp, h_mm, z, kind='mark', rot_deg=0, align='CENTER'):
    cu = bpy.data.curves.new('t', 'FONT'); cu.body = s; cu.size = h_mm; cu.align_x = align; cu.align_y = 'CENTER'
    cu.space_line = 1.05
    o = bpy.data.objects.new('t_' + s.split('\n')[0], cu); scene.collection.objects.link(o)
    x, y = px(xp, yp); o.location = (x, y, z + 0.004); o.rotation_euler = (0, 0, math.radians(rot_deg))
    text_objs[kind].append(o)

# ---------------------------------------------------------------- 1. the board outline (traced off the photo)
OUT_PX = [(480, 282), (1305, 282), (1305, 437), (1712, 437), (1735, 1060), (1410, 1060), (1410, 870),
          (440, 870), (440, 1075), (108, 1075), (148, 437), (480, 437)]
outline = [px(*p) for p in OUT_PX]
bm = bmesh.new()
bm.faces.new([bm.verts.new((x, y, 0)) for x, y in outline])
bmesh.ops.bevel(bm, geom=list(bm.verts), offset=1.6, segments=6, affect='VERTICES')
ext = bmesh.ops.extrude_face_region(bm, geom=list(bm.faces))
bmesh.ops.translate(bm, vec=(0, 0, -T), verts=[v for v in ext['geom'] if isinstance(v, bmesh.types.BMVert)])
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
board = obj_from_bm(bm, 'pcb', M['mask'])
board.data.materials.append(M['core'])
for p in board.data.polygons:
    if abs(p.normal.z) < 0.5: p.material_index = 1

def inside(x, y, margin=1.0):
    n, c = len(outline), False
    for i in range(n):
        x1, y1 = outline[i]; x2, y2 = outline[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1: c = not c
    if not c: return False
    for i in range(n):
        x1, y1 = outline[i]; x2, y2 = outline[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1; LL = dx * dx + dy * dy
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / LL))
        if math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)) < margin: return False
    return True

keep = []                                    # occupied rectangles (mm), so the passives don't land on parts
def reserve_px(x0p, y0p, x1p, y1p, pad=0.3):
    (ax, ay), (bx, by) = px(x0p, y0p), px(x1p, y1p)
    keep.append((min(ax, bx) - pad, min(ay, by) - pad, max(ax, bx) + pad, max(ay, by) + pad))
def reserve(cx, cy, w, d, pad=0.05): keep.append((cx - w / 2 - pad, cy - d / 2 - pad, cx + w / 2 + pad, cy + d / 2 + pad))
def free(x, y, r=0.0): return all(not (a - r < x < c + r and b - r < y < d + r) for a, b, c, d in keep)

anchors = {}
def anchor(name, xp, yp, z):
    x, y = px(xp, yp)
    e = bpy.data.objects.new(name, None); scene.collection.objects.link(e); e.parent = root
    e.location = (x, y, z); e.empty_display_size = 2; anchors[name] = e

# ---------------------------------------------------------------- 2. chips, with their real markings
chips, tops = bmesh.new(), bmesh.new()

# SK hynix H58G66BK8BX105: 8 GB LPDDR5 (marking printed sideways, reading top to bottom)
pbox(chips, 715, 400, 940, 615, 0, 1.0); reserve_px(715, 400, 940, 615)
text('SK hynix\nH58G66BK8B\nX105   323A', 840, 505, 0.62, 1.0, rot_deg=-90)
text('8TAJHS41QC1', 745, 505, 0.5, 1.0, rot_deg=-90)
dm = bmesh.new()                             # its little 2D code, bottom right of the top
for i in range(12):
    for j in range(12):
        if i == 0 or j == 0 or rnd.random() < 0.45:
            x0, y0 = px(900 + i * 3.2, 600 - j * 3.2); add_box(dm, x0, y0, 1.0, L(3.0), L(3.0), 0.006)
anchor('anchor_mem', 828, 508, 1.0)

# Qualcomm SXR2230P 100-AB: Snapdragon XR2 Gen 2. Thin package, glossy dark top (reflects like the photo's), marking top left
pbox(chips, 975, 410, 1175, 615, 0, 0.55); pbox(tops, 980, 415, 1170, 610, 0.55, 0.12); reserve_px(975, 410, 1175, 615)
text('Qualcomm\nSXR2230P\n100-AB', 1028, 455, 0.55, 0.67, align='LEFT')
text('JC3252MZ', 1128, 576, 0.5, 0.67)
dot = bmesh.new(); x, y = px(1033, 577); add_cyl(dot, x, y, 0.67, 0.25, 0.006, seg=16)
anchor('anchor_soc', 1075, 512, 0.67)

# Qualcomm WCN6856-102: Wi-Fi 6E (marking sideways)
pbox(chips, 1410, 625, 1530, 745, 0, 0.8); reserve_px(1410, 625, 1530, 745)
text('Qualcomm\nWCN6856\n102', 1480, 680, 0.5, 0.8, rot_deg=-90)
text('HK3237CB', 1430, 680, 0.4, 0.8, rot_deg=-90)
anchor('anchor_wifi', 1470, 685, 0.8)
# Qorvo QM45392 (5 GHz front end) and QM42391 (2.4 GHz front end)
pbox(chips, 1365, 548, 1440, 608, 0, 0.7); reserve_px(1365, 548, 1440, 608)
text('45392\nFBVW', 1402, 580, 0.38, 0.7)
pbox(chips, 1518, 548, 1593, 608, 0, 0.7); reserve_px(1518, 548, 1593, 608)
text('QM42391\nEXB9', 1555, 580, 0.36, 0.7)
anchor('anchor_fem', 1478, 578, 0.7)
# Triad TS5231 (probably): IR analog front end, in its dashed silkscreen box
pbox(chips, 510, 592, 562, 640, 0, 0.6); reserve_px(510, 592, 562, 640)
anchor('anchor_ir', 536, 616, 0.6)
# the blue-grey part left of the left can (a small module, marked 840-05446-00 on its label)
blue = bmesh.new(); pbox(blue, 207, 615, 300, 650, 0, 0.5); reserve_px(207, 615, 300, 650)
text('840-05446-00', 253, 632, 0.32, 0.5)
# smaller ICs that the photo shows but iFixit does not name (QFN/WLCSP packages around the board)
for (x0p, y0p, x1p, y1p) in [(300, 862, 340, 895), (330, 905, 360, 930), (1290, 785, 1320, 805), (1025, 672, 1045, 688),
                             (580, 375, 610, 400)]:
    pbox(chips, x0p, y0p, x1p, y1p, 0, 0.5); reserve_px(x0p, y0p, x1p, y1p)
anchor('anchor_dac', 320, 878, 0.5)          # AKM AK4333 audio DAC: iFixit marks it in this area of side 1
obj_from_bm(chips, 'chips', M['epoxy']); obj_from_bm(tops, 'soc_top', M['die']); obj_from_bm(dm, 'mem_code', M['mark'])
obj_from_bm(dot, 'soc_pin1', M['mark']); obj_from_bm(blue, 'blue_module', M['blue'])

# ---------------------------------------------------------------- 3. shield frames (lids removed, as in the photo)
# A stamped tin frame: a flat flange around the opening, on a short wall. The centre one hugs memory + SoC; the right one wraps
# the Wi-Fi block (its opening is offset: the flange is wider on the left and bottom).
sh = bmesh.new()
def frame(o, i, h=0.75, t=0.15):
    (ox0, oy1), (ox1, oy0) = px(o[0], o[1]), px(o[2], o[3])          # outer edge
    (ix0, iy1), (ix1, iy0) = px(i[0], i[1]), px(i[2], i[3])          # opening
    for (a, b, c, d) in [(ox0, iy1, ox1, oy1), (ox0, oy0, ox1, iy0), (ox0, iy0, ix0, iy1), (ix1, iy0, ox1, iy1)]:
        add_box(sh, (a + c) / 2, (b + d) / 2, h - t, c - a, d - b, t)          # the flange
    for (a, b, c, d) in [(ox0, oy1 - t, ox1, oy1), (ox0, oy0, ox1, oy0 + t), (ox0, oy0, ox0 + t, oy1), (ox1 - t, oy0, ox1, oy1)]:
        add_box(sh, (a + c) / 2, (b + d) / 2, 0, c - a, d - b, h - t)          # the wall
    reserve_px(o[0], o[1], o[2], o[3], pad=0.2)
frame((675, 360, 1270, 665), (712, 395, 1235, 625))
frame((1315, 505, 1625, 775), (1355, 538, 1600, 752))
obj_from_bm(sh, 'shield_frames', M['tin'])

# ---------------------------------------------------------------- 4. connectors
conn, pins, holds, goldbody = bmesh.new(), bmesh.new(), bmesh.new(), bmesh.new()
def b2b(xp, yp, len_px, w_px, vertical):
    """A board-to-board flex receptacle, open so you can see where the flex plugs in (Nic, 16 Nov: "you don't see the input of
    the connectors"). From the outside in: a low black housing wall all round; an open cavity with a floor; a raised centre
    tongue down the middle; and on both long walls a row of gold contacts that wrap over the wall top and run down into the
    cavity, at a 0.35 mm pitch like the real parts. Metal hold-down tabs at both ends."""
    x, y = px(xp, yp); Lm, Wm = L(len_px), L(w_px); rot = math.pi / 2 if vertical else 0
    c, s = math.cos(rot), math.sin(rot)
    def at(u, v): return x + u * c - v * s, y + u * s + v * c       # connector-local (u along, v across) -> board mm
    H, wall, floor = 0.6, 0.28, 0.12
    add_box(conn, *at(0, 0), 0, Lm, Wm, floor, rot)                                   # cavity floor
    for side in (-1, 1):
        add_box(conn, *at(0, side * (Wm / 2 - wall / 2)), 0, Lm, wall, H, rot)        # long walls
        add_box(conn, *at(side * (Lm / 2 - wall / 2), 0), 0, wall, Wm, H, rot)        # end walls
    tw = max(0.3, Wm * 0.26)
    add_box(conn, *at(0, 0), 0, Lm - 2 * wall - 0.5, tw, H * 0.82, rot)               # the centre tongue
    pitch = 0.35; n = max(6, int((Lm - 2 * wall - 0.6) / pitch))
    for side in (-1, 1):
        for i in range(n):
            u = (i - (n - 1) / 2) * pitch
            add_box(pins, *at(u, side * (Wm / 2 - wall / 2)), H, 0.14, wall + 0.05, 0.025, rot)          # over the wall top
            add_box(pins, *at(u, side * (Wm / 2 - wall - 0.02)), floor, 0.14, 0.05, H - floor, rot)     # down the inner face
            add_box(pins, *at(u, side * (Wm / 2 + 0.12)), 0, 0.14, 0.26, 0.04, rot)                     # solder foot outside
    for end in (-1, 1):
        add_box(holds, *at(end * (Lm / 2 + 0.2), 0), 0, 0.35, Wm * 0.8, 0.45, rot)
    reserve(x, y, (Wm if vertical else Lm) + 0.8, (Lm if vertical else Wm) + 0.8)
for args in [(148, 698, 165, 35, True), (305, 795, 145, 35, False), (318, 468, 85, 30, False), (145, 915, 110, 35, True),
             (322, 955, 210, 40, False), (330, 1042, 80, 30, False), (1697, 690, 150, 35, True), (1608, 925, 100, 35, True),
             (1705, 955, 100, 30, True), (1527, 1040, 205, 40, False)]:
    b2b(*args)
# the small gold-bodied connector top right
pbox(goldbody, 1578, 445, 1655, 478, 0, 0.6); reserve_px(1578, 445, 1655, 478)
for i in range(7):
    x, y = px(1585 + i * 11, 462); add_box(pins, x, y, 0.6, 0.2, 1.2, 0.03)
# U.FL antenna sockets: a gold square base with a round post (ANT1, ANT0, and one on the left)
ufl = bmesh.new()
for (xp, yp) in [(1400, 462), (1520, 462), (382, 693)]:
    x, y = px(xp, yp); add_box(ufl, x, y, 0, 2.6, 2.6, 0.35); add_cyl(ufl, x, y, 0.35, 0.9, 0.9, seg=24)
    reserve(x, y, 2.6, 2.6, pad=0.3)
# small wire-to-board headers (black, 2 and 4 pins)
hdr = bmesh.new()
for (x0p, y0p, x1p, y1p, n, vert) in [(165, 492, 215, 548, 2, True), (1343, 805, 1425, 838, 4, False), (1668, 808, 1712, 862, 2, True)]:
    pbox(hdr, x0p, y0p, x1p, y1p, 0, 1.4); reserve_px(x0p, y0p, x1p, y1p)
    (ax, ay), (bx, by) = px(x0p, y0p), px(x1p, y1p)
    for k in range(n):
        f = (k + .5) / n
        x, y = (ax + (bx - ax) * 0.5, ay + (by - ay) * f) if vert else (ax + (bx - ax) * f, ay + (by - ay) * 0.5)
        add_box(pins, x, y, 1.4, 0.35, 0.35, 0.02)
obj_from_bm(conn, 'connectors', M['plastic']); obj_from_bm(pins, 'connector_pins', M['gold'])
obj_from_bm(holds, 'connector_holds', M['tin']); obj_from_bm(goldbody, 'gold_connector', M['gold'])
obj_from_bm(ufl, 'ufl_sockets', M['gold']); obj_from_bm(hdr, 'wire_headers', M['plastic'])

# ---------------------------------------------------------------- 5. mechanical: standoffs, mounting holes, cans, LED, label
std, std_hole = bmesh.new(), bmesh.new()
for (xp, yp) in [(540, 455), (1305, 447), (655, 715), (1200, 715)]:          # threaded steel standoffs with a flange
    x, y = px(xp, yp)
    add_ring(std, x, y, 0, 1.9, 0.8, 0.25); add_ring(std, x, y, 0.25, 1.35, 0.8, 1.4)
    add_cyl(std_hole, x, y, 0.2, 0.8, 1.44, seg=24); reserve(x, y, 3.8, 3.8, pad=0.3)
obj_from_bm(std, 'standoffs', M['steel']); obj_from_bm(std_hole, 'standoff_holes', M['res'])
rings, holes = bmesh.new(), bmesh.new()
for (xp, yp, r_px, hole_px) in [(507, 318, 25, 13), (1278, 318, 25, 13), (1663, 540, 26, 14), (237, 880, 26, 14),
                                (862, 825, 26, 14), (1610, 812, 26, 14), (590, 310, 12, 0), (1192, 305, 12, 0), (190, 800, 12, 0)]:
    x, y = px(xp, yp)
    # plating 0.05 mm proud of the mask; the hole's dark cap stops 0.04 mm lower, so no two faces share a height (they flickered)
    if hole_px: add_ring(rings, x, y, 0, L(r_px), L(hole_px), 0.05); add_cyl(holes, x, y, -T - 0.01, L(hole_px) - 0.02, T + 0.02, seg=24)
    else: add_cyl(rings, x, y, 0, L(r_px), 0.05, seg=20)
    reserve(x, y, 2 * L(r_px), 2 * L(r_px), pad=0.4)
x, y = px(293, 550)                                                         # the larger plated hole top left
add_ring(rings, x, y, 0, 1.95, 1.15, 0.05, seg=40); add_cyl(holes, x, y, -T - 0.01, 1.13, T + 0.02, seg=32)
reserve(x, y, 4, 4, pad=0.3)
obj_from_bm(rings, 'mount_rings', M['gold']); obj_from_bm(holes, 'mount_holes', M['res'])
cans = bmesh.new(); ctop = bmesh.new()
for (xp, yp, rot) in [(400, 582, 90), (1522, 840, -90)]:                    # polymer caps, "100 VFT A73" printed on top
    x, y = px(xp, yp); add_cyl(cans, x, y, 0, 3.4, 5.0, seg=40); add_cyl(ctop, x, y, 5.0, 3.15, 0.01, seg=40)
    text('100\nVFT\nA73', xp, yp, 1.05, 5.01, kind='ink', rot_deg=rot); reserve(x, y, 7, 7, pad=0.3)
obj_from_bm(cans, 'cap_cans', M['can']); obj_from_bm(ctop, 'cap_tops', M['can'])
led, phos = bmesh.new(), bmesh.new()                                        # the status LED at the top
pbox(led, 893, 305, 940, 338, 0, 0.6); pbox(phos, 900, 312, 933, 331, 0.6, 0.01); reserve_px(893, 305, 940, 338)
obj_from_bm(led, 'led', M['led']); obj_from_bm(phos, 'led_window', M['phos'])
anchor('anchor_led', 916, 322, 0.6)
lab, qr = bmesh.new(), bmesh.new()                                          # the white serial label with its 2D code
pbox(lab, 417, 757, 503, 843, 0, 0.06); reserve_px(417, 757, 503, 843)
for i in range(16):
    for j in range(16):
        if i == 0 or j == 0 or (i == 15 and j % 2 == 0) or (j == 15 and i % 2 == 0) or rnd.random() < 0.5:
            x, y = px(428 + i * 4.0 + 2, 832 - j * 4.0 - 2); add_box(qr, x, y, 0.06, L(4.0), L(4.0), 0.008)
obj_from_bm(lab, 'serial_label', M['label']); obj_from_bm(qr, 'serial_code', M['ink'])
tan = bmesh.new()                                                           # two tan power inductors left of the U.FL socket
for (x0p, y0p, x1p, y1p) in [(210, 675, 230, 705), (234, 675, 254, 705)]:
    pbox(tan, x0p, y0p, x1p, y1p, 0, 0.8); reserve_px(x0p, y0p, x1p, y1p)
obj_from_bm(tan, 'inductors', M['tan'])

# ---------------------------------------------------------------- 6. silkscreen (pale gold): labels, corner brackets, the IR box
text('01', 640, 308, 1.4, 0, kind='silk')
text('ANT1', 1357, 465, 0.7, 0, kind='silk', rot_deg=90)
text('ANT0', 1477, 465, 0.7, 0, kind='silk', rot_deg=90)
text('844-02465-08', 740, 845, 1.75, 0, kind='silk')                        # the board number (Meta's logo left of it is left out)
silk = bmesh.new()
def bracket(xp, yp, w_px, h_px, arm=0.35):
    """Silkscreen L-brackets at the four corners of a part's outline."""
    (x0, y1), (x1, y0) = px(xp - w_px / 2, yp - h_px / 2), px(xp + w_px / 2, yp + h_px / 2)
    a, t = min(1.6, (x1 - x0) * arm), 0.15
    for (cx, cy, sx, sy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        add_box(silk, cx + sx * a / 2, cy, 0, a, t, 0.01); add_box(silk, cx, cy + sy * a / 2, 0, t, a, 0.01)
for (xp, yp, w, h) in [(148, 698, 60, 200), (305, 795, 175, 60), (318, 468, 115, 55), (145, 915, 60, 140), (322, 955, 240, 65),
                       (330, 1042, 110, 55), (1697, 690, 60, 180), (1608, 925, 60, 130), (1705, 955, 55, 130),
                       (1527, 1040, 235, 65)]:
    bracket(xp, yp, w, h)
for k in range(28):                                                         # the dashed box around the IR front end
    t = k / 28; per = 2 * (110 + 190)
    d = t * per
    if d < 110: xp, yp, horiz = 482 + d, 547, True
    elif d < 300: xp, yp, horiz = 592, 547 + (d - 110), False
    elif d < 410: xp, yp, horiz = 592 - (d - 300), 737, True
    else: xp, yp, horiz = 482, 737 - (d - 410), False
    if k % 2 == 0:
        x, y = px(xp, yp); add_box(silk, x, y, 0, 0.6 if horiz else 0.15, 0.15 if horiz else 0.6, 0.01)
obj_from_bm(silk, 'silk_marks', M['silk'])

# ---------------------------------------------------------------- 7. passives, where the photo has them (groups, not a carpet)
bodies_c, bodies_r, caps = bmesh.new(), bmesh.new(), bmesh.new()
def passive(x, y, rot, size):
    Lp, Wd, H = (1.0, 0.5, 0.35) if size == '0402' else (0.6, 0.3, 0.3)
    e = Lp * 0.2; c, s = math.cos(rot), math.sin(rot)
    add_box(bodies_c if rnd.random() < 0.6 else bodies_r, x, y, 0, Lp - 2 * e, Wd, H, rot)
    for end in (-1, 1):
        u = end * (Lp / 2 - e / 2); add_box(caps, x + u * c, y + u * s, 0, e, Wd, H, rot)
# Each group: a photo-pixel box the photo shows parts in, and roughly how many. Parts sit on a loose grid inside it.
GROUPS = [
    (565, 365, 665, 470, 34), (590, 475, 650, 540, 10), (160, 590, 205, 790, 18), (225, 560, 330, 610, 14),
    (225, 650, 330, 700, 10), (300, 720, 345, 750, 4), (260, 838, 400, 905, 18), (180, 870, 215, 990, 10),
    (215, 975, 425, 1000, 22), (520, 545, 590, 590, 6), (520, 650, 590, 730, 10), (480, 560, 500, 720, 6),
    (1010, 655, 1060, 700, 6), (1270, 770, 1345, 830, 12), (1275, 470, 1320, 520, 6), (1365, 615, 1408, 750, 12),
    (1535, 615, 1598, 750, 14), (1450, 545, 1515, 615, 6), (1625, 560, 1665, 790, 16), (1430, 970, 1640, 1012, 26),
    (1545, 870, 1590, 960, 10), (1645, 880, 1690, 1010, 12), (1680, 470, 1705, 560, 6), (965, 430, 975, 600, 4),
    (205, 720, 300, 760, 8), (1180, 300, 1240, 340, 4),
]
placed = 0
for (x0p, y0p, x1p, y1p, n) in GROUPS:
    (ax, ay), (bx, by) = px(x0p, y0p), px(x1p, y1p)
    x0, x1, y0, y1 = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
    rot = 0 if (x1 - x0) >= (y1 - y0) else math.pi / 2
    tries = 0; got = 0
    while got < n and tries < n * 40:
        tries += 1
        gx, gy = (1.1, 0.75) if rot == 0 else (0.75, 1.1)                    # snap to a grid, so rows line up like a real layout
        x = x0 + round((rnd.uniform(0, x1 - x0)) / gx) * gx; y = y0 + round((rnd.uniform(0, y1 - y0)) / gy) * gy
        if not (inside(x, y, 0.6) and free(x, y, 0.05)): continue
        r = rot if rnd.random() < 0.8 else rot + math.pi / 2
        passive(x, y, r, '0402' if rnd.random() < 0.3 else '0201'); reserve(x, y, 1.0, 1.0, pad=0.02); got += 1
    placed += got
obj_from_bm(bodies_c, 'passives_caps', M['mlcc']); obj_from_bm(bodies_r, 'passives_res', M['res']); obj_from_bm(caps, 'passive_ends', M['silver'])

# vias: a few, tented (under the mask), near the groups
via = bmesh.new(); n_via = 0
for (x0p, y0p, x1p, y1p, n) in GROUPS:
    for k in range(n // 3):
        x, y = px(rnd.uniform(x0p - 40, x1p + 40), rnd.uniform(y0p - 40, y1p + 40))
        if inside(x, y, 0.8) and free(x, y, 0.1): add_cyl(via, x, y, 0, 0.15, 0.012, seg=10); reserve(x, y, 0.3, 0.3); n_via += 1
obj_from_bm(via, 'vias', M['trace'])

# ================================================================ SIDE 2 (the underside), 16 Nov 2026
# Reference: iFixit's step-3 photo of side 2 (the board turned over like a page: the tab stays at the top, left and right swap).
# Measured in that photo's own pixels (a 800 x 500 view of it: the board spans x 100-688, y 108-397), then mirrored in x, so a part
# lands under the board where it really is: px2(u, v) = (-(u - 394) * k, (252 - v) * k), k = 100 mm / 588 px.
# Parts hang DOWN from the board's bottom face (z = -T). Text on this side is turned half a turn about y, so it reads the right way
# once the page turns the board over (journey.js: the flip group, a half turn about the same axis).
# Parts named by iFixit's chip ID (step 3): SK hynix HN8T05DEHKX073 128 GB storage; Qualcomm PMXR2230B, PMXR2230, PMXR2230VE,
# PMXR2230VS power management; a likely Qualcomm clock generator; STMicroelectronics STM32G071EB microcontroller.
K2 = 100 / 588
def px2(u, v): return (-(u - 394) * K2, (252 - v) * K2)
B = -T                                                                      # the bottom face
def bbox2(bm, u0, v0, u1, v1, h, z_from=None):
    """A box under the board, given by its corners in the side-2 photo; h mm deep (downwards)."""
    (ax, ay), (bx, by) = px2(u0, v0), px2(u1, v1); z0 = (B if z_from is None else z_from) - h
    add_box(bm, (ax + bx) / 2, (ay + by) / 2, z0, abs(bx - ax), abs(by - ay), h)
def text2(s_, u, v, h_mm, depth, kind='mark', rot_deg=0, align='CENTER'):
    cu = bpy.data.curves.new('t2', 'FONT'); cu.body = s_; cu.size = h_mm; cu.align_x = align; cu.align_y = 'CENTER'; cu.space_line = 1.05
    o = bpy.data.objects.new('t2_' + s_.split('\n')[0], cu); scene.collection.objects.link(o)
    x, y = px2(u, v); o.location = (x, y, B - depth - 0.004); o.rotation_euler = (0, math.pi, math.radians(rot_deg))
    text_objs[kind].append(o)
def anchor2(name, u, v, depth):
    x, y = px2(u, v)
    e = bpy.data.objects.new(name, None); scene.collection.objects.link(e); e.parent = root
    e.location = (x, y, B - depth); e.empty_display_size = 2; anchors[name] = e

chips2, sh2, ind2, lab2, qr2, pads2 = bmesh.new(), bmesh.new(), bmesh.new(), bmesh.new(), bmesh.new(), bmesh.new()
# three tin shield frames (lids off), like side 1's: a flange on a short wall
def frame2(o, i, h=0.75, t=0.15):
    (ox0, oy1), (ox1, oy0) = px2(o[2], o[1]), px2(o[0], o[3])               # x mirrored: the photo's right edge is the board's left
    (ix0, iy1), (ix1, iy0) = px2(i[2], i[1]), px2(i[0], i[3])
    for (a_, b_, c_, d_) in [(ox0, iy1, ox1, oy1), (ox0, oy0, ox1, iy0), (ox0, iy0, ix0, iy1), (ix1, iy0, ox1, iy1)]:
        add_box(sh2, (a_ + c_) / 2, (b_ + d_) / 2, B - h, c_ - a_, d_ - b_, t)
    for (a_, b_, c_, d_) in [(ox0, oy1 - t, ox1, oy1), (ox0, oy0, ox1, oy0 + t), (ox0, oy0, ox0 + t, oy1), (ox1 - t, oy0, ox1, oy1)]:
        add_box(sh2, (a_ + c_) / 2, (b_ + d_) / 2, B - h + t, c_ - a_, d_ - b_, h - t)
frame2((133, 212, 208, 281), (140, 219, 201, 274))
frame2((228, 172, 292, 315), (235, 179, 285, 308))
frame2((370, 145, 428, 290), (377, 152, 421, 283))
# the power chips and the clock generator, inside the frames
bbox2(chips2, 145, 224, 196, 270, 0.8); text2('PMXR2230B\n001', 170, 247, 0.55, 0.8, rot_deg=0)
bbox2(chips2, 243, 186, 276, 216, 0.7); text2('PMXR2230\nVE', 259, 201, 0.4, 0.7)
bbox2(chips2, 248, 250, 282, 290, 0.7); text2('PMXR2230', 265, 270, 0.4, 0.7)
bbox2(chips2, 385, 157, 415, 180, 0.7); text2('PMXR2230\nVS', 400, 168, 0.36, 0.7)
bbox2(chips2, 385, 190, 415, 213, 0.6)
bbox2(chips2, 390, 225, 410, 245, 0.5)                                       # the likely clock generator
bbox2(chips2, 385, 255, 415, 278, 0.6)
# SK hynix 128 GB storage (UFS), marking sideways as in the photo
bbox2(chips2, 462, 150, 525, 222, 1.0)
text2('SK hynix\nHN8T05DEHK\nX073   2SGY', 497, 186, 0.62, 1.0, rot_deg=90)
# STMicroelectronics STM32G071EB: a small package right of the storage chip
bbox2(chips2, 543, 260, 557, 274, 0.55)
# other small ICs the photo shows
for (u0, v0, u1, v1) in [(600, 255, 615, 270), (628, 225, 645, 240), (610, 330, 628, 345), (160, 310, 178, 327), (190, 340, 205, 355),
                         (335, 265, 350, 280), (500, 240, 512, 252), (636, 175, 652, 190)]:
    bbox2(chips2, u0, v0, u1, v1, 0.5)
# the grey power inductor "H330"
bbox2(ind2, 563, 188, 598, 226, 2.0); text2('H330', 580, 207, 1.0, 2.0, kind='ink', rot_deg=90)
# the small QR label and the assembly marking
bbox2(lab2, 437, 273, 451, 287, 0.05)
for i in range(12):
    for j in range(12):
        if i == 0 or j == 0 or rnd.random() < 0.5:
            u = 438 + i * 1.08; v = 274 + j * 1.08; x, y = px2(u + .54, v + .54); add_box(qr2, x, y, B - 0.06, K2 * 1.05, K2 * 1.05, 0.008)
text2('01', 490, 118, 1.2, 0, kind='silk')
text2('MLB 8349-PNx\n2323 0090441\nMS 13-02', 482, 276, 0.55, 0, kind='silk', align='LEFT')
# passives under the board, in the groups the photo shows
caps2, ends2, res2 = bmesh.new(), bmesh.new(), bmesh.new()
def passive2(x, y, rot, size):
    Lp, Wd, H = (1.0, 0.5, 0.35) if size == '0402' else (0.6, 0.3, 0.3)
    e = Lp * 0.2; c, s_ = math.cos(rot), math.sin(rot)
    add_box(caps2 if rnd.random() < 0.6 else res2, x, y, B - H, Lp - 2 * e, Wd, H, rot)
    for end in (-1, 1):
        u = end * (Lp / 2 - e / 2); add_box(ends2, x + u * c, y + u * s_, B - H, e, Wd, H, rot)
keep2 = []
def free2(x, y): return all(not (a_ < x < c_ and b_ < y < d_) for a_, b_, c_, d_ in keep2)
for (u0, v0, u1, v1) in [(133, 212, 208, 281), (228, 172, 292, 315), (370, 145, 428, 290), (462, 150, 525, 222), (563, 188, 598, 226),
                         (543, 260, 557, 274), (437, 273, 451, 287)]:
    (ax, ay), (bx, by) = px2(u0, v0), px2(u1, v1); keep2.append((min(ax, bx) - .3, min(ay, by) - .3, max(ax, bx) + .3, max(ay, by) + .3))
for (u0, v0, u1, v1, n) in [(300, 180, 365, 232, 50), (110, 300, 240, 390, 26), (525, 160, 560, 182, 8), (525, 240, 680, 380, 55),
                            (600, 165, 680, 215, 20), (430, 160, 460, 280, 16), (300, 240, 365, 300, 18), (210, 180, 228, 300, 10)]:
    got = 0; tries = 0
    while got < n and tries < n * 40:
        tries += 1
        u = u0 + rnd.uniform(0, u1 - u0); v = v0 + rnd.uniform(0, v1 - v0); x, y = px2(u, v)
        x = round(x / 1.1) * 1.1; y = round(y / 0.75) * 0.75
        if not (inside(x, y, 0.6) and free2(x, y)): continue
        passive2(x, y, 0 if rnd.random() < 0.7 else math.pi / 2, '0402' if rnd.random() < 0.3 else '0201'); keep2.append((x - .55, y - .4, x + .55, y + .4)); got += 1
# rows of gold test pads (side 2 has many: the factory's probe points)
for (u0, v0, u1, v1, n) in [(295, 235, 370, 300, 40), (515, 280, 600, 330, 30), (520, 330, 570, 385, 22), (140, 290, 230, 330, 20),
                            (430, 240, 460, 270, 10), (305, 150, 365, 175, 14)]:
    for k in range(n):
        u = u0 + round(rnd.uniform(0, u1 - u0) / 4) * 4; v = v0 + round(rnd.uniform(0, v1 - v0) / 4) * 4
        x, y = px2(u, v)
        if inside(x, y, 0.6) and free2(x, y): add_cyl(pads2, x, y, B - 0.03, 0.32, 0.03, seg=12)   # on the board, off the parts
# plated rings round the through-holes on this side too (the same holes as side 1)
rings2 = bmesh.new()
for (xp, yp, r_px, hole_px) in [(507, 318, 25, 13), (1278, 318, 25, 13), (1663, 540, 26, 14), (237, 880, 26, 14), (862, 825, 26, 14), (1610, 812, 26, 14)]:
    x, y = px(xp, yp); add_ring(rings2, x, y, B - 0.05, L(r_px), L(hole_px), 0.05)
x, y = px(293, 550); add_ring(rings2, x, y, B - 0.05, 1.95, 1.15, 0.05, seg=40)
anchor2('anchor_mcu', 550, 267, 0.55); anchor2('anchor_nand', 493, 186, 1.0); anchor2('anchor_pmic', 170, 247, 0.8)
obj_from_bm(chips2, 'chips_side2', M['epoxy']); obj_from_bm(sh2, 'shield_frames_side2', M['tin']); obj_from_bm(ind2, 'inductor_side2', M['can'])
obj_from_bm(lab2, 'label_side2', M['label']); obj_from_bm(qr2, 'code_side2', M['ink']); obj_from_bm(pads2, 'test_pads_side2', M['gold'])
obj_from_bm(caps2, 'passives_caps_side2', M['mlcc']); obj_from_bm(res2, 'passives_res_side2', M['res']); obj_from_bm(ends2, 'passive_ends_side2', M['silver'])
obj_from_bm(rings2, 'mount_rings_side2', M['gold'])

# ---------------------------------------------------------------- 8. text objects -> one mesh per material
for kind, mat_key in [('mark', 'mark'), ('silk', 'silk'), ('ink', 'ink')]:
    objs = text_objs[kind]
    if not objs: continue
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.convert(target='MESH'); bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active; o.name = 'text_' + kind; o.parent = root
    o.data.materials.clear(); o.data.materials.append(M[mat_key])

# ---------------------------------------------------------------- 9. scale to page units, save, export
root.scale = (0.1, 0.1, 0.1)
bpy.context.view_layer.update()
os.makedirs(ASSETS, exist_ok=True); os.makedirs(PREV, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, 'board.blend'))
bpy.ops.object.select_all(action='SELECT')
# Draco mesh compression (16 Nov, smaller files): quantized to 14 bits for positions (~0.006 mm on this 100 mm board), 10 for normals;
# the page decodes it with three.js's Draco decoder (hero_model.js makeGLTFLoader)
bpy.ops.export_scene.gltf(filepath=os.path.join(ASSETS, 'board_q3.glb'), export_format='GLB', use_selection=True,
                          export_apply=True, export_animations=False, export_yup=True,
                          export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=7,
                          export_draco_position_quantization=14, export_draco_normal_quantization=10)
tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in scene.objects if o.type == 'MESH')
print('BOARD', 'passives', placed, 'vias', n_via, 'tris', tris, 'glb', os.path.getsize(os.path.join(ASSETS, 'board_q3.glb')))

# ---------------------------------------------------------------- 10. previews
if RENDER:
    scene.render.engine = 'CYCLES'; scene.cycles.samples = 64; scene.cycles.device = 'CPU'
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1000
    world = bpy.data.worlds.new('w'); scene.world = world; world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.9, 0.9, 0.9, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.6
    def light(name, loc, energy, size=3.0, color=(1, 1, 1)):
        l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.size = size; l.color = color
        o = bpy.data.objects.new(name, l); scene.collection.objects.link(o); o.location = loc
        o.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    light('key', (-6, 4, 6), 120, 6); light('fill', (6, -4, 5), 60, 6)
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); scene.collection.objects.link(cam); scene.camera = cam
    # 'top': straight down, orthographic, framed like the reference photo, to compare part by part
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = 11.5; cam.location = (0, 0, 10); cam.rotation_euler = (0, 0, 0)
    scene.render.filepath = os.path.join(PREV, 'board_top.png'); bpy.ops.render.render(write_still=True)
    cam.data.type = 'PERSP'
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.004, 0.004, 0.005, 1)
    for name, loc, tgt, lens in [('wide', (2.5, -7.5, 6.5), (0, 0, 0), 50), ('soc', (1.9, -2.6, 2.4), (0.6, 0.6, 0), 75)]:
        cam.location = loc; cam.data.lens = lens
        cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = os.path.join(PREV, f'board_{name}.png'); bpy.ops.render.render(write_still=True)
    print('RENDERED')

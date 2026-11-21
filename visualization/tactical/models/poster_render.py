# poster_render.py: renders hero/poster.webp, the still picture the page shows when its live 3D cannot start (no WebGL, very short
# windows) and while TAKE CONTROL's head is loading. 16 Nov 2026: remade with the Quest 3 (it showed the old box headset).
#   /Applications/Blender.app/Contents/MacOS/Blender -b models/headset.blend --python-exit-code 1 -P models/poster_render.py
# Headless; the .blend is NOT saved (everything below changes only this run's copy).
#
# What it draws: the page's head (REF / ref_skin, at its fitted 0.96 size) wearing Elin's CC BY "Meta Quest 3" exactly as fitted
# (QUEST / quest3), with quest3_web's LOGO-FREE material, and the page's thin pointer from the visor's front. Colours are the page's
# (hero.js LOOK: skin #c8a590, cloth #474c57, pointer #dde4ee), the camera is the page's (30° to the player's left, 8° above),
# background the page's black. 1600 x 1000 WebP, like the old poster, so the page needs no other change.
import bpy, math, os, json
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'hero', 'poster.webp')
sc = bpy.context.scene

# textures: the .blend stores them relative to itself (//quest_tex/...); make sure they load
for im in bpy.data.images:
    if im.source == 'FILE' and not im.packed_file and im.filepath: im.reload()

# The bust. The .blend's ref_skin has the shoulders' points but only the head and neck faces (the clothing face list was never
# imported), so the bust is built here from the page's own head file, hero/head.js: face and neck (skin) and the clothing (i2),
# placed exactly where ref_skin sits (its fitted 0.96 size). Axes: the page head's (x, y, z) is Blender's (x, z, y).
H = json.loads(open(os.path.join(HERE, '..', 'hero', 'head.js')).read().split('=', 1)[1].strip().rstrip(';'))
sk = next(p for p in H['parts'] if p['group'] == 'skin'); u = H['unit']
verts = [(sk['p'][3 * k] * u, sk['p'][3 * k + 2] * u, sk['p'][3 * k + 1] * u) for k in range(len(sk['p']) // 3)]
tri = lambda ix: [tuple(ix[3 * k:3 * k + 3]) for k in range(len(ix) // 3)]
faces_skin, faces_cloth = tri(sk['i']), tri(sk['i2'])
me = bpy.data.meshes.new('poster_bust'); me.from_pydata(verts, [], faces_skin + faces_cloth); me.update()
for k, poly in enumerate(me.polygons): poly.material_index = 0 if k < len(faces_skin) else 1; poly.use_smooth = True
ref = bpy.data.objects['ref_skin']
bust = bpy.data.objects.new('poster_bust', me); sc.collection.objects.link(bust); bust.matrix_world = ref.matrix_world.copy()
# check: the same points as ref_skin (it was built from this file), else the bust would not sit under the headset
rv = ref.data.vertices
d = max((Vector(verts[k]) - rv[k].co).length for k in range(0, min(len(verts), len(rv)), 97)) if len(rv) == len(verts) else None
print('BUST CHECK', len(verts), len(rv), d)
if d is None or d > 1e-4:
    # different vertex order or a mirrored axis: try the other axis order before giving up
    raise SystemExit(f'bust does not match ref_skin (max point difference {d}); not rendering')

# only the bust and the Quest render
keep = {'poster_bust', 'quest3'}
for o in bpy.data.objects:
    if o.type in ('MESH', 'CURVE', 'FONT'): o.hide_render = o.name not in keep
for c in bpy.data.collections: c.hide_render = False
head, quest = bust, bpy.data.objects['quest3']
quest.data.materials.clear(); quest.data.materials.append(bpy.data.objects['quest3_web'].data.materials[0])   # logo-free

def srgb(h):
    h = h.lstrip('#'); c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c) + (1,)
def principled(name, color, rough, sheen=0.0, sheen_tint=None, emit=None):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = srgb(color); b.inputs['Roughness'].default_value = rough
    if sheen: b.inputs['Sheen Weight'].default_value = sheen; b.inputs['Sheen Tint'].default_value = srgb(sheen_tint)
    if emit: b.inputs['Emission Color'].default_value = srgb(emit); b.inputs['Emission Strength'].default_value = 3.0
    return m
skin = principled('poster_skin', '#c8a590', 0.58, 0.3, '#ff9d80')
cloth = principled('poster_cloth', '#474c57', 0.92, 0.5, '#8f97a6')
head.data.materials.clear(); head.data.materials.append(skin); head.data.materials.append(cloth)
n_skin = len(faces_skin)                                                 # slots set AFTER the materials exist (setting them first did not stick)
for k, poly in enumerate(head.data.polygons): poly.material_index = 0 if k < n_skin else 1
# the skin mesh has two material groups like the page's (face/neck, then the clothing): keep its existing slot indices
# (ref_skin was built with slot 0 = skin, slot 1 = cloth by tools/blender/build_head.py)

# the pointer: from the visor's front, 1.5 head-lengths forward (the face looks along -y here)
hb = [quest.matrix_world @ Vector(v) for v in quest.bound_box]
front_y = min(p.y for p in hb); mid_z = (min(p.z for p in hb) + max(p.z for p in hb)) / 2 - 0.02
sb = [head.matrix_world @ Vector(v) for v in head.bound_box]
head_len = 0.6
L = 1.5 * head_len
bpy.ops.mesh.primitive_cylinder_add(radius=0.0035, depth=L, location=(0, front_y - L / 2, mid_z), rotation=(math.pi / 2, 0, 0))
ptr = bpy.context.active_object; ptr.name = 'poster_pointer'
ptr.data.materials.append(principled('poster_pointer', '#b9cbe6', 0.4, emit='#b9cbe6'))

# camera: the page's, 30° to the player's left (+x when facing -y) and 8° above, looking at the headset
target = Vector((0, (front_y + 0.25) / 2, mid_z - 0.22))               # a little low, so the shoulders come into the frame
yaw, pitch, dist = math.radians(30), math.radians(8), 3.0
cam_pos = target + Vector((math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch))) * dist
cam = bpy.data.objects.new('poster_cam', bpy.data.cameras.new('poster_cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.location = cam_pos; cam.rotation_euler = (target - cam_pos).to_track_quat('-Z', 'Y').to_euler(); cam.data.lens = 50
cam.data.shift_x = -0.15                                                 # the head right of centre, the pointer reaching into the left (as before)

# lights: a warm key from the front left above, a cool rim from behind, a weak fill; the page's black world
for o in [o for o in bpy.data.objects if o.type == 'LIGHT']: o.hide_render = True
def area(name, loc, energy, size, color):
    l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.size = size; l.color = color
    o = bpy.data.objects.new(name, l); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (target - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
area('p_key', (1.8, -2.4, 1.8), 260, 2.2, (1, .96, .9)); area('p_rim', (-1.6, 1.8, 1.2), 220, 1.6, (.82, .88, 1))
area('p_fill', (-2.2, -1.6, 0.2), 60, 3.0, (1, 1, 1))
w = bpy.data.worlds.new('poster_world'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = srgb('#010101'); w.node_tree.nodes['Background'].inputs['Strength'].default_value = 1

r = sc.render; r.engine = 'CYCLES'; sc.cycles.samples = 96; sc.cycles.device = 'CPU'; sc.cycles.use_denoising = True
r.resolution_x, r.resolution_y, r.resolution_percentage = 1600, 1000, 100
sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'None'
r.image_settings.file_format = 'WEBP'; r.image_settings.quality = 82; r.image_settings.color_mode = 'RGB'
r.filepath = OUT
bpy.ops.render.render(write_still=True)
print('POSTER', OUT, os.path.getsize(OUT))

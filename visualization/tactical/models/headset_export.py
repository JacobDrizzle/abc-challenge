# headset_export.py: writes assets/headset_q3.glb from models/headset.blend (the page's headset: Elin's CC BY "Meta Quest 3",
# logo removed, see CREDITS.md). Headless, and it does not save the .blend:
#   /Applications/Blender.app/Contents/MacOS/Blender -b models/headset.blend --python-exit-code 1 -P models/headset_export.py
#
# What it exports: the FULL original mesh (object `quest3`, 76,260 faces, its own custom normals), not the decimated `quest3_web`,
# whose 35% decimation smeared the shading. It is moved into the head's rest frame the way quest3_web was: the head was fitted to the
# headset at scale 0.96 and y −0.129, so the headset gets the inverse (scale 1/0.96, then +0.129/0.96 in y). Material: quest3_web's
# (the logo-free textures in quest_tex/).
# Size (16 Nov 2026): Draco mesh compression, JPEG textures. Was 3.7 MB with plain buffers.
import bpy, mathutils, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'assets', 'headset_q3.glb')

# The .blend stores the textures by ABSOLUTE path, from before the folder moved (16 Nov: nic_scratch -> ~/Dev/projects/ABC-Challenge),
# so they load empty ("has no size") and the export silently drops them. Point each one at quest_tex/ beside this script and reload;
# the *_1k copies are the 1024 px versions the page uses (they were scaled down in Blender), so they are scaled again here.
# (Only for this run: the .blend is not saved.)
TEX = os.path.join(HERE, 'quest_tex')
for im in bpy.data.images:
    if im.source != 'FILE' or im.packed_file or not im.filepath: continue
    name = os.path.basename(bpy.path.abspath(im.filepath))
    if os.path.exists(os.path.join(TEX, name)):
        im.filepath = os.path.join(TEX, name); im.reload()
        if im.name.endswith('_1k') or '_1k.' in im.name: im.scale(1024, 1024)
missing = [im.name for im in bpy.data.images if im.source == 'FILE' and not im.packed_file and im.filepath and tuple(im.size) == (0, 0)]
if missing: raise SystemExit('textures still missing: ' + ', '.join(missing))

q = bpy.data.objects['quest3']; w = bpy.data.objects['quest3_web']
me = q.data.copy(); me.name = 'quest3_web_full'
o = bpy.data.objects.new('quest3_web_full', me)
bpy.context.scene.collection.objects.link(o)
M = mathutils.Matrix.Translation((0, 0.129 / 0.96, 0)) @ mathutils.Matrix.Scale(1 / 0.96, 4)
me.transform(M @ q.matrix_world); o.matrix_world = mathutils.Matrix.Identity(4)
me.materials.clear(); me.materials.append(w.data.materials[0])
for x in bpy.context.view_layer.objects: x.select_set(False)
o.select_set(True); bpy.context.view_layer.objects.active = o
bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', use_selection=True, export_apply=True, export_animations=False, export_yup=True,
                          export_image_format='JPEG', export_image_quality=85,   # (WEBP wrote no images when run headless, Blender 5.2: JPEG)
                          # 14-bit positions (~0.04 mm at headset size), 10-bit normals, 12-bit UVs (a quarter pixel on the 1024 px textures)
                          export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=7,
                          export_draco_position_quantization=14, export_draco_normal_quantization=10, export_draco_texcoord_quantization=12)
print('EXPORTED', os.path.getsize(OUT), len(me.polygons))

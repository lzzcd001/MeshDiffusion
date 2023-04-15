import sys
BLENDER_PATH = './' # change this to your path to “path/to/BlenderToolbox/
sys.path.append(BLENDER_PATH)
import BlenderToolBox as bt
import os, bpy, bmesh
import glob
import numpy as np
cwd = os.getcwd()


import mathutils


mesh_folder_path = 'PLACEHOLDER'
output_path = 'PLACEHOLDER'
os.makedirs(output_path, exist_ok=True)

def readOBJ(filePath, location, rotation_euler, scale):
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)

	prev = []
	for ii in range(len(list(bpy.data.objects))):
		prev.append(bpy.data.objects[ii].name)
	bpy.ops.import_scene.obj(filepath=filePath, split_mode='OFF')
	after = []
	for ii in range(len(list(bpy.data.objects))):
		after.append(bpy.data.objects[ii].name)
	name = list(set(after) - set(prev))[0]
	mesh = bpy.data.objects[name]

	mesh.location = location
	mesh.rotation_euler = angle
	mesh.scale = scale
	bpy.context.view_layer.update()

	return mesh

blend_path = './scene.blend'
with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
	data_to.materials = data_from.materials
bpy.ops.wm.open_mainfile(filepath=blend_path)

# Set the device_type
bpy.context.preferences.addons[
            "cycles"
            ].preferences.compute_device_type = "CUDA" # or "OPENCL"

# Set the device and feature set
bpy.context.scene.cycles.device = "GPU"

# get_devices() to let Blender detects GPU device
bpy.context.preferences.addons["cycles"].preferences.get_devices()
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d["use"] = 1 # Using all devices, include GPU and CPU

## set shading (uncomment one of them)
bpy.ops.object.shade_smooth() # Option1: Gouraud shading
# bpy.ops.object.shade_flat() # Option2: Flat shading
# bt.edgeNormals(mesh, angle = 10) # Option3: Edge normal shading

## set light
## Option1: Three Point Light System 
# bt.setLight_threePoints(radius=4, height=10, intensity=1700, softness=6, keyLoc='left')
## Option2: simple sun light
lightAngle = (6, -30, -155) 
strength = 2
shadowSoftness = 0.10

sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

## set ambient light
bt.setLight_ambient(color=(0.2, 0.2, 0.2, 1)) 


path_list = sorted(glob.glob(os.path.join(mesh_folder_path, '*.obj')))

for k, meshPath in enumerate(path_list):
	out_file_path = os.path.join(output_path, '{:06d}.png'.format(k))
	if os.path.exists(out_file_path):
		continue

	imgRes_x = 720 # recommend > 1080 
	imgRes_y = 720 # recommend > 1080 
	numSamples = 10
	exposure = 1.5
	bpy.context.scene.render.film_transparent = True
	bpy.context.scene.cycles.samples = numSamples
	# bpy.context.scene.cycles.max_bounces = 6
	bpy.context.scene.cycles.max_bounces = 24
	bpy.context.scene.cycles.film_exposure = exposure

	## read mesh
	bpy.ops.object.select_all(action='DESELECT')
	location = (1.0, 2.0, 1.6) # (GUI: click mesh > Transform > Location)
	# rotation = (90, 0, 270) # (GUI: click mesh > Transform > Rotation) ## for car
	rotation = (90, 0, 180) # (GUI: click mesh > Transform > Rotation)
	# scale = (7.,7.,7.) # (GUI: click mesh > Transform > Scale) ## for car
	scale = (3.,3.,3.) # (GUI: click mesh > Transform > Scale)
	mesh = readOBJ(meshPath, location, rotation, scale)

	## subdivision
	bt.subdivision(mesh, level = 1)

	mesh.active_material = bpy.data.materials.get("cbrewer medium red")

	minz = 999999.0

	for vertex in mesh.data.vertices:
		# object vertices are in object space, translate to world space
		v_world = mesh.matrix_world @ mathutils.Vector((vertex.co[0],vertex.co[1],vertex.co[2]))

		if v_world[2] < minz:
			minz = v_world[2]

	mesh.location.z = mesh.location.z - minz

	bpy.data.scenes['Scene'].render.filepath = out_file_path
	bpy.ops.render.render(write_still = True)
	

	# Remember which meshes were just imported
	meshes_to_remove = []
	for ob in bpy.context.selected_objects:
		meshes_to_remove.append(ob.data)

	bpy.ops.object.delete()

	# Remove the meshes from memory too
	for mesh in meshes_to_remove:
		bpy.data.meshes.remove(mesh)

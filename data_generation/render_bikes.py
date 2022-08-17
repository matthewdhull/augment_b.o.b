import bpy
import numpy as np
import math
import os
from math import radians

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])    

def rotate(point, angle_degrees, axis=(0,1,0)):
    theta_degrees = angle_degrees
    theta_radians = math.radians(theta_degrees)
    
    rotated_point = np.dot(rotation_matrix(axis, theta_radians), point)
    return rotated_point

bpy.ops.mesh.primitive_cube_add(size=10, location=(0, 0, 1.5))
cube = bpy.context.active_object

path_to_file = '/scenes/roadBike/roadBike.obj'
imported_bike = bpy.ops.import_scene.obj(filepath = path_to_file)
bike = bpy.context.selected_objects[0] 
print(f'imported {bike.name}')
# transform origin of bike so that camera tracks properly
bpy.context.scene.tool_settings.use_transform_data_origin = True
bpy.ops.transform.translate(value=(0, 0, 0.539356) \
    , orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)) \
    , orient_matrix_type='GLOBAL', constraint_axis=(False, False, True) \
    , mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH' \
    , proportional_size=1, use_proportional_connected=False \
    , use_proportional_projected=False)
bpy.context.scene.tool_settings.use_transform_data_origin = False



bpy.ops.mesh.primitive_plane_add(size=50)
plane = bpy.context.active_object

light_data = bpy.data.lights.new('light', type='AREA')
light = bpy.data.objects.new('light', light_data)
bpy.context.collection.objects.link(light)
light.location = (1.0785, -0.025681, 1.8664)
light.rotation_euler = (0, radians(48.6), 0)
light.scale = (15.0, 15.0, 15.0)
light.data.energy = 35.0

# create camera
cam_data = bpy.data.cameras.new('camera')
cam = bpy.data.objects.new('camera', cam_data)
cam.location = (3.76, 0.008329, 0.794164)

# camera will point to bike regardless of position
constraint = cam.constraints.new(type='TRACK_TO')
constraint.target = bike
bpy.context.collection.objects.link(cam)

#create material
mat = bpy.data.materials.new(name='Material')
mat.use_nodes = True
mat_nodes = mat.node_tree.nodes
mat_links = mat.node_tree.links

# metallic
mat_nodes['Principled BSDF'].inputs['Metallic'].default_value=1.0
mat_nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.371, 0.371, 0.371, 1.0)
mat_nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.167

#create material
mat = bpy.data.materials.new(name='Material')
mat.use_nodes = True
mat_nodes = mat.node_tree.nodes
mat_links = mat.node_tree.links

plane.data.materials.append(mat)

scene = bpy.context.scene
scene.camera  = cam

# Use 224x224 res to match dimensions of bird-or-bicycle dataset
scene.render.resolution_y = 224
scene.render.resolution_x = 224
scene.render.image_settings.file_format = 'PNG'

# get bike materials
bike_materials = bike.data.materials.keys()
# get bike frame color (RGB)
bike_frame_mat  = bike.data.materials[bike.data.materials.keys()[0]]
bike_frame_color_rgba = ( \
    bike_frame_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[0] \
    , bike_frame_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[1] \
    , bike_frame_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[2] \
    , bike_frame_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[3])

# get bike handlebar material
bike_handlebar_mat = bike.data.materials[bike.data.materials.keys()[8]]
bike_handlebar_color_rgba = ( \
    bike_handlebar_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[0] \
    , bike_handlebar_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[1] \
    , bike_handlebar_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[2] \
    , bike_handlebar_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[3])

# get bike seat material
bike_seat_mat = bike.data.materials[bike.data.materials.keys()[7]]
bike_seat_color_rgba = ( \
    bike_seat_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[0] \
    , bike_seat_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[1] \
    , bike_seat_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[2] \
    , bike_seat_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value[3])
    
    
print(f'original bike frame color: {bike_frame_color_rgba}')

# set new colors
black = (0.0, 0.0, 0.0, 1.0)


# render black bike at various angles
for angle in range(0, 180, 15):    
    cam_location = cam.location    
    cam.location = rotate(cam_location, 15, axis=(0, 0, 1))    
    
    scene.render.filepath=f'/Users/matthewhull/Downloads/blender_{angle}_black.png'
    # change color and render
    bike_frame_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = black
    bike_handlebar_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = black   
    bike_seat_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = black
    bpy.ops.render.render(write_still=1)
    
#    # set original color and render
    bike_frame_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = bike_frame_color_rgba
    bike_handlebar_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = bike_handlebar_color_rgba   
    bike_seat_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value_value = bike_seat_color_rgba 
    scene.render.filepath=f'/Users/matthewhull/Downloads/blender_{angle}.png'
    bpy.ops.render.render(write_still=1)

"""
    Rendering script
    Gabriel Moreira
    Aug 25 2023
    
    Run from terminal instructions (Important: Blender == 3.0)
    Run desired render
        >./blender/blender -b -noaudio /mnt/localdisk/gabriel/nodocker/smartshop/noise_modeling.blend --python /mnt/localdisk/gabriel/nodocker/smartshop/noise_modeling.py
"""
import os
import bpy
import json
import math
import numpy as np
import mathutils as mu
import random
from time import time
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view
from typing import Iterable, Callable


def clear_scene():
    """
        Delete all objects in the scene
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=True)
    bpy.ops.object.select_all(action='DESELECT')


def clear_cameras():
    """
        Delete all cameras
    """
    # Remove camera objects
    objs = [ob for ob in bpy.context.scene.objects if ob.type == "CAMERA"]
    bpy.ops.object.delete({"selected_objects": objs})
    # Also remove camera data
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    # Also remove camera collection if exists
    coll = bpy.data.collections.get("cameras")
    if coll is not None:
        bpy.data.collections.remove(coll)


def load_aruco(texture_path: str,
               size: float,
               specular: float=0.08,
               roughness: float=0.75,
               metallic: float=0.0,
               sheen: float=0.0):
    """
        Add an Aruco cube
    """
    # Add cube
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 1.0))

    box_obj = bpy.context.selected_objects[0]
    box_obj.name = 'aruco'

    box_material_obj = bpy.data.materials.new("aruco" + '-Material')
    box_material_obj.use_nodes = True

    # Add cube texture (aruco markers)
    bsdf         = box_material_obj.node_tree.nodes["Principled BSDF"]
    tex_im       = box_material_obj.node_tree.nodes.new('ShaderNodeTexImage')
    tex_im.image = bpy.data.images.load(texture_path)

    box_material_obj.node_tree.links.new(bsdf.inputs['Base Color'], tex_im.outputs['Color'])
    box_obj.data.materials.append(box_material_obj)
    
    bpy.data.materials["aruco-Material"].node_tree.nodes["Principled BSDF"].inputs[6].default_value = metallic
    bpy.data.materials["aruco-Material"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = specular
    bpy.data.materials["aruco-Material"].node_tree.nodes["Principled BSDF"].inputs[12].default_value = sheen
    bpy.data.materials["aruco-Material"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = roughness
    bpy.data.materials["aruco-Material"].node_tree.nodes["Image Texture"].interpolation = "Closest"
    

def clear_aruco():
    """
        Delete aruco marker be obbject
    """
    objs = [ob for ob in bpy.context.scene.objects if ob.name == "aruco"]
    bpy.ops.object.delete({"selected_objects": objs})
    for obj in bpy.data.objects:
        if obj.name == "aruco":
            bpy.data.objects.remove(obj)
            
    # Remove materials as well
    for material in bpy.data.materials:
        if material.name.split('-')[0] == "aruco":
            bpy.data.materials.remove(material, do_unlink=True)

    
def move_obj(obj_name: str,
             location: mu.Vector,
             euler_angles: mu.Vector):

    # Select object to move by name
    obj = bpy.data.objects[obj_name]
    
    obj.location = location
    obj.rotation_euler = mu.Euler(euler_angles, 'XYZ')
    bpy.ops.object.select_all(action='DESELECT')


def location_in_view(cam, location):
    """
        Check object center is in-view from camera
    """
    scene = bpy.context.scene
    
    cs, ce = cam.data.clip_start, cam.data.clip_end
    co_ndc = world_to_camera_view(scene, cam, location)
    
    if (0.0 < co_ndc.x < 1.0 and
        0.0 < co_ndc.y < 1.0 and
        cs < co_ndc.z <  ce):
            return True
    else:
        return False

    
def intersect(obj1, obj2):
    """
        Check if obj1's mesh intersects obj2's mesh
    """
    # Get their world mat
    mat1 = obj1.matrix_world
    mat2 = obj2.matrix_world

    # Get the geometry in world coordinates
    vert1 = [mat1 @ v.co for v in obj1.data.vertices] 
    poly1 = [p.vertices for p in obj1.data.polygons]

    vert2 = [mat2 @ v.co for v in obj2.data.vertices] 
    poly2 = [p.vertices for p in obj2.data.polygons]

    # Create the BVH trees
    bvh1 = BVHTree.FromPolygons(vert1, poly1)
    bvh2 = BVHTree.FromPolygons(vert2, poly2)

    # Test if overlap
    if bvh2.overlap(bvh1):
        return True
    else:
        return False


def intersects_anything(obj):
    """
        Check if obj's mesh intersects anything
    """
    furniture_coll = bpy.data.collections["furniture"] 
    building_coll  = bpy.data.collections["building"] 
    
    for obj2 in furniture_coll.all_objects:
        if intersect(obj, obj2):
            print("Intersection with furniture obj: {}".format(obj2.name))
            return True
    for obj2 in building_coll.all_objects:
        if intersect(obj, obj2):
            print("Intersection with building obj: {}".format(obj2.name))
            return True
    return False


def small_room_random_pose_gen():
    """
        Generate valid poses inside small room volume
    """
    print("Trying random pose...")
    valid = False
    while (not valid):
        location = mu.Vector((np.random.uniform(-4.04, 4.05), 
                              np.random.uniform(-3.65, 3.55),
                              np.random.uniform(0, 1.8)))
                                
        euler_angles = mu.Euler((np.random.rand()*2*np.pi,
                                np.random.rand()*2*np.pi,
                                np.random.rand()*2*np.pi), "XYZ")
        
        if location[0] >= -4.04 and location[0] <= 4.05 and location[1] >= -3.65 and location[1] <= 3.55:
            valid = True

    return location, euler_angles
             

def get_sensor_size(sensor_fit: str,
                    sensor_x: float,
                    sensor_y: float) -> float:
    """
    """
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit: str,
                   size_x: float,
                   size_y: float) -> str:
    """
    """
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def get_camera_intrinsics(scene, cam) -> dict:
    """
    """
    if cam.data.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')

    f_in_mm = cam.data.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(cam.data.sensor_fit, cam.data.sensor_width, cam.data.sensor_height)
    sensor_fit = get_sensor_fit(cam.data.sensor_fit,
                                scene.render.pixel_aspect_x * resolution_x_in_px,
                                scene.render.pixel_aspect_y * resolution_y_in_px)
    
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px

    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px

    s_u = 1.0 / pixel_size_mm_per_px
    s_v = 1.0 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2.0 - cam.data.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2.0 + cam.data.shift_y * view_fac_in_px / pixel_aspect_ratio

    cam = {'fx'           : s_u,
           'fy'           : s_v,
           'cx'           : u_0,
           'cy'           : v_0,
           'resolution_x' : resolution_x_in_px,
           'resolution_y' : resolution_y_in_px,
           'clip_start'   : cam.data.clip_start,
           'clip_end'     : cam.data.clip_end,
           't'            : np.array(cam.location).tolist(),
           'R'            : (np.array(cam.rotation_quaternion.to_matrix()) @ np.diag((1,-1,-1))).tolist(),
           'distortion'   : np.zeros(12).tolist()}
    return cam
    

    
def render(output_path: str,
           cam_name: str,
           num_frames: int,
           distance_cutoff: float):
    """
        Render marker images
    """
    bpy.context.scene.render.image_settings.file_format = "JPEG"
    
    # Fetch camera objects by name
    cam    = bpy.data.objects[cam_name]
    cam_id = cam.name.split('_')[-1]
    scene  = bpy.data.scenes[0]

    cams = {cam_id : get_camera_intrinsics(scene, cam)}
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        
    # Save selected camera pose
    cameras_path = os.path.join(output_path, "cameras.json")
    with open(cameras_path, 'w') as f:
        json.dump(cams, f)
    print("Camera dictionary saved to {}".format(cameras_path))

    # Check if dictionary with aruco_cube poses already exists
    aruco_filename = "aruco_pose.json"
    aruco_pose = {}
    if aruco_filename in os.listdir(render_path):
        with open(os.path.join(render_path, aruco_filename)) as f:
            aruco_pose = json.load(f)
        print("Loaded JSON file with marker poses from {}".format(os.path.join(render_path, aruco_filename)))
            
    # Select rendering camera
    bpy.context.scene.camera = cam
    for t in range(num_frames):
        os.mkdir(os.path.join(output_path, str(t)))
        
        valid = False
        while not valid:
            location, euler_angles = small_room_random_pose_gen()
        
            move_obj(obj_name="aruco",
                     location=location,
                     euler_angles=euler_angles)
                       
            visible = location_in_view(cam, location)
            intersecting = intersects_anything(bpy.data.objects["aruco"])
            faraway = math.dist(cam.location, location) > distance_cutoff
            valid = visible and not intersecting and not faraway      
                                        
        im_path = os.path.join(output_path, "{}/{}.jpg".format(t, cam_id))
        bpy.context.scene.render.filepath = im_path
        print("Rendering frame {}".format(t))
        bpy.ops.render.render(write_still=True)
        
        # Update dictionary and save
        aruco_pose[t] = {'t' : np.array(location).tolist(),
                         'R' : np.array(euler_angles.to_matrix()).tolist()}
        
        with open(os.path.join(render_path, aruco_filename), 'w') as f:
            json.dump(aruco_pose, f)

    
if __name__ == "__main__" :
    np.random.seed(0)

    root               = "/mnt/localdisk/gabriel/nodocker/smartshop/"
    render_path        = os.path.join(root, "noise_modeling_render")
    aruco_texture_path = os.path.join(root, "marker_0_texture.png")
    
    clear_aruco()
    load_aruco(texture_path=aruco_texture_path,
               size=0.2875*48/50)
    
    # Force rendering to GPU
    bpy.context.scene.cycles.device = 'GPU'
    cpref = bpy.context.preferences.addons['cycles'].preferences
    cpref.compute_device_type = 'CUDA'
    cpref.get_devices()
    for device in cpref.devices:
        device.use = True if device.type == 'CUDA' else False

    # RENDER SETTINGS
    for scene in bpy.data.scenes:
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 720
        scene.render.resolution_percentage = 100
        scene.render.use_border = False
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 1024
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        # IF USING EEVEE UNCOMMENT BELOW 
        #scene.render.engine = EEVEE'
        #scene.eevee.taa_render_samples = 64  
        #scene.eevee.bokeh_overblur     = 1.9
        #scene.eevee.bokeh_denoise_fac  = 0.5
        #scene.eevee.bokeh_threshold    = 6
        #scene.eevee.bokeh_max_size     = 190
        
        
    # LOAD ARUCO CUBE
    render(output_path=render_path,
           cam_name="camera_7",
           num_frames=2000,
           distance_cutoff=6.0)
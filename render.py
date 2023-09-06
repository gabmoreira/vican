"""
    Rendering script
    Gabriel Moreira
    Aug 25 2023
    
    Run from terminal instructions (Important: Blender == 3.0)
    Start an X11 server in one shell
        >sudo Xorg :1
    On a different shell
        >export DISPLAY=:1
    Run desired render
        >./blender/blender -b -noaudio /mnt/localdisk/gabriel/nodocker/smartshop/large_shop.blend --python /mnt/localdisk/gabriel/nodocker/smartshop/render.py
        >./blender/blender -b -noaudio /mnt/localdisk/gabriel/nodocker/smartshop/small_room.blend --python /mnt/localdisk/gabriel/nodocker/smartshop/render.py
    Get the Xorg PID
        >nvidia-smi 
    Kill the server
        >sudo kill -9 [PID]
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

    
def save_cameras(path: str):
    """
        Saves all cameras to a JSON dictionary
        Call before starting renders to make sure
        actual camera parameters are recorded!
        Parameters stored use OpenCV conventions.
    """
    objs  = [ob for ob in bpy.context.scene.objects if ob.type == "CAMERA"]
    
    scene = bpy.data.scenes[0]
    
    cams = {}
    for obj in objs:
        cam_id = obj.name.split('_')[-1]
        cams[cam_id] = {cam_id : get_camera_intrinsics(scene, obj)}
    
    with open(path, 'w') as f:
        json.dump(cams, f)
    print("Cameras' dictionaries saved to {}".format(path))


def load_aruco(texture_path,
               specular=0.08,
               roughness=0.75,
               metalic=0.0,
               sheen=0.0):
    """
        Add an Aruco cube
    """
    # Add cube
    bpy.ops.mesh.primitive_cube_add(size=2, location=(1,1,1), scale=(.575/2, .575/2, .575/2))
    box_obj = bpy.context.selected_objects[0]
    box_obj.name = 'aruco_cube'

    box_material_obj = bpy.data.materials.new("aruco_cube" + '-Material')
    box_material_obj.use_nodes = True

    # Add cube texture (aruco markers)
    bsdf         = box_material_obj.node_tree.nodes["Principled BSDF"]
    tex_im       = box_material_obj.node_tree.nodes.new('ShaderNodeTexImage')
    tex_im.image = bpy.data.images.load(texture_path)

    box_material_obj.node_tree.links.new(bsdf.inputs['Base Color'], tex_im.outputs['Color'])
    box_obj.data.materials.append(box_material_obj)
    
    for node in bpy.data.materials["aruco_cube-Material"].node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            for input in node.inputs:
                if input.name == "Metallic":
                    input.default_value = metalic
                if input.name == "Specular":
                    input.default_value = specular
                if input.name == "Sheen":
                    input.default_value = sheen
                if input.name == "Roughness":
                    input.default_value = roughness

    bpy.data.materials["aruco_cube-Material"].node_tree.nodes["Image Texture"].interpolation = "Closest"
    

def clear_aruco_cube():
    """
        Delete aruco cube obbject
    """
    objs = [ob for ob in bpy.context.scene.objects if ob.name == "aruco_cube"]
    bpy.ops.object.delete({"selected_objects": objs})
    for obj in bpy.data.objects:
        if obj.name == "aruco_cube":
            bpy.data.objects.remove(obj)
            
    # Remove materials as well
    for material in bpy.data.materials:
        if material.name.split('-')[0] == "aruco_cube":
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
        
        if location[0] >= -4.04 and location[0] <= 4.05 and \
            location[1] >= -3.65 and location[1] <= 3.55:
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


def large_shop_random_pose_gen():
    """
        Generate valid poses inside large_shop volume
    """
    print("Trying random pose...")
    valid = False
    while(not valid):
        location = mu.Vector((np.random.uniform(0, 22), 
                              np.random.uniform(0, 25),
                              np.random.uniform(0, 1.8)))
                                
        euler_angles = mu.Euler((np.random.rand()*2*np.pi,
                                 np.random.rand()*2*np.pi,
                                 np.random.rand()*2*np.pi), "XYZ")
        
        if (location[1] >= 0 and location[1] <= 4.7) and location[0] >= 8:
            valid = False
        elif (location[1] > 4.7 and location[1] <= 13) and (location[0] >= 7):
            valid = False
        elif (location[1] > 13 and location[1] <= 18.7) and (location[0] >= 14):
            valid = False
        elif (location[1] >= 12 and location[1] <= 18.6) and location[0] <= 3.3:
            valid = False
        else:
            valid =  True

    return location, euler_angles

    
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



def aruco_cube_pose_candidate(pose_generator: Callable,
                              distance_cutoff: float=7.0):
    """
        Generate a candidate pose for aruco_cube
    """
    
    cams = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
    while True:
        location, euler_angles = pose_generator()

        # Move aruco_cube to the new pose
        move_obj(obj_name="aruco_cube",
                 location=location,
                 euler_angles=euler_angles)
        visible_cam_ids = []
        if not intersects_anything(bpy.data.objects["aruco_cube"]):
            for cam in cams:
                if math.dist(cam.location, location) <= distance_cutoff:
                    if location_in_view(cam, location):
                        visible_cam_ids.append(cam.name)
        if len(visible_cam_ids) > 1:
            print("Successful! Visible cameras:")
            print(visible_cam_ids)
            return visible_cam_ids, location, euler_angles
                
              
def render(output_path: str,
           cam_ids: Iterable[str]):
    """
        Render image from specified cameras
    """
    bpy.context.scene.render.image_settings.file_format = "JPEG"
    # Fetch camera objects by name
    cameras = [bpy.data.objects[id] for id in cam_ids]
    
    for i, cam in enumerate(cameras):
        cam_id = cam.name.split('_')[-1]
        bpy.context.scene.camera = cam
        filepath = os.path.join(output_path, "{}.jpg".format(cam_id))
        bpy.context.scene.render.filepath = filepath
        print("Rendering camera {}".format(cam_id))
        bpy.ops.render.render(write_still=True)
    

    
def render_cube_calib(output_path: str,
                      cam_name: str,
                      num_frames: int):
    """
        Render image from specified cameras
    """
    bpy.context.scene.render.image_settings.file_format = "JPEG"
    
    # Fetch camera objects by name
    cam = bpy.data.objects[cam_name]
    scene  = bpy.data.scenes[0]

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        
    # Pretend there are num_frames cameras and the cube is fixed
    cams = {str(t) : get_camera_intrinsics(scene, cam) for t in range(num_frames)}

    # Save selected camera pose
    cameras_path = os.path.join(output_path, "cameras.json")
    with open(cameras_path, 'w') as f:
        json.dump(cams, f)
    print("Camera dictionary saved to {}".format(cameras_path))

    bpy.context.scene.camera = cam
    for t in range(num_frames):
        os.mkdir(os.path.join(output_path, str(t)))
        
        move_obj(obj_name="aruco_cube",
                 location=mu.Vector((np.random.uniform(5.6, 6.2),
                                     np.random.uniform(20.7, 22),
                                     np.random.uniform(0.15, 0.8))),
                 euler_angles=mu.Euler((np.random.rand()*2*np.pi,
                                        np.random.rand()*2*np.pi,
                                        np.random.rand()*2*np.pi), "XYZ"))
                                        
        im_path = os.path.join(output_path, "{}/{}.jpg".format(t,t))
        bpy.context.scene.render.filepath = im_path
        print("Rendering frame {}".format(t))
        bpy.ops.render.render(write_still=True)
        


    
if __name__ == "__main__" :
    root               = "/mnt/localdisk/gabriel/nodocker/smartshop"
    # Where all camera images and metadata should go
    render_path        = os.path.join(root, "large_shop_render")
    # Where data for cube calibration should go
    cube_calib_path    = os.path.join(root, os.path.join(root, "cube_calib_render"))
    aruco_texture_path = os.path.join(root, "aruco_texture.png")

    # force rendering to GPU
    bpy.context.scene.cycles.device = 'GPU'
    cpref = bpy.context.preferences.addons['cycles'].preferences
    cpref.compute_device_type = 'CUDA'
    # Use GPU devices only
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
        # Higher value = better quality but slower
        scene.cycles.samples = 1000
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
    clear_aruco_cube()
    load_aruco(texture_path=aruco_texture_path)
 

    # GENERATE RENDERS FOR CUBE CALIBRATION    
    render_cube_calib(output_path=cube_calib_path,
                      cam_name="camera_471",
                      num_frames=1000)
    """
        
    # GENERATE RENDERS FOR CAMERA CALIBRATION
    num_timesteps = 5000
    
    if not os.path.isdir(render_path):
        os.mkdir(render_path)
        print("Created directory {}".format(render_path))
    else:
        print("Directory {} already exists".format(render_path))
    
    # Save cameras to make sure we know where images came from
    save_cameras(os.path.join(render_path, "cameras.json"))
    
    # When lauching several processes to prevent name collision
    num_cores = 8
    core_id   = 7 # 0...num_cores-1 (always zero if only launching one process)
    offset    = 0
    np.random.seed(core_id + int(time()))
        
    for i in range(num_timesteps):
        # Next timestep
        t = i * num_cores + core_id + offset

        # Generate new aruco_marker pose 
        print("\nCurrent time instant t={}".format(t))
        
        # EDIT FUNCTION TO GENERATE ARUCO POSES 
        visible_cam_ids, location, euler_angles = aruco_cube_pose_candidate(large_shop_random_pose_gen)

        # Check if dictionary with aruco_cube poses already exists
        dict_name = "aruco_cube_pose_" + str(core_id) + ".json"
        aruco_cube_pose = {}
        if dict_name in os.listdir(render_path):
            with open(os.path.join(render_path, dict_name)) as f:
                aruco_cube_pose = json.load(f)
            print("Loaded JSON file with aruco_marker poses")

        # Update dictionary and save
        aruco_cube_pose[t] = {'t' : np.array(location).tolist(),
                              'R' : np.array(euler_angles.to_matrix()).tolist()}
        
        with open(os.path.join(render_path, dict_name), 'w') as f:
            json.dump(aruco_cube_pose, f)

        # Create timestamp directory
        os.mkdir(os.path.join(render_path, str(t)))  
        # Render
        render(output_path=os.path.join(render_path, str(t)),
               cam_ids=visible_cam_ids)
        """
    
                    
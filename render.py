"""
    Rendering script
    Gabriel Moreira
    Jun 5 2023
    
    Instructions:
    
    Run from terminal:
    >./blender/blender ~/work/gabriel/maks_shop.blend --background --python ~/work/gabriel/render.py
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
from typing import Iterable



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

        
        
def load_cameras(extrinsics_path: str, 
                 focal_length: float):
    """
        Load cameras to blender scene
        Opens a JSON dictionary with 
        keys corresponding to camera IDs (string)
        and values corresponding to SE(3) pose List[List[]]
    """
    # Create new collection
    cam_coll = bpy.data.collections.new("cameras")
    
    # Add collection to scene collection
    bpy.context.scene.collection.children.link(cam_coll)
    
    # Load JSON file with camera poses
    with open(gt_extrinsics_path, 'r') as f:
        extrinsics = json.load(f)
        
    # Add new cameras
    for cam_id in extrinsics.keys():
        pose = extrinsics[cam_id]
        
        t = mu.Vector([pose[0][3], pose[1][3], pose[2][3]])
        
        R = mu.Matrix([[pose[0][0], -pose[0][1], -pose[0][2]],
                       [pose[1][0], -pose[1][1], -pose[1][2]],
                       [pose[2][0], -pose[2][1], -pose[2][2]]])
                         
        q = R.to_quaternion()
        
        cam = bpy.data.cameras.new(name=f"camera_{cam_id}")
        cam.lens       = focal_length
        cam.clip_start = 0.1
        cam.clip_end   = 1000.0

        cam_obj = bpy.data.objects.new(name=f"camera_{cam_id}", object_data=cam)
        cam_obj.rotation_mode = 'QUATERNION'
        cam_obj.location = t
        cam_obj.rotation_quaternion = q

        cam_coll.objects.link(cam_obj)



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



def aruco_cube_pose_candidate():
    """
        Generate a candidate pose for aruco_cube
    """
    distance_cutoff = 7.0
    
    def within_bounds(location):
        if (location[1] >= 0 and location[1] <= 4.7) and location[0] >= 8:
            return False
        if (location[1] > 4.7 and location[1] <= 13) and (location[0] >= 7):
            return False
        if (location[1] > 13 and location[1] <= 18.7) and (location[0] >= 14):
            return False
        if (location[1] >= 12 and location[1] <= 18.6) and location[0] <= 3.3:
            return False
        return True
    
    cams = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
    while True:
        print("Trying random pose...")
        location = mu.Vector((np.random.uniform(0, 22), 
                              np.random.uniform(0, 25),
                              np.random.uniform(0, 1.8)))
                              
        euler_angles = mu.Euler((np.random.rand()*2*np.pi,
                                 np.random.rand()*2*np.pi,
                                 np.random.rand()*2*np.pi), "XYZ")
        
        # Check if proposal location is within bounds
        if within_bounds(location):
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
    camera = bpy.data.objects[cam_name]
    cam_id = cam_name.split('_')[-1]
    bpy.context.scene.camera = camera
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        
    for frame_id in range(num_frames):
        os.mkdir(os.path.join(output_path, str(frame_id)))
        
        move_obj(obj_name="aruco_cube",
                 location=mu.Vector((np.random.uniform(5.6, 6.2),
                                     np.random.uniform(20.7, 22),
                                     np.random.uniform(0.15, 0.8))),
                 euler_angles=mu.Euler((np.random.rand()*2*np.pi,
                                        np.random.rand()*2*np.pi,
                                        np.random.rand()*2*np.pi), "XYZ"))
                                        
        im_path = os.path.join(output_path, "{}/{}.jpg".format(frame_id,frame_id))
        bpy.context.scene.render.filepath = im_path
        print("Rendering frame {}".format(frame_id))
        bpy.ops.render.render(write_still=True)
        
        
    
def save_cameras(path: str):
    """
        Saves all cameras to a JSON dictionary
        Call before starting renders to make sure
        actual camera parameters are recorded!
        Parameters stored use OpenCV conventions.
    """
    objs  = [ob for ob in bpy.context.scene.objects if ob.type == "CAMERA"]
    
    scene = bpy.data.scenes[0]
    resolution_x = scene.render.resolution_x
    resolution_y = scene.render.resolution_y
    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    
    cams = {}
    for obj in objs:
        cam_id = obj.name.split('_')[-1]
        cams[cam_id] = {'fx'           : obj.data.lens / obj.data.sensor_width * resolution_x,
                        'fy'           : (obj.data.lens / obj.data.sensor_width * resolution_x) * pixel_aspect,
                        'cx'           : resolution_x * (0.5 - obj.data.shift_x),
                        'cy'           : resolution_y * 0.5 + resolution_x * obj.data.shift_y,
                        'resolution_x' : resolution_x,
                        'resolution_y' : resolution_y,
                        'clip_start'   : obj.data.clip_start,
                        'clip_end'     : obj.data.clip_end,
                        't'            : np.array(obj.location).tolist(),
                        'R'            : (np.array(obj.rotation_quaternion.to_matrix()) @ np.diag((1,-1,-1))).tolist(),
                        'distortion'   : np.zeros(12).tolist()}
    
    with open(path, 'w') as f:
        json.dump(cams, f)
    print("Cameras' dictionaries saved to {}".format(path))



    
if __name__ == "__main__" :
    # PATHS
    root               = "/home/jovyan/work/gabriel"
    render_path        = os.path.join(root, "render")
    cube_calib_path    = os.path.join(root, os.path.join(root, "cube_calib_im"))
    aruco_texture_path = os.path.join(root, "aruco_texture.png")

    # RENDERING SETTINGS
    for scene in bpy.data.scenes:
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 720
        scene.render.resolution_percentage = 100
        scene.render.use_border = False
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = 5    # Higher value = better quality but slower
        scene.eevee.bokeh_overblur     = 1.9
        scene.eevee.bokeh_denoise_fac  = 0.5
        scene.eevee.bokeh_threshold    = 6
        scene.eevee.bokeh_max_size     = 190
        
        
    # LOAD ARUCO CUBE
    clear_aruco_cube()
    load_aruco(texture_path=aruco_texture_path)

    
    """
    # LOAD CAMERAS (.blend FILE ALREADY CONTAINS CAMERAS!)
    clear_cameras()
    load_cameras(extrinsics_path=os.path.join(root, "cameras_extrinsics.json"),
                 focal_length=29)
    """

        
    """
    # GENERATE RENDERS FOR CUBE CALIBRATION    
    render_cube_calib(output_path=cube_calib_path,
                      cam_name="camera_471",
                      num_frames=1000)
    """
        
        
    # GENERATE RENDERS FOR CAMERA CALIBRATION
    
    print("\nStarting renders...")
    num_timesteps = 900
    
    if not os.path.isdir(render_path):
        os.mkdir(render_path)
        print("Created directory {}".format(render_path))
    else:
        print("Directory {} already exists".format(render_path))
    
    # Save cameras to make sure we know where images came from
    save_cameras(os.path.join(render_path, "cameras.json"))
    
    # When lauching several processes to prevent name collision
    num_cores = 6
    core_id   = 5 # 0...num_cores-1 (always zero if only launching one process)
    
    np.random.seed(core_id + int(time()))
        
    for i in range(num_timesteps):
        # Next timestep
        t = 0 + i * num_cores + core_id

        # Generate new aruco_marker pose 
        print("\nCurrent time instant t={}".format(t))
        print("Generating candidate aruco_cube poses")
        visible_cam_ids, location, euler_angles = aruco_cube_pose_candidate()

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
    
                    
import pickle
import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils import path
from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import VariableTopologyMeshes as VTMeshes
from aitviewer.renderables.point_clouds import PointClouds as VTPoints
from collections import defaultdict
from global_vars import RESULTS_DIR, SCANDATASET_DIR

cmap = plt.get_cmap('gist_rainbow')

def pickle_load(file):
    """
    Load a pickle file.
    """
    with open(file, 'rb') as f:
        loadout = pickle.load(f)

    return loadout

def pickle_save(path, obj):
    """
    Save a pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def quads2tris(F):
	F_out = []
	for f in F:
		if len(f) <= 3: F_out += [f]
		elif len(f) == 4:
			F_out += [
				[f[0], f[1], f[2]],
				[f[0], f[2], f[3]]
			]
	return np.array(F_out, np.int32)

def read_obj(filename):
    # Read the OBJ file and store vertices and faces in a dictionary
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
                
            if parts[0] == 'v':
                vertex = tuple(map(float, parts[1:]))
                vertices.append(vertex)
            elif parts[0] == 'f':
                face = tuple(map(int, [p.split('/')[0] for p in parts[1:]]))
                faces.append(face)

    vertices = np.array(vertices)
    faces = np.array(faces) - 1

    obj_data = {'vertices': vertices, 'faces': faces}
    return obj_data

def adjust_color(num):
    """
    Adjusts color to look nice.
    :param color: [..., 4] RGBA color
    :return: updated color
    """
    color = np.array(cmap(num))
    color[..., :3] /= color[..., :3].max(axis=-1, keepdims=True)
    color[..., :3] *= 0.3
    color[..., :3] += 0.3
    return tuple(color)

def visualize(viewer, results_folder, pos_index):
    results_folder, sequence = results_folder.split('/')[-2:]
    smpl = 'smpl' in results_folder
    type = results_folder.split("_")[-1]

    ########### load body ###########
    body_vertices = np.load(os.path.join(RESULTS_DIR, results_folder, sequence, 'body.npy'))
    transl = body_vertices[0].mean(0)[None,None]
    body_vertices -= transl
    body_obj = read_obj(os.path.join(SCANDATASET_DIR, results_folder, 'body.obj'))
    body_faces = body_obj['faces']

    cmap = plt.get_cmap('gist_rainbow')
    body_frames = {'vertices':[], 'faces':[], "colors":[]}
    for i in range(len(body_vertices)):
        # if i == 0: body_vertices[i] = body_obj['vertices']
        body_frames['vertices'].append(body_vertices[i])
        body_frames['faces'].append(body_faces)
        # colors = np.array([cmap(1.)]*len(body_vertices[0]))
        # body_frames['colors'].append(colors)


    ########### load garment ###########
    garment_vertices = np.load(os.path.join(RESULTS_DIR, results_folder, sequence, f'{type}.npy'))
    garment_vertices -= transl
    garment_obj = read_obj(os.path.join(SCANDATASET_DIR, results_folder, f'{type}.obj'))
    garment_faces = quads2tris(garment_obj['faces'])

    garment_frames = {'vertices':[], 'faces':[], "colors":[]}
    for i in range(len(garment_vertices)):
        # if i == 0: garment_vertices[i] = garment_obj['vertices']
        garment_frames['vertices'].append(garment_vertices[i])
        garment_frames['faces'].append(garment_faces)
    garment_frames['colors'] = adjust_color(0.15*(pos_index+1))



    # garment
    garment_mesh = VTMeshes(garment_frames['vertices'], garment_frames['faces'], color=garment_frames['colors'],
                            position=np.array([-1.5 * pos_index, 2., 0.]), name=results_folder)
    garment_mesh.backface_culling = False
    # garment_point = VTPoints(garment_frames['vertices'], garment_frames['colors'], name='garment')

    # body
    body_mesh = VTMeshes(body_frames['vertices'], body_frames['faces'], 
                         position=np.array([-1.5 * pos_index, 2., 0.]), name=results_folder)
    body_mesh.backface_culling = False
    # body_point = VTPoints(body_frames['vertices'], body_frames['colors'], name='body')

    # set camera position
    positions, targets = path.lock_to_node(garment_mesh, [0, 0, 3])
    camera = PinholeCamera(positions, targets, viewer.window_size[0], viewer.window_size[1], viewer=viewer)
    viewer.scene.add(garment_mesh)
    # viewer.scene.add(garment_point)
    viewer.scene.add(body_mesh)
    # viewer.scene.add(body_point)

    return camera


def main():
    name = "smpl_tshirt"
    result_folders = [x[0] for x in os.walk(os.path.join(RESULTS_DIR, name))][1:]
    viewer = Viewer()
    viewer.scene.nodes = viewer.scene.nodes[:5]

    for i, folder in enumerate(result_folders):
        camera = visualize(viewer, folder, i)

    viewer.scene.add(camera)
    viewer.set_temp_camera(camera)

    viewer.playback_fps = 30
    viewer.run()

if __name__ == "__main__":
    main()
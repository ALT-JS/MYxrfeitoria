from pathlib import Path
import json
import numpy as np
import itertools
import xrfeitoria.camera.camera_parameter
from scipy.spatial.transform import Rotation

import xrfeitoria as xf
from xrfeitoria.rpc import remote_blender
from xrfeitoria.data_structure.models import RenderPass
# from xrfeitoria.data_structure.models import SequenceTransformKey as SeqTransKey

# from .. import config
from config import assets_path, blender_exec
from utils import setup_logger, visualize_vertices
import bpy_types

root = Path(__file__).parents[2].resolve()
# output_path = '~/xrfeitoria/output/samples/blender/{file_name}'
output_path = root / 'output' / Path(__file__).relative_to(root).with_suffix('')
log_path = output_path / 'blender.log'
output_blend_file = output_path / 'source.blend'

seq_1_name = 'NeuralDomeOutput'
# seq_2_name = 'Testoutput2'

def rgba2list(r, g, b, a):
    return [r/255, g/255, b/255, a/255]
@remote_blender()
def import_obj(path: str, col) -> str:
    import bpy

    bpy.ops.wm.obj_import(filepath=path)
    mesh = bpy.context.selectable_objects[-1]
    if mesh.name.startswith('human'):
        mod = mesh.modifiers.new("Subsurf", "SUBSURF")
        mod.levels = 3
        mod.subdivision_type = "CATMULL_CLARK"
    # new material
    mat = bpy.data.materials.new(name="Material")
    mat.use_nodes = True

    tree = mat.node_tree
    # node_vertex = tree.nodes.new(type='ShaderNodeVertexColor')
    # tree.nodes['Principled BSDF'].inputs['Subsurface'].default_value = 0.5
    tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = col
    # tree.nodes['Principled BSDF'].inputs['Subsurface Color'].default_value = col
    # link vertex color to material
    # tree.links.new(node_vertex.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

    # assign material to object
    mesh.data.materials.append(mat)

    return mesh.name

@remote_blender()
def setLightforShadow(name, loc, reu):
    import bpy
    light = bpy.data.lights.new(name, type='POINT')
    light.energy = 1000
    light.color = (1, 1, 1)
    light.shadow_soft_size = 0.5
    light_obj = bpy.data.objects.new(name='light_shadow', object_data=light)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = reu
    light_obj.location = loc
    return

def setFloorHeight(h):
    with open(assets_path['floor'], 'w') as Hf:
        Hf.write(f'v -100 -100 {h}\n') # modified to be larger
        Hf.write(f'v 100 100 {h}\n')
        Hf.write(f'v 100 -100 {h}\n')
        Hf.write(f'v -100 100 {h}\n')
        Hf.write(f'f 2 3 4\n')
        Hf.write(f'f 1 4 3\n')
    return

def main(debug=False, background=False):
    logger = setup_logger(debug=debug, log_path=log_path)

    #############################
    #### use default level ######
    #############################
    setFloorHeight(0.08)
    # open blender and automatically create a default level named `XRFeitoria`
    xf_runner = xf.init_blender(exec_path=blender_exec, background=background, new_process=True)

    # set hdr map
    xf_runner.utils.set_env_color((0.8, 0.8, 0.9, 1.0), 1)
    
    
    RANGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 19, 23, 24, 25, 26, 27, 30, 31, 33, 35, 36, 37, 38]

    # import actors to the level
    human_mesh = {}; actor_human = {}; object_mesh = {}; actor_object = {}
    for i in RANGES:
        human_mesh['human_mesh'+str(i)] = import_obj(path=assets_path[f'human{i}'], col=rgba2list(50, 128, 181, 255))
        actor_human['actor_human'+str(i)] = xf_runner.Actor(human_mesh['human_mesh'+str(i)])
    for i in RANGES:
        object_mesh['object_mesh'+str(i)] = import_obj(path=assets_path[f'object{i}'], col=rgba2list(83, 25, 103, 255))
        actor_object['actor_object'+str(i)] = xf_runner.Actor(object_mesh['object_mesh'+str(i)]) # ori 90, 52, 103
    
    floor_mesh = import_obj(path=assets_path['floor'], col=rgba2list(125, 70, 59, 255))
    actor_floor = xf_runner.Actor(floor_mesh)
    # xf_runner.Actor.import_from_file()

    # ALL modified
    HORT = {
        "Set1" : [(-0.016295, -4.3631, 1.9678), (0., 0., 0.)],
        "Set2" : [(-0.021059, -3.3731, 2.2155), (0., 0., 0.)],
        "Set3" : [(-0.013314, -2.1238, 2.5876), (0., 0., 0.)],
        "Set4" : [(-0.002444, -1.4976, 4.0067), (0., 0., 0.)],
        "Set5" : [(0.000863, -2.5669, 3.7195), (0., 0., 0.)],
        "Set6" : [(0.006615, -3.6312, 3.4345), (0., 0., 0.)],
        # baseball
        "Set7" : [[1.0837, -3.8817, -1.1616], (72.562, 10.972, -3.8321)],
        "Set8" : [[0.66361, -1.7631, -1.8373], (50.827, 0., 0.)],
        "Set9" : [[0.63723, 0.41182, -2.9955], (10.475, 0., 0.)],
        "Set10" : [[0.66791, 0.79145, -1.6155], (0., 0., 0.)],
        "Set11" : [[1.3725, -0.68303, -4.311], (-44.638, 11.793, -40.254)],
        "Set12" : [[1.2216, -3.2245, -5.0351], (-59.438, 0.72746, -38.549)],
        # skateboard
        "Set15" : [[-0.069023, 2.5786, 1.5337], (-90.0, -1.6026, 0.)],
        "Set16" : [[0.041376, -1.3725, 3.2109], (0., 0., 0.)],
        # suitcase
        "Set17" : [(0.69605, -2.1809, 2.781), (-1.2292, 0.18067, 17.256)],
        "Set19" : [[0.58709, -0.6711, 3.3216], (-46.859, 14.207, 2.0161)],
        # dumbbell
        "Set23" : [[0.82112, 4.0387, -4.2377], (-58.051, 84.731, 28.744)],
        "Set24" : [[0.24686, -1.4611, -1.931], (-180.31, -13.104, -92.213)],
        # chair
        "Set25" : [[0.32376, -1.1683, 2.231], (-1.5377, 0.13927, 10.819)],
        "Set26" : [[0.59409, -1.4199, 3.4428], (-2.2931, 0.36105, 16.293)],
        # kettlebell
        "Set27" : [[0.53136, 0.27165, 1.4486], (-0.078802, -51.173, 86.759)],
        "Set30" : [[0.26398, 1.398, 2.7374], (-5.2588, -24.576, 89.84)],
        # pan
        "Set31" : [[0.18396, 1.449, 1.3238], (9.4174, -1.1119, 6.458)],
        "Set33" : [[0.60574, 0.74009, 1.3387], (22.073, -6.045, 13.88)],
        # tennis
        "Set35" : [[0.5484, 6.0901, 1.3396], (-6.4033, -2.5698, 82.322)],
        "Set36" : [[0.58429, 5.6472, 2.8154], (6.2454, -0.72565, 93.201)],
        # broom
        "Set37" : [[0.9382, -0.23794, 4.5301], (170.98, 2.0303, 11.366)],
        "Set38" : [[0.63055, 0.40996, 3.9191], (163.07, 1.8364, 5.8645)]
        # golf
    }
    
    
    # Human&Object
    for i in RANGES:
        actor_human[f"actor_human{i}"].set_transform(HORT[f"Set{i}"][0], HORT[f"Set{i}"][1], (1.0, 1.0, 1.0))
    for i in RANGES:
        actor_object[f"actor_object{i}"].set_transform(HORT[f"Set{i}"][0], HORT[f"Set{i}"][1], (1.0, 1.0, 1.0))

        
    
    actor_floor.set_transform((2.48, -0.34, -3.29), (-90., 111.57, -90.), (1.0, 1.0, 1.0))
    setLightforShadow("light1", [-1.9954, -2.1747, -0.77373], [0, 1, 0])
    setLightforShadow("light2", [-1.9954, 1.923, 0.47519], [0, 1, 0])

    # add a new camera looking at the bunny, and the distance between them is 2m
    with open('../camera/camera_zhaochf_lefthand_swing1.txt', 'r') as cameraF:
        camera_data = json.load(cameraF)
    with open('../camera/intrinsic.txt', 'r') as intriF:
        K_data = json.load(intriF)
    R = np.array(camera_data['rot']).reshape(3, 3)
    T = np.array(camera_data['transl']).reshape(3, 1)
    # R = np.eye(3)
    K = np.array(K_data['K']).reshape(3, 3)
    RT = np.hstack((R, T.reshape(-1, 1)))

    RT = np.vstack((RT, np.array([0, 0, 0, 1])))
    RT = np.linalg.inv(RT)
    RT[..., 1] *= -1
    RT[..., 2] *= -1
    location = (RT[0][3], RT[1][3], RT[2][3])
    rot = np.linalg.inv(R)
    rot[..., 1] *= -1
    rot[..., 2] *= -1
    # 获取欧拉角
    rotation_euler = Rotation.from_matrix(rot).as_euler('xyz', degrees=True)
    rotation_euler = rotation_euler.tolist()
    location = list(location)
    location = list(map(float, location))
    rotation_euler = list(map(float, rotation_euler))
    camera = xf_runner.Camera.spawn(
        location=[-1.7, -1.3724, 4.0483],
        rotation=[-34.5, -18.442, 88.189]
    )
    # # set camera's fov to make the bunny fill the 10% of screen
    # fov = np.rad2deg(2 * np.arctan(3840 / (float(2 * K[0][0]))))
    camera.fov = float(90.0)
    ##################################################
    #### Create a new sequence with default level ####
    ##################################################
    with xf_runner.Sequence.new(seq_name=seq_1_name) as seq:
        # use the `camera` in level to render
        seq.use_camera(camera=camera)

        # add render job to renderer
        seq.add_to_renderer(
            output_path=output_path / seq_1_name,
            render_passes=[
                RenderPass('img', 'png'),
                # RenderPass('mask', 'png'),
                # RenderPass('normal', 'png')
            ],
            resolution=[3840, 2160],
            render_engine='CYCLES',
            render_samples=32,
            transparent_background=True,
            arrange_file_structure=True,
        )

    logger.info('🎉 [bold green]Success!')
    input('Press Any Key to Exit...')

    # render
    xf_runner.render()

    # Save the blender file to the current directory.
    xf_runner.utils.save_blend(save_path=output_blend_file)
    
    # collect the names for visualization
    camera_name = camera.name
    # actor_names = [actor_human.name]

    # close the blender process
    xf_runner.close()

    seq1_out = output_path / seq_1_name / 'img' / camera_name / '0000.png'
    logger.info(f'Check the output of seq_1 in "{seq1_out.as_posix()}"')
main(debug=False, background=False)

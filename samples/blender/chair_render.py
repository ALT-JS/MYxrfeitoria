from pathlib import Path
import json
import numpy as np
from mathutils import Matrix
import xrfeitoria.camera.camera_parameter
from scipy.spatial.transform import Rotation

import xrfeitoria as xf
from xrfeitoria.rpc import remote_blender
from xrfeitoria.data_structure.models import RenderPass
from xrfeitoria.data_structure.models import SequenceTransformKey as SeqTransKey

# from .. import config
from config import assets_path, blender_exec
from utils import setup_logger, visualize_vertices
import bpy_types

root = Path(__file__).parents[2].resolve()
# output_path = '~/xrfeitoria/output/samples/blender/{file_name}'
output_path = root / 'output' / Path(__file__).relative_to(root).with_suffix('')
log_path = output_path / 'blender.log'
output_blend_file = output_path / 'source.blend'

seq_1_name = 'CHAIRSNewTestCam'
def rgba2list(r, g, b, a):
    return [r/255, g/255, b/255, a/255]
@remote_blender()
def import_obj(path: str, col) -> str:
    import bpy

    bpy.ops.wm.obj_import(filepath=path)
    mesh = bpy.context.selectable_objects[-1]
    if mesh.name == 'human':
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
def setLightforShadow(loc, reu):
    import bpy
    light = bpy.data.lights.new("light", type='POINT')
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
        Hf.write(f'v -6 -5 {h}\n')
        Hf.write(f'v 5 6 {h}\n')
        Hf.write(f'v 6 -5 {h}\n')
        Hf.write(f'v -5 6 {h}\n')
        Hf.write(f'f 2 3 4\n')
        Hf.write(f'f 1 4 3\n')
    return

def main(debug=False, background=False):
    logger = setup_logger(debug=debug, log_path=log_path)

    #############################
    #### use default level ######
    #############################
    setFloorHeight(-0.55)
    # open blender and automatically create a default level named `XRFeitoria`
    xf_runner = xf.init_blender(exec_path=blender_exec, background=background, new_process=True)

    # set hdr map
    # xf_runner.utils.set_hdr_map(hdr_map_path=assets_path['hdr_sample'])
    xf_runner.utils.set_env_color((0.8, 0.8, 0.9, 1.0), 1)

    # import an actor to the level
    human_mesh = import_obj(path="../input/testChairHuman.obj", col=rgba2list(70, 81, 105, 255))
    actor_human = xf_runner.Actor(human_mesh)
    object_mesh = import_obj(path="../input/testChairObject.obj", col=rgba2list(222, 116, 135, 255))
    actor_object = xf_runner.Actor(object_mesh)
    floor_mesh = import_obj(path=assets_path['floor'], col=rgba2list(125, 70, 59, 255))
    actor_floor = xf_runner.Actor(floor_mesh)
    # xf_runner.Actor.import_from_file()

    actor_human.set_transform((0., 0., 0.), (90, 0., 90), (1.0, 1.0, 1.0))
    actor_object.set_transform((0., 0., 0.), (90, 0., 90), (1.0, 1.0, 1.0))
    actor_floor.set_transform((0., 0., 0.), (0., 0., 0.), (1.0, 1.0, 1.0))
    setLightforShadow([-1.5, -2.2, 2.3], [0, 1, 0])

    with open('../camera/CHAIRS_extrinsic.txt', 'r') as cameraF:
        camera_data = json.load(cameraF)
    with open('../camera/CHAIRS_intrinsic.txt', 'r') as intriF:
        K_data = json.load(intriF)

    R = np.array(camera_data['rot']).reshape(3, 3)
    T = np.array(camera_data['transl']).reshape(3, 1)
    # R = np.eye(3)
    K = np.array(K_data['K']).reshape(3, 3)
    RT = np.hstack((R, T.reshape(-1, 1)))

    camera_matrix = Matrix(K)
    RT = np.vstack((RT, np.array([0, 0, 0, 1])))
    RT = np.linalg.inv(RT)
    RT[..., 1] *= -1
    RT[..., 2] *= -1
    transform_matrix = Matrix(RT)
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
        location=location,
        rotation=rotation_euler
    )
    # # set camera's fov to make the bunny fill the 10% of screen
    fov = np.rad2deg(2 * np.arctan(960 / (float(2 * K[0][0]))))
    camera.fov = float(fov)
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
            ],
            resolution=[960, 540],
            render_engine='CYCLES',
            render_samples=32,
            transparent_background=True,
            arrange_file_structure=True,
        )

    # render
    xf_runner.render()

    # Save the blender file to the current directory.
    xf_runner.utils.save_blend(save_path=output_blend_file)

    logger.info('🎉 [bold green]Success!')
    input('Press Any Key to Exit...')

    # collect the names for visualization
    camera_name = camera.name
    actor_names = [actor_human.name]

    # close the blender process
    xf_runner.close()

    seq1_out = output_path / seq_1_name / 'img' / camera_name / '0000.png'
    logger.info(f'Check the output of seq_1 in "{seq1_out.as_posix()}"')

# args = parse_args()

main(debug=False, background=False)

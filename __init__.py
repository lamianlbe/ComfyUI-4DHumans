import os

REPO_PATH = os.path.dirname(os.path.abspath(__file__))

from folder_paths import models_dir

SMPL_PATH = os.path.join(models_dir, 'smpl')
os.makedirs(SMPL_PATH, exist_ok=True)


from .nodes.process_humans_node import ProcessHumansNode
from .nodes.load_detectron_node import LoadDetectronNode
from .nodes.load_hmr_node import LoadHMRNode
from .nodes.pose_controlnet_node import HumanPoseControlNetNode
from .nodes.load_phalp_node import LoadPHALPNode
from .nodes.phalp_pose_node import PHALPPoseControlNetNode
from .nodes.load_smplestx_node import LoadSMPLestXNode
from .nodes.load_sapiens_node import LoadSapiensNode
# from .nodes.select_human_node import SelectHumanNode


NODE_CLASS_MAPPINGS = {
    'ProcessHumans': ProcessHumansNode,
    'LoadDetectron': LoadDetectronNode,
    'LoadHMR': LoadHMRNode,
    'HumanPoseControlNet': HumanPoseControlNetNode,
    'LoadPHALP': LoadPHALPNode,
    'PHALPPoseControlNet': PHALPPoseControlNetNode,
    'LoadSMPLestX': LoadSMPLestXNode,
    'LoadSapiens': LoadSapiensNode,
    # 'SelectHuman': SelectHumanNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ProcessHumans': 'Process 4D Humans',
    'LoadDetectron': 'Load Detectron Model',
    'LoadHMR': 'Load HMR Model',
    'HumanPoseControlNet': '4D Human Pose (ControlNet)',
    'LoadPHALP': 'Load PHALP',
    'PHALPPoseControlNet': '4D Human Pose Tracking (ControlNet)',
    'LoadSMPLestX': 'Load SMPLest-X',
    'LoadSapiens': 'Load Sapiens Pose',
    # 'SelectHuman': 'Select 4D Human'
}

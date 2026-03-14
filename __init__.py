import os

REPO_PATH = os.path.dirname(os.path.abspath(__file__))

from folder_paths import models_dir

SMPL_PATH = os.path.join(models_dir, 'smpl')
os.makedirs(SMPL_PATH, exist_ok=True)


from .nodes.load_phalp_node import LoadPHALPNode
from .nodes.phalp_pose_node import PHALPPoseControlNetNode
from .nodes.load_sapiens_node import LoadSapiensNode
from .nodes.sapiens_pose_node import SapiensPoseNode
from .nodes.load_prompthmr_node import LoadPromptHMRNode
from .nodes.prompthmr_pose_node import PromptHMRPoseNode


NODE_CLASS_MAPPINGS = {
    'LoadPHALP': LoadPHALPNode,
    'PHALPPoseControlNet': PHALPPoseControlNetNode,
    'LoadSapiens': LoadSapiensNode,
    'SapiensPose': SapiensPoseNode,
    'LoadPromptHMR': LoadPromptHMRNode,
    'PromptHMRPose': PromptHMRPoseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'LoadPHALP': 'Load PHALP',
    'PHALPPoseControlNet': '4D Human Pose Tracking (ControlNet)',
    'LoadSapiens': 'Load Sapiens Pose',
    'SapiensPose': 'Sapiens 2D Human Pose',
    'LoadPromptHMR': 'Load PromptHMR',
    'PromptHMRPose': 'PromptHMR 3D Human Pose',
}

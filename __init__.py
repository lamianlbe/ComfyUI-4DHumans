import os

REPO_PATH = os.path.dirname(os.path.abspath(__file__))

from folder_paths import models_dir

SMPL_PATH = os.path.join(models_dir, 'smpl')
os.makedirs(SMPL_PATH, exist_ok=True)


from .nodes.load_phalp_node import LoadPHALPNode
from .nodes.phalp_pose_node import PHALPPoseControlNetNode
from .nodes.load_smplestx_node import LoadSMPLestXNode
from .nodes.load_sapiens_node import LoadSapiensNode
from .nodes.sapiens_single_pose_node import SapiensSinglePoseNode
from .nodes.sapiens_multi_pose_node import SapiensMultiPoseNode
from .nodes.smplestx_pose_node import SMPLestXPoseNode
from .nodes.sapiens_goliath_pose_node import SapiensGoliathPoseNode
from .nodes.load_prompthmr_node import LoadPromptHMRNode
from .nodes.prompthmr_pose_node import PromptHMRPoseNode


NODE_CLASS_MAPPINGS = {
    'LoadPHALP': LoadPHALPNode,
    'PHALPPoseControlNet': PHALPPoseControlNetNode,
    'LoadSMPLestX': LoadSMPLestXNode,
    'LoadSapiens': LoadSapiensNode,
    'SapiensSinglePose': SapiensSinglePoseNode,
    'SapiensMultiPose': SapiensMultiPoseNode,
    'SMPLestXPose': SMPLestXPoseNode,
    'SapiensGoliathPose': SapiensGoliathPoseNode,
    'LoadPromptHMR': LoadPromptHMRNode,
    'PromptHMRPose': PromptHMRPoseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'LoadPHALP': 'Load PHALP',
    'PHALPPoseControlNet': '4D Human Pose Tracking (ControlNet)',
    'LoadSMPLestX': 'Load SMPLest-X',
    'LoadSapiens': 'Load Sapiens Pose',
    'SapiensSinglePose': 'Sapiens Single Person Pose Tracking',
    'SapiensMultiPose': 'Sapiens Multiple Person Pose Tracking',
    'SMPLestXPose': 'SMPLest-X Human Pose Tracking',
    'SapiensGoliathPose': 'Sapiens Single Person Pose Tracking (Goliath)',
    'LoadPromptHMR': 'Load PromptHMR',
    'PromptHMRPose': 'PromptHMR 3D Human Pose',
}

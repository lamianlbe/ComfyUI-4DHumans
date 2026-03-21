import os

REPO_PATH = os.path.dirname(os.path.abspath(__file__))

from folder_paths import models_dir

SMPL_PATH = os.path.join(models_dir, 'smpl')
os.makedirs(SMPL_PATH, exist_ok=True)

WEB_DIRECTORY = "./web"

from .nodes.load_phalp_node import LoadPHALPNode
from .nodes.phalp_pose_node import PHALPPoseControlNetNode
from .nodes.load_sapiens_node import LoadSapiensNode
from .nodes.load_prompthmr_node import LoadPromptHMRNode
from .nodes.sapiens_prompthmr_pose_node import SapiensPromptHMRPoseNode
from .nodes.sapiens_prompthmr_to_nlf_node import SapiensPromptHMRToNLFNode
from .nodes.sam3_node import LoadSAM3Node, SAM3VideoSegmentationNode
from .nodes.pose_renderer_node import PoseRendererNode
from .nodes.save_pose_node import SavePoseDataNode
from .nodes.load_pose_node import LoadPoseDataNode
from .nodes.pose_editor_node import PoseEditorNode


NODE_CLASS_MAPPINGS = {
    'LoadPHALP': LoadPHALPNode,
    'PHALPPoseControlNet': PHALPPoseControlNetNode,
    'LoadSapiens': LoadSapiensNode,
    'LoadPromptHMR': LoadPromptHMRNode,
    'SapiensPromptHMRPose': SapiensPromptHMRPoseNode,
    'SapiensPromptHMRToNLF': SapiensPromptHMRToNLFNode,
    'LoadSAM3': LoadSAM3Node,
    'SAM3VideoSegmentation': SAM3VideoSegmentationNode,
    'PoseRenderer': PoseRendererNode,
    'SavePoseData': SavePoseDataNode,
    'LoadPoseData': LoadPoseDataNode,
    'PoseEditor': PoseEditorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'LoadPHALP': 'Load PHALP',
    'PHALPPoseControlNet': '4D Human Pose Tracking (ControlNet)',
    'LoadSapiens': 'Load Sapiens Pose',
    'LoadPromptHMR': 'Load PromptHMR',
    'SapiensPromptHMRPose': 'Sapiens PromptHMR Human Pose',
    'SapiensPromptHMRToNLF': 'Sapiens PromptHMR to NLF Poses',
    'LoadSAM3': 'Load SAM3',
    'SAM3VideoSegmentation': 'SAM3 Video Segmentation',
    'PoseRenderer': 'Sapiens PromptHMR Pose Renderer',
    'SavePoseData': 'Save Pose Data',
    'LoadPoseData': 'Load Pose Data',
    'PoseEditor': 'Pose Editor',
}

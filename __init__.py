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
from .nodes.load_nlf_node import LoadNLFNode
from .nodes.sapiens_prompthmr_pose_node import SapiensPromptHMRPoseNode
from .nodes.sapiens_prompthmr_to_nlf_node import SapiensPromptHMRToNLFNode
from .nodes.sam3_node import LoadSAM3Node, SAM3VideoSegmentationNode
from .nodes.sam3_image_node import SAM3ImageSegmentationNode
from .nodes.yolo_seg_node import LoadYOLONode, YOLOInstanceSegmentationNode
from .nodes.pose_renderer_node import PoseRendererNode
from .nodes.save_pose_node import SavePoseDataNode
from .nodes.load_pose_node import LoadPoseDataNode
from .nodes.pose_editor_node import PoseEditorNode
from .nodes.wan_animate_face_preprocess_node import WanAnimateFacePreprocessNode
from .nodes.load_fast_sam_3d_body_node import LoadFastSAM3DBodyNode
from .nodes.load_yolo11_pose_node import LoadYOLO11PoseNode
from .nodes.load_farl_face_node import LoadFaRLFaceNode
from .nodes.fastsam3db_farl_pose_node import FastSAM3DBodyFaRLPoseNode
from .nodes.load_crowdsam_node import LoadCrowdSAMNode
from .nodes.crowdsam_seg_node import CrowdSAMInstanceSegmentationNode


NODE_CLASS_MAPPINGS = {
    'LoadPHALP': LoadPHALPNode,
    'PHALPPoseControlNet': PHALPPoseControlNetNode,
    'LoadSapiens': LoadSapiensNode,
    'LoadPromptHMR': LoadPromptHMRNode,
    'LoadNLF': LoadNLFNode,
    'SapiensPromptHMRPose': SapiensPromptHMRPoseNode,
    'SapiensPromptHMRToNLF': SapiensPromptHMRToNLFNode,
    'LoadSAM3': LoadSAM3Node,
    'SAM3VideoSegmentation': SAM3VideoSegmentationNode,
    'SAM3ImageSegmentation': SAM3ImageSegmentationNode,
    'LoadYOLO': LoadYOLONode,
    'YOLOInstanceSegmentation': YOLOInstanceSegmentationNode,
    'PoseRenderer': PoseRendererNode,
    'SavePoseData': SavePoseDataNode,
    'LoadPoseData': LoadPoseDataNode,
    'PoseEditor': PoseEditorNode,
    'WanAnimateFacePreprocess': WanAnimateFacePreprocessNode,
    'LoadFastSAM3DBody': LoadFastSAM3DBodyNode,
    'LoadYOLO11Pose': LoadYOLO11PoseNode,
    'LoadFaRLFace': LoadFaRLFaceNode,
    'FastSAM3DBodyFaRLPose': FastSAM3DBodyFaRLPoseNode,
    'LoadCrowdSAM': LoadCrowdSAMNode,
    'CrowdSAMInstanceSegmentation': CrowdSAMInstanceSegmentationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'LoadPHALP': 'Load PHALP',
    'PHALPPoseControlNet': '4D Human Pose Tracking (ControlNet)',
    'LoadSapiens': 'Load Sapiens Pose',
    'LoadPromptHMR': 'Load PromptHMR',
    'LoadNLF': 'Load NLF',
    'SapiensPromptHMRPose': 'Sapiens PromptHMR Human Pose',
    'SapiensPromptHMRToNLF': 'Sapiens PromptHMR to NLF Poses',
    'LoadSAM3': 'Load SAM3',
    'SAM3VideoSegmentation': 'SAM3 Video Segmentation',
    'SAM3ImageSegmentation': 'SAM3 Image Segmentation',
    'LoadYOLO': 'Load YOLO',
    'YOLOInstanceSegmentation': 'YOLO Instance Segmentation',
    'PoseRenderer': 'Sapiens PromptHMR Pose Renderer',
    'SavePoseData': 'Save Pose Data',
    'LoadPoseData': 'Load Pose Data',
    'PoseEditor': 'Pose Editor',
    'WanAnimateFacePreprocess': 'Wan Animate Face Preprocess',
    'LoadFastSAM3DBody': 'Load Fast SAM 3D Body',
    'LoadYOLO11Pose': 'Load YOLO11m-Pose',
    'LoadFaRLFace': 'Load FaRL Face',
    'FastSAM3DBodyFaRLPose': 'Fast SAM 3D Body + FaRL Face Pose',
    'LoadCrowdSAM': 'Load CrowdSAM',
    'CrowdSAMInstanceSegmentation': 'CrowdSAM Instance Segmentation',
}

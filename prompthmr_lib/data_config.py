"""
Bundled data_config for PromptHMR.

Paths are set at runtime by the ComfyUI loader node before any prompt_hmr
module is imported.  The defaults here are placeholders.
"""
from os.path import join

ROOT = './'
ANN_ROOT = 'data/annotations'

DATASET_FILES = {}
DATASET_FOLDERS = {}

# These will be patched at runtime by load_prompthmr_node.py
SMPLX_PATH = 'data/body_models/smplx'
SMPL_PATH = 'data/body_models/smpl'
SMPLX2SMPL = 'data/body_models/smplx2smpl.pkl'

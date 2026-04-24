import torch
import json
import os
from pycocotools.coco import COCO
from PIL import Image
from .coco_names import coco_classes
data_meta = {'crowdhuman':["./datasets/crowdhuman", 1, {1:'person'}],
             'occhuman':["./datasets/OCHuman", 1, {1:'person'}],
             'coco_occ':["./datasets/coco", 80, coco_classes],
             'coco':["./datasets/occ_coco", 80, coco_classes], 
             }
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_directory, transform=None):
        """
        Initialize COCO dataset
        
        Args:
            annotation_file (str): Path to COCO annotation file (JSON)
            image_directory (str): Path to directory containing images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.coco = COCO(annotation_file)
        self.image_directory = image_directory
        self.transform = transform
        self.image_ids = self.coco.getImgIds()
        self.mapping = self.create_category_mapping()
    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.image_ids)
    def create_category_mapping(self):
        """
        Create a mapping from COCO category IDs to a continuous range from 0 to 80
        
        Returns:
            dict: Mapping from COCO category IDs to a continuous range from 0 to 80
        """
        coco_categories = self.coco.loadCats(self.coco.getCatIds())
        coco_category_ids = sorted([category['id'] for category in coco_categories])
        category_mapping = {coco_id: idx for idx, coco_id in enumerate(coco_category_ids)}
        return category_mapping
    def __getitem__(self, idx):
        """
        Get the sample at the specified index
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            dict: Dictionary containing the image and its associated masks and categories
        """
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_directory}/train2017/{image_info['file_name']}"
        image = Image.open(image_path)

        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        
        masks = [self.coco.annToMask(annotation) for annotation in annotations]
        category_ids = [self.mapping[annotation['category_id']] for annotation in annotations]

        sample = {
            'image': image,
            'masks': masks,
            'categories': category_ids
        }


        return sample

class CrowdHuman(torch.utils.data.Dataset):
    def __init__(self, dataset_root, annot_path, transform):
        self.dataset_root = dataset_root
        self.transform = transform
        img_dir = 'Images'        
        annots = json.load(open( annot_path))
        annotations = annots['annotations']
        images = annots['images']
        self.image_ids = [img['id'] for img in images]
        self.boxes = {}
        for annot in annotations:
            image_id = int(annot['image_id'])
            if image_id not in self.boxes.keys():
                self.boxes[image_id] = []
            self.boxes[image_id].append(annot['bbox'])
        
        self.image_files = [os.path.join(dataset_root, img_dir,img['file_name']) for img in images]    
    def __getitem__(self, item):
        img = Image.open(self.image_files[item])
        w,h = img.size
        boxes = torch.tensor(self.boxes[item])
        boxes = boxes / torch.tensor([w,h,w,h]).unsqueeze(0)
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
        return img, boxes
    def __len__(self):
        return len(self.image_files)
    
def collate_fn_crowdhuman(data):
    images, boxes= zip(*data)
    return images, boxes

def collate_fn_coco(data):
    images = [d['image'] for d in  data]
    masks = [d['masks'] for d in data]
    categories = [d['categories'] for d in data]
    # images,boxes, boxe_full= zip(*data.values())
    return images, masks, categories


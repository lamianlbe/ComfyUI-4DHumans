import torch
from loguru import logger
import tqdm
import numpy as np
from crowdsam.loss import compute_loss_coco, compute_loss_crowdhuman
import torch.nn.functional as F
from collections import defaultdict
@torch.no_grad()
def cache_feature_coco(train_dataloader,  sam,  max_steps = 100,):
        #training loop
    dataloder_iter = iter(train_dataloader)
    # logger.info('Start caching SAM\'s image embeddings for training.. (This will take several seconds) ')
    cached_sam_feature = []
    for step in tqdm.tqdm(range(0, max_steps)):
        data = next(dataloder_iter)
        imgs, masks, categories = data
        #select one image for training
        image = imgs[0]
        masks = masks[0]
        categories = categories[0]
        image_np = np.array(image)
        
        # image_np = (255*image_np.permute(1,2,0).numpy()).astype(np.uint8)
        img_height, img_width = image_np.shape[:2] #C,H,W        \
        scale = min(256/img_height, 256/img_width)
        if (len(masks)==0):
            continue
        res_h, res_w = int(scale*img_height), int(scale*img_width)
        masks = F.interpolate(torch.tensor(masks).unsqueeze(0).float(), (res_h, res_w), mode='bilinear')[0]
        # scale = torch.tensor([img_width,img_height, img_width, img_height]).unsqueeze(0)
        sam.set_image(image_np)
        dino_features = sam.dino_feats
        # masks = masks.any(dim=0,keepdims=True)
        data_item = {
            'image': image_np,
            'masks': masks, 
            'categories': categories,
            'img_height': img_height,   
            'img_width': img_width,
            'sam_embedding': sam.get_image_embedding().cpu(),
            'dino_features': dino_features.cpu(),
        }
        cached_sam_feature.append(data_item)
        del sam.dino_feats
        del sam.features
    return cached_sam_feature

@torch.no_grad()
def cache_feature_crowdhuman(train_dataloader, sam, max_steps=100, feat_size=40, patch_size=14, debug=False):
    """
    Caches image embeddings for the SAM model from the training data.

    Args:
    - train_dataloader (DataLoader): DataLoader providing the training data.
    - sam (SamPredictor): The SAM model instance.
    - max_steps (int): Maximum number of steps to cache embeddings.
    - feat_size (int): Feature size (default 40).
    - patch_size (int): Patch size (default 14).
    - debug (bool): If true, enables debugging mode.

    Returns:
    - cache (list): List of cached features including image embeddings, DINO features, target boxes, image dimensions, and masks.
    """
    # Initialize dataloader iterator
    dataloader_iter = iter(train_dataloader)
    logger.info('Start caching SAM\'s image embeddings for training.. (This will take several seconds) ')
    cached_sam_feature = []
    for step in tqdm.tqdm(range(0, max_steps)):
        # Get the next batch of data
        data = next(dataloader_iter)
        imgs, target_boxes = data

        # Select one image for training
        image = imgs[0]
        image_np = np.array(image)
        target_boxes = target_boxes[0] 
        img_height, img_width = image_np.shape[:2]

        # Scale the target boxes to the image dimensions
        scale = torch.tensor([img_width, img_height, img_width, img_height]).unsqueeze(0)
        target_boxes = target_boxes * scale

        # Set the image in the SAM model
        sam.set_image(image_np)

        # Apply transformations to the target boxes
        prompt_boxes = sam.transform.apply_boxes(target_boxes.numpy(), sam.original_size)
        prompt_boxes = torch.tensor(prompt_boxes)[:, None, :].cuda()

        # Extract only low resolution masks
        masks_list = []
        masks = predict_torch(sam, boxes=prompt_boxes, multimask_output=False)[0].cpu()
        masks = (masks > sam.model.mask_threshold)


        img_height, img_width = image_np.shape[:2] #C,H,W        \
        scale = min(256/img_height, 256/img_width)
        if (len(masks)==0):
            continue
        res_h, res_w = int(scale*img_height), int(scale*img_width)
        masks = masks[:,0, :res_h, :res_w]  
        assert len(masks) == len(target_boxes)
        
        # Get DINO features
        dino_features = sam.dino_feats

        # Cache the image embeddings, DINO features, target boxes, image dimensions, and masks
        data_item = {
            'image': image_np,
            'masks': masks, 
            'categories': [0 for  i in range(len(masks))],
            'img_height': img_height,   
            'img_width': img_width,
            'sam_embedding': sam.get_image_embedding().cpu(),
            'dino_features': dino_features.cpu(),
        }
        cached_sam_feature.append(data_item)
    return cached_sam_feature

def clip_grads(params,  max_norm=0.1):    
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        return torch.nn.utils.clip_grad_norm_(
            parameters=params,
            max_norm  = max_norm,
        )

def predict_torch(
        predictor,
        point_coords = None,
        point_labels = None,
        boxes = None,
        multimask_output = True,
    ) :
        #we modify the definition of point_labels here to define pos point point label = 1 , neg point label = 0
    if point_coords is not None:
        assert len(point_coords) == len(point_labels)
        points = (point_coords,  point_labels)
    else:
        points = None

    # Embed prompts
    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=None,
    )

    # import pdb;pdb.set_trace()
    # Predict masks
    low_res_masks, iou_predictions, categories = predictor.model.mask_decoder(
        image_embeddings=predictor.features,
        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        dino_feats = predictor.dino_feats,
    )
    #B,C,H,W -> B,H,W,C for MLP to process

    return low_res_masks, iou_predictions, categories
def train_loop(data_loader,  predictor, optimizer, max_steps=3000, n_shot=10, instance_num=20, neg_instance_num= 40, class_num =1, loss_type='coco', debug=False, use_cache=True):
    #sample num: total data used for training
    # instance_num: number of instance per image
    if loss_type == 'coco':
        cached_sam_feature = cache_feature_coco(data_loader,predictor,  n_shot, )    
    else:
        cached_sam_feature = cache_feature_crowdhuman(data_loader, predictor, n_shot)
    accuracy = 0
    for step in range(0, max_steps):
        #Extract sample according to step
        sample_idx = step%len(cached_sam_feature)
        data_item = cached_sam_feature[sample_idx]
        features = data_item['sam_embedding']
        dino_features = data_item['dino_features']
        target_masks = data_item['masks']
        target_categories = data_item['categories']
        (img_height, img_width) = data_item['img_height'], data_item['img_width']
        image_np = data_item['image']
        if len(target_masks) == 0:
            continue
 
        # features, dino_features, target_masks, target_categories , (img_height, img_width)= 
        #compute scale of the image
        scale = min(256/img_height, 256/img_width)     
    
        # Convert data to tensor
        target_masks =  torch.tensor(target_masks)
        target_categories =  torch.tensor(target_categories)
        fg_bin_mask = target_masks.any(dim=0)
        # Create a foreground classification mask
        fg_cls_mask = torch.zeros(target_masks.shape[1], target_masks.shape[2], class_num)
        for mask,class_id in zip(target_masks, target_categories):
            coords = mask.nonzero()
            fg_cls_mask[coords[:,0],coords[:,1], class_id] = 1
        #Sample positive point prompts. Each target mask will have one point prompt
        #Sample instance_num instances from the image
        instance_num = instance_num
        sample_ind = np.random.choice(np.arange(len(target_masks)), instance_num,replace=True)
        target_masks = target_masks[sample_ind]
        target_categories = target_categories[sample_ind]

        pos_point_coords = []
        for mask in target_masks:
            coords = mask.nonzero()[:, [1,0]] # convert to xy
            select_point = coords[np.random.randint(0, len(coords))].view(-1,2)
            pos_point_coords.append(select_point)        
        pos_point_coords = torch.cat(pos_point_coords, dim=0) / scale #divide scale to get the original image size
    
        #Sample negative point prompts
        neg_coords = (~fg_bin_mask)[:, :].nonzero()[:,[1,0]] / scale
        if len(neg_coords) == 0:
            continue
        neg_point_coords = neg_coords[np.random.choice(np.arange(len(neg_coords)), neg_instance_num)].view(-1,2)
        point_coords = torch.cat([pos_point_coords, neg_point_coords], dim=0)
        # move data to cuda        
        prompt_coords_trans = predictor.transform.apply_coords(point_coords.unsqueeze(1).numpy(), (img_height, img_width))
        prompt_coords_trans = torch.from_numpy(prompt_coords_trans).cuda()
        prompt_labels = torch.ones_like(prompt_coords_trans)[:,:,0].cuda() # set sam labels to 1 for positive points and 0 for negative points
        target_masks = target_masks.cuda()
        fg_cls_mask = fg_cls_mask.cuda()
        predictor.features = features.cuda()
        predictor.dino_feats = dino_features.cuda()

        #predict foreground classification logits
        fg_cls_logits = predictor.predict_fg_map((img_height, img_width))[0]
        # crop the logits to valid size (no bg included)
        fg_cls_logits = fg_cls_logits[:,:int(scale*img_height), :int(scale*img_width)]
        # predict masks according to the point prompts
        low_res_masks, iou_predictions, cls_predictions = predict_torch(predictor, prompt_coords_trans, prompt_labels)
    
        if loss_type == 'coco':
            loss_dict = compute_loss_coco(low_res_masks , #256x256
                                    iou_predictions,  #Nx3
                                    cls_predictions,  #Nx3xN_class
                                    fg_cls_logits.permute(1,2,0),    #256x256
                                    target_masks= target_masks,
                                    target_categories = target_categories,
                                    fg_target=fg_cls_mask,
                                    num_pos_sample=instance_num,
                                    debug = debug)
        elif loss_type == 'crowdhuman': 
            loss_dict = compute_loss_crowdhuman(low_res_masks , #256x256
                                    iou_predictions,  #Nx3
                                    cls_predictions,  #Nx3xN_class
                                    fg_cls_logits.permute(1,2,0),    #256x256
                                    target_masks= target_masks,
                                    target_categories = target_categories,
                                    fg_target=fg_cls_mask,
                                    num_pos_sample=instance_num,
                                    debug = debug)
        if debug and step % 300  in range(9):
            from detectron2.utils.visualizer import Visualizer
            from detectron2.data import MetadataCatalog
            from detectron2.structures import Instances, Boxes, BitMasks

            # 创建 metadata（你也可以注册自己的 dataset）
            coco_metadata = MetadataCatalog.get("coco_2017_val")   # or create a dummy one
            coco_metadata.stuff_classes = coco_metadata.thing_classes
            # coco_metadata.thing_classes = coco_names  # 如果你没用 register_coco_instances
            # 获取预测标签（H, W）
            with torch.no_grad():   
                upsampled_cls_pred = F.interpolate(fg_cls_logits.unsqueeze(0), size=(img_height, img_width), mode='bilinear', align_corners=False)
                fg_mask = upsampled_cls_pred.sigmoid() > 0.1
                cls_conf, cls_labels = upsampled_cls_pred.cpu().max(dim=1)  # shape: [H, W]
                cls_labels[cls_conf<0.1] = 80  # 将背景区域设置为0
                cls_labels = cls_labels.squeeze(0).long()  # shape: [H, W]  
                # 可视化叠加在原图上（image_np 为原图）
                visualizer = Visualizer(image_np, metadata=coco_metadata, scale=1.0)
                vis_output = visualizer.draw_sem_seg(cls_labels.int())
                # 保存或显示
                vis_output.save(f"outputs/sam_adapter_debug/{step}_cls_vis.jpg")
                
                low_res_masks= low_res_masks.cpu()
                pred_iou_conf,select_mask = iou_predictions.cpu().max(dim=1)
                low_res_masks = low_res_masks[np.arange(len(low_res_masks)), select_mask]
                low_res_masks = low_res_masks[:,:int(scale*img_height), :int(scale*img_width)]
                cls_predictions = cls_predictions[np.arange(len(cls_predictions)), select_mask]
                # import pdb;pdb.set_trace()
                # 1. 上采样 mask 到原图尺寸
                upsampled_masks = F.interpolate(low_res_masks.float().unsqueeze(1), size=(img_height, img_width), mode='bilinear') > 0
                upsampled_masks = upsampled_masks.squeeze(1).cpu().bool().numpy()  # [N, H, W]

                # 2. 获取类别
                pred_conf, pred_classes = cls_predictions.sigmoid().cpu().max(dim=1)  # [N]
                pred_conf = pred_conf #torch.sqrt(pred_conf*pred_iou_conf)
                conf_thresh = 0.2
                pred_classes = pred_classes[pred_conf > conf_thresh].long()  # [N]
                upsampled_masks = upsampled_masks[pred_conf > conf_thresh]# [N]
                # 3. 创建 Instances 对象
                instances = Instances(image_size=(img_height, img_width))
                instances.pred_classes = pred_classes  # [N]
                instances.pred_masks = (upsampled_masks)  # [N, H, W]
                instances.scores = pred_conf[pred_conf > conf_thresh]  # [N]
# 可视化

                # 可选：创建伪 bbox（用于可视化）
                visualizer = Visualizer(image_np, metadata=coco_metadata, scale=1.0)
                vis_output = visualizer.draw_instance_predictions(instances)
                vis_output.save(f"outputs/sam_adapter_debug/{step}_cls_inst_vis.jpg")

        # Compute the total loss
        optimizer.zero_grad()  
        total_loss = sum([v for k,v in loss_dict.items() if 'loss' in k])
        total_loss.backward()
        clip_grads(predictor.model.parameters(), 0.1)
        optimizer.step()
        loss_dict_data = {k:round(float(v.data),3) for k,v in loss_dict.items() if 'accuracy' not in k} 
        # Compute accuracy  
        # accuracy = 0.99* accuracy + 0.01 * loss_dict['accuracy']
        if step %5 == 0:
            # Print the training progress
            output_str = f"step: {step}/{max_steps}" # accuracy: {round(float(accuracy),3)} "
            for k,v in loss_dict_data.items():
                output_str += f"{k}: {v} "
            logger.info(output_str, flush=True)
    
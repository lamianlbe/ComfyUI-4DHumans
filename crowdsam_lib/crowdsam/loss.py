import torch
import crowdsam.utils as utils
import torch.nn.functional as F
def clip_grads(params,  max_norm=0.1):    
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        return torch.nn.utils.clip_grad_norm_(
            parameters=params,
            max_norm  = max_norm,
        )

def compute_loss_coco(low_res_masks:torch.Tensor,
                 iou_pred:torch.Tensor, 
                 cls_logits_pred:torch.Tensor,
                 fg_cls_logits_pred:torch.Tensor,
                 target_masks:torch.Tensor,
                 target_categories:torch.Tensor,
                 fg_target:torch.Tensor,
                 num_pos_sample:int, 
                 debug=False):
    # low_res_masks: (num_masks, 3, 256, 256)
    # iou_pred: (num_masks, 3, 1)
    # cls_logits_pred: (num_masks, num_classes)
    # fg_cls_logits_pred: (h, w, num_classes) 
    # target_masks: (num_pos_sample,  h, w)
    # target_categories: (num_pos_sample)
    # fg_target: (h, w, num_classes)
    # num_pos_sample: number of positive samples
    # debug: whether to print debug information
    
    
    #keep masks predicted positive prompts only
    low_res_masks = low_res_masks[:num_pos_sample]
    _, h, w = target_masks.shape
    valid_pred_mask = low_res_masks[:,:,:h,:w]
    
    #[num_masks, 3]
    iou_pred_target = utils.mIoU(valid_pred_mask, target_masks.unsqueeze(1))
    # Compute classification loss (Dice Loss) for the foreground mask
    # H,W,C -> B,C,H,W
    dice_loss = utils.dice_loss(fg_cls_logits_pred.permute(2,0,1).unsqueeze(0), fg_target.permute(2,0,1).unsqueeze(0)).mean()
    fg_cls_loss = utils.sigmoid_focal_loss(fg_cls_logits_pred, fg_target)

    # Compute IoU loss
    iou_loss = F.mse_loss(iou_pred[:num_pos_sample], iou_pred_target) + F.mse_loss(iou_pred[num_pos_sample:], torch.zeros_like(iou_pred[num_pos_sample:]))

    # Separate classification loss for positive and negative samples    
    cls_target = torch.zeros_like(cls_logits_pred)
    cls_target[torch.arange((num_pos_sample)), :, target_categories] = 1
    inst_cls_loss = utils.sigmoid_focal_loss(cls_logits_pred, cls_target)
    

    #compute accuracy
    with torch.no_grad():
        categories_pred =  cls_logits_pred.mean(dim=1).max(dim=-1)[1]
        accuracy  = ((categories_pred[:num_pos_sample].cpu() == target_categories).float().mean())
    # Build loss dictionary
    loss_dict = {
        'iou_loss':iou_loss,
        'focal_loss':  inst_cls_loss  + fg_cls_loss ,
        'accuracy': accuracy
    }
    return loss_dict


def compute_loss_crowdhuman(low_res_masks:torch.Tensor,
                 iou_pred:torch.Tensor, 
                 cls_logits_pred:torch.Tensor,
                 fg_cls_logits_pred:torch.Tensor,
                 target_masks:torch.Tensor,
                 target_categories:torch.Tensor,
                 fg_target:torch.Tensor,
                 num_pos_sample:int, 
                 debug=False):
    # low_res_masks: (num_masks, 3, 256, 256)
    # iou_pred: (num_masks, 3, 1)
    # cls_logits_pred: (num_masks, num_classes)
    # fg_cls_logits_pred: (h, w, num_classes) 
    # target_masks: (num_pos_sample,  h, w)
    # target_categories: (num_pos_sample)
    # fg_target: (h, w, num_classes)
    # num_pos_sample: number of positive samples
    # debug: whether to print debug information
    assert low_res_masks.shape[0] == iou_pred.shape[0], "low_res_masks and iou_pred must have the same number of masks"
  
    low_res_masks = low_res_masks[:num_pos_sample]
    _, h, w = target_masks.shape
    valid_pred_mask = low_res_masks[:,:,:h,:w]
    
    # Keep masks predicted for positive prompts only
    # Compute Dice Loss between predicted and target masks
    dc_loss = utils.dice_loss(valid_pred_mask, target_masks.unsqueeze(1).float())
    iou_pred_target = utils.mIoU(valid_pred_mask, target_masks.unsqueeze(1).float())

    # Find the index of the mask with the minimum Dice loss

    # Compute classification loss (Dice Loss) for the foreground mask
    dice_loss = utils.dice_loss(fg_cls_logits_pred.permute(2,0,1).unsqueeze(0), fg_target.permute(2,0,1).unsqueeze(0)).mean()
    focal_loss = utils.sigmoid_focal_loss(fg_cls_logits_pred, fg_target)
    # Prepare IoU target tensor
    iou_target = torch.zeros_like(iou_pred)
    iou_target[torch.arange(num_pos_sample)] = iou_pred_target
    cls_loss = F.mse_loss(iou_pred * cls_logits_pred.sigmoid().squeeze(2), iou_target, reduction='none').sum(dim=[1])

    # Separate classification loss for positive and negative samples
    pos_cls_loss = cls_loss[:num_pos_sample].mean()
    neg_cls_loss = cls_loss[num_pos_sample:].mean()
    # Build loss dictionary
    loss_dict = {
        'pos_cls_loss': pos_cls_loss,
        'neg_cls_loss': neg_cls_loss,
        'dice_loss': dice_loss,
        # 'focal_loss': focal_loss,   
    }
    # Log the loss values if in debug mode
    return loss_dict

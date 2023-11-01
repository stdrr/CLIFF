from .cliff_hr48.cliff import CLIFF
from lib.yolov3_detector import HumanDetector
from torch import nn
from common.imutils import process_image
from common.utils import estimate_focal_length



class CLIFF_FULL(nn.Module):
    
    def __init__(self, smpl_mean_params, img_feat_num=2048, device="cuda"):
        super(CLIFF_FULL, self).__init__()
        
        self.detector = HumanDetector()
        self.cliff = CLIFF(smpl_mean_params, img_feat_num)
        self.device = device
        
        
    def _estimate_from_detector(self, img_bgr, detection_result):
        """
        bbox: [batch_id, min_x, min_y, max_x, max_y, det_conf, nms_conf, category_id]
        :param idx:
        :return:
        """
        item = {}
        img_idx = int(detection_result[0].item())
        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        focal_length = estimate_focal_length(img_h, img_w)

        bbox = detection_result[1:5]
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(img_rgb, bbox)

        item["norm_img"] = norm_img
        item["center"] = center
        item["scale"] = scale
        item["crop_ul"] = crop_ul
        item["crop_br"] = crop_br
        item["img_h"] = img_h
        item["img_w"] = img_w
        item["focal_length"] = focal_length
        item["img_idx"] = img_idx
        return item 
    
    
    def _estimate_from_bbox_gt(self, img_bgr, bbox):
        item = {}
        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        focal_length = estimate_focal_length(img_h, img_w)

        norm_img, center, scale, crop_ul, crop_br, _ = process_image(img_rgb, bbox)

        item["norm_img"] = norm_img
        return item 
        
        
    def forward(self, batch, batch_idx, use_detector=True):
        dim = batch["dim"].to(self.device).float()
        img = batch["img"].to(self.device).float()
        if use_detector:
            detection_result = self.detector.detect_batch(img, dim)
            det_batch_size = img.shape[0]
            detection_result[:, 0] += batch_idx * det_batch_size
            detection_result = detection_result.cpu().numpy()
            new_params = self._estimate_from_detector(img, detection_result)
        else:
            bbox_info = batch['bbox_info'].to(self.device)
            new_params = self._estimate_from_bbox_gt(img, bbox_info)
        
        self.cliff(img, bbox_info)
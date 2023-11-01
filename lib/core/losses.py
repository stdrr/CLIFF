import torch
import torch.nn.functional as F

from lib.utils.geometry import batch_rodrigues
from lib.utils import rotation_conversions as geo

def compute_l2_loss(batch):
    x2 = batch["x2"]
    output = batch["output"]
    
    loss = F.mse_loss(x2, output, reduction='mean')
    return loss


def keypoint_loss(batch, openpose_weight=0., gt_weight=1.):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    pred_keypoints_2d = batch['pred_keypoints_2d']
    gt_keypoints_2d = batch['keypoints']

    # Adapted from SPIN
    conf = gt_keypoints_2d[:, :, [-1]].clone()
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight

    mse  = F.mse_loss(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1], reduction='none')
    loss = (conf * mse).mean()
    return loss


def keypoint_3d_loss(batch):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = batch['pred_keypoints_3d']
    gt_keypoints_3d = batch['pose_3d']
    has_pose_3d = batch['has_pose_3d']
    device = pred_keypoints_3d.device

    # Adapted from SPIN
    # "has_pose_3d" indicates samples that have pose_3d
    # "conf" indicates valid 3d keypoints
    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]

    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

        mse  = F.mse_loss(pred_keypoints_3d, gt_keypoints_3d, reduction='none')
        loss = (conf * mse).mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).mean().to(device)

    return loss


def smpl_losses(batch, pose_weight=1., beta_weight=0.001):
    pred_rotmat = batch['pred_rotmat']
    pred_betas  = batch['pred_betas']
    gt_pose  = batch['pose']
    gt_betas = batch['betas']
    has_smpl = batch['has_smpl']
    device = pred_rotmat.device

    # Adapted from SPIN
    pred_rotmat_valid = pred_rotmat[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose  = F.mse_loss(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_betas = F.mse_loss(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose  = torch.FloatTensor(1).fill_(0.).mean().to(device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).mean().to(device)

    loss = pose_weight*loss_regr_pose + beta_weight*loss_regr_betas
    return loss

def vertice_loss(batch):
    pred_rotmat = batch['pred_rotmat']
    pred_betas  = batch['pred_betas']
    has_smpl = batch['has_smpl']
    smpl = batch['smpl']
    device = pred_rotmat.device

    # pred vertices
    pred_out = smpl(global_orient=pred_rotmat[:,0],
                    body_pose=pred_rotmat[:,1:],
                    betas=pred_betas)
    pred_vert = pred_out.vertices

    # gt vertices
    if 'gt_vert' not in batch:
        gt_pose  = batch['pose']
        gt_betas = batch['betas']
        gt_rotmat = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)

        gt_out = smpl(global_orient=gt_rotmat[:,0],
                      body_pose=gt_rotmat[:,1:],
                      betas=gt_betas)
        gt_vert = gt_out.vertices
        batch['gt_vert'] = gt_vert
    else:
        gt_vert = batch['gt_vert']

    gt_vert = gt_vert[has_smpl == 1]
    pred_vert = pred_vert[has_smpl == 1]

    if len(gt_vert) > 0:
        loss  = F.l1_loss(pred_vert, gt_vert)
    else:
        loss = torch.FloatTensor(1).fill_(0.).mean().to(device)

    return loss


def cam_depth_loss(batch):
    # The last component is a loss that forces the network to predict positive depth values
    pred_camera = batch['pred_cam']
    loss = ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()

    return loss.clamp(min=None, max=10.0)


def cam_loss(batch):
    cam = batch['cam']
    has_cam = batch['has_cam']
    pred_cam = batch['pred_cam']
    device = cam.device

    cam = cam[has_cam==1]
    pred_cam = pred_cam[has_cam==1]

    if len(pred_cam) > 0:
        mse  = F.mse_loss(pred_cam, cam, reduction='none')
        loss = mse.mean()

    else:
        loss = torch.FloatTensor(1).fill_(0.).mean().to(device)

    return loss.clamp(min=None, max=10.0)


collection = {'KPT2D': keypoint_loss, 'KPT3D': keypoint_3d_loss, 'SMPL':  smpl_losses,
              'CAM_S': cam_depth_loss, 'CAM': cam_loss, 'V3D': vertice_loss}


def compile_criterion(cfg):
    MixLoss = BaseLoss()
    for t, w in cfg.LOSS.items():
        MixLoss.weights[t] = w
        MixLoss.functions[t] = collection[t]

    return MixLoss


class BaseLoss(torch.nn.Module):
    def __init__(self,):
        super(BaseLoss, self).__init__()
        self.weights = {}
        self.functions = {}

    def forward(self, batch):
        losses = {}
        mixes_loss = 0
        for t, w in self.weights.items():
            loss = self.functions[t](batch)
            mixes_loss += w * loss
            losses[t] = loss.item()

        losses['mixed'] = mixes_loss.item()
        return mixes_loss, losses

    def report(self, ):
        return


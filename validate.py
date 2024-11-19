import torch

import os
import cv2
import matplotlib.cm as cm

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from utils.common import make_matching_plot_fast


@torch.no_grad()
def validate(superpoint:SuperPoint, superglue:SuperGlue, image_A:torch.Tensor, image_B:torch.Tensor):
    pred = {}
    pred0 = superpoint({'image':image_A})
    pred1 = superpoint({'image':image_B})
    pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
    pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
    data = {'image0': image_A, 'image1': image_B, **pred}
    for k in data:
        if isinstance(data[k], (list, tuple)):
            data[k] = torch.stack(data[k])
    pred = {**pred, **superglue(data)}
    return pred


if __name__ == '__main__':
    configs = {
        'superpoint': {
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4,
        },
        'superglue': {
            'descriptor_dim': 256,
            'weights_path': None,
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'sinkhorn_iterations': 20,
            'match_threshold': 0.5,
            'use_layernorm': False
        }
    }
    save_dir = '/workspace/test_results'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    superpoint = SuperPoint(config=configs['superpoint']).to(device)
    superglue = SuperGlue(config=configs['superglue']).to(device)
    superpoint.load_state_dict(torch.load('/workspace/SuperGlue_training_for_yfcc100m_dataset/models/weights/superpoint_v1.pth'))
    superglue.load_state_dict(torch.load('/workspace/SuperGlue_training_for_yfcc100m_dataset/output/train4/default/weights/best.pt')['model'])
    superpoint.eval()
    superglue.eval()

    image_list = ['/workspace/test_data/A1.jpg', '/workspace/test_data/A2.jpg', '/workspace/test_data/A3.jpg',
                  '/workspace/test_data/A11.jpg', '/workspace/test_data/A21.jpg', '/workspace/test_data/A31.jpg']
    preprocess = lambda x: torch.from_numpy(cv2.imread(x, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0  # (1, 1, H, W)
    for i in range(len(image_list)):
        for j in range(i+1, len(image_list)):
            image_tensor_A, image_tensor_B = preprocess(image_list[i]), preprocess(image_list[j])
            pred = validate(superpoint, superglue, image_tensor_A, image_tensor_B)
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            image0, image1 = image_tensor_A[0, 0].cpu().numpy() * 255, image_tensor_B[0, 0].cpu().numpy() * 255
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            text = ['SuperGlue', f'{len(mkpts0)} matches', f'{len(kpts0)} : {len(kpts1)}']
            path = os.path.join(save_dir, '{}_{}_matches.png'.format(str(image_list[i].split('/')[-1].split('.')[0]), str(image_list[j].split('/')[-1].split('.')[0])))
            make_matching_plot_fast(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, cm.jet(mconf),
                text, path, False, 10, False, 'matches', [])
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import torchvision
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam import GradCAM, AblationCAM

import argparse
from tqdm import tqdm

import json
import os
import sys
sys.path.append("../")
sys.path.append('../dl_models')
sys.path.append('../directory_organization')
from directory_organization.fileProcessingUtil import get_fgrp, get_id, get_sensor
from dl_models.embedding_models import EmbeddingNet
from dl_models.parameterized_multiple_finger_tester import load_data, create_output_dir, euclideanDist, DEFAULT_OUTPUT_ROOT, ALL_FINGERS

FGRP_TO_NAME = {'01':'Right Thumb',
                '02':'Right Index',
                '03':'Right Middle',
                '04':'Right Ring',
                '05':'Right Little',
                '06':'Left Thumb',
                '07':'Left Index',
                '08':'Left Middle',
                '09':'Left Ring',
                '10':'Left Little'
                }

# Thanks https://github.com/jacobgil/pytorch-grad-cam
# Thanks https://jacobgil.github.io/pytorch-gradcam-book/Pixel%20Attribution%20for%20embeddings.html

class ContrastiveSaliency:
    def __init__(self, pos_features, neg_features):
        self.pos_features = pos_features
        self.neg_features = neg_features
        return
    
    def __call__(self, model_output):
        # distance from negative example should be greater than distance from positive example
        ret_val = torch.maximum(euclideanDist(model_output.flatten(), self.neg_features.flatten()) - \
                                euclideanDist(model_output.flatten(), self.pos_features.flatten()), torch.Tensor([0]).cuda())
        #print(ret_val)
        return ret_val

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
        return
    
    def __call__(self, model_output):
        #cos = torch.nn.CosineSimilarity(dim=0)
        #res = cos(model_output.squeeze(), self.features.squeeze())
        res = -torch.sqrt(torch.sum((model_output - self.features).pow(2)))
        #print(res)
        return res
    
class DissimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
        return
    
    def __call__(self, model_output):
        #cos = torch.nn.CosineSimilarity(dim=0)
        #res = cos(model_output.squeeze(), self.features.squeeze())
        res = torch.sqrt(torch.sum((model_output - self.features).pow(2)))
        #print(res)
        return res

def create_float_img(tensor_image):
    image_float = tensor_image.squeeze().cpu().numpy()
    image_float = (image_float - image_float.min()) / (image_float.max() - image_float.min())
    image_float = np.transpose(image_float, (1, 2, 0))
    return image_float

if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameterized_multiple_finger_tester.py')
    parser.add_argument('--dataset', help='Path to folders containing images', 
                        default='/data/therealgabeguo/fingerprint_data/sd302_split', type=str)
    parser.add_argument('--weights', help='Path to model weights', 
                        default='/data/therealgabeguo/updated_fingerprint_results_fall23/model_weights/full_based_model_PRETRAINED.pth',
                        type=str)
    parser.add_argument('--output_root', help='Root directory for output',
                        default='/data/therealgabeguo/fingerprint_gradcam_outputs', nargs='?', \
                        type=str)
    parser.add_argument('--num_triplets', help='Number of contrastive triplets to go through',
                        default=50,
                        type=int)

    args = parser.parse_args()

    dataset = args.dataset
    weights = args.weights
    output_dir = create_output_dir(args.output_root)

    print(args)

    plt.rcParams.update({'font.size': 15})

    fingerprint_dataset, test_dataset, test_dataloader = load_data(
        the_data_folder=args.dataset, \
        num_anchors=1, num_pos=1, num_neg=1, scale_factor=1, \
        diff_fingers_across_sets=True, diff_fingers_within_set=True, \
        diff_sensors_across_sets=True, same_sensor_within_set=True, \
        possible_fgrps=ALL_FINGERS
    )

    pretrained_model = EmbeddingNet()
    pretrained_model.load_state_dict(torch.load(args.weights, map_location=torch.device('cuda')))
    pretrained_model = pretrained_model.cuda()

    print(pretrained_model.feature_extractor)

    target_layers = [
        pretrained_model.feature_extractor[7][1]
    ]
    cam = GradCAM(
        model=pretrained_model,
        target_layers=target_layers,
        use_cuda=True
    )

    total_correct = 0
    all_data = list()

    data_iter = iter(test_dataloader)
    for i in tqdm(range(args.num_triplets)):#tqdm(range(len(test_dataloader))):
        # test_images is 3 (anchor, pos, neg) * N (number of sample images) * image_size (1*3*224*224)
        test_images, _, test_filepaths = next(data_iter)
        test_images = [torch.unsqueeze(curr_img[0], 0).cuda() for curr_img in test_images]
        test_filepaths = [curr_filepath[0] for curr_filepath in test_filepaths]

        # 0th image is anchor, 1st image is positive, 2nd image is negative
        anchor_embedding = pretrained_model(test_images[0]).flatten()
        pos_embedding = pretrained_model(test_images[1]).flatten()
        neg_embedding = pretrained_model(test_images[2]).flatten()

        #print(torch.sum(anchor_embedding.pow(2)))
        #print(torch.sum(pos_embedding.pow(2)))
        #print(torch.sum(neg_embedding.pow(2)))
        #print(euclideanDist(anchor_embedding, pos_embedding), torch.sum(anchor_embedding - pos_embedding).pow(2))
        #print(euclideanDist(anchor_embedding, neg_embedding), torch.sum(anchor_embedding - neg_embedding).pow(2))

        # create images
        anchor_image_float = create_float_img(test_images[0])
        pos_image_float = create_float_img(test_images[1])
        neg_image_float = create_float_img(test_images[2])

        # closeness to positive example (as opposed to negative example)
        contrastive_targets = [ContrastiveSaliency(pos_features=pos_embedding, neg_features=neg_embedding)]
        grayscale_contrastive_cam = cam(input_tensor=test_images[0], targets=contrastive_targets, aug_smooth=True, eigen_smooth=True)[0, :]
        contrastive_cam_image = show_cam_on_image(anchor_image_float, grayscale_contrastive_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)
        
        # closeness to negative example (as opposed to positive example)
        reverse_contrastive_targets = [ContrastiveSaliency(pos_features=neg_embedding, neg_features=pos_embedding)] # reversed
        reverse_grayscale_contrastive_cam = cam(input_tensor=test_images[0], targets=reverse_contrastive_targets, aug_smooth=True, eigen_smooth=True)[0, :]
        reverse_contrastive_cam_image = show_cam_on_image(anchor_image_float, reverse_grayscale_contrastive_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

        # plt.imshow(contrastive_cam_image)
        # plt.savefig('contrastive_img.png')

        fig, axes = plt.subplots(2, 3, figsize=(3 * 5, 2 * 5))

        for row_num in range(2):
            for col_num in range(3):
                axes[row_num, col_num].axis('off')

        filenames = [x.split('/')[-1] for x in test_filepaths]

        anchor_pos_dist = euclideanDist(anchor_embedding, pos_embedding)
        anchor_neg_dist = euclideanDist(anchor_embedding, neg_embedding)

        # keep track of stats
        if anchor_pos_dist < anchor_neg_dist:
            total_correct += 1
        all_data.append(test_filepaths)

        axes[0,1].imshow(pos_image_float); axes[0,1].set_title(f'Person {get_id(filenames[1])}:\n{FGRP_TO_NAME[get_fgrp(filenames[1])]}')
        axes[0,2].imshow(neg_image_float); axes[0,2].set_title(f'Person {get_id(filenames[2])}:\n{FGRP_TO_NAME[get_fgrp(filenames[2])]}')
        axes[1,0].imshow(anchor_image_float); axes[1,0].set_title(f'Person {get_id(filenames[0])}:\n{FGRP_TO_NAME[get_fgrp(filenames[0])]}')
        axes[1,1].imshow(contrastive_cam_image); axes[1,1].set_title(f'Intra-Person Similarity\n(Distance = {round(anchor_pos_dist.item(), 3)})')
        axes[1,2].imshow(reverse_contrastive_cam_image); axes[1,2].set_title(f'Inter-Person Similarity\n(Distance = {round(anchor_neg_dist.item(), 3)})')

        plt.savefig(os.path.join(output_dir, f'contrastive_img_{i}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'contrastive_img_{i}.png'), bbox_inches='tight')
        plt.close()
    
    print(f'{total_correct} of {args.num_triplets} correct')

    output_summary = {
        'files': all_data,
        'num_correct': total_correct
    }
    output_summary.update(vars(args))

    with open(os.path.join(output_dir, '_output_summary.json'), 'w') as f:
        json.dump(output_summary, f, indent=4)

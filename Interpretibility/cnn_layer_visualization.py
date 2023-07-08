"""
Created on Sat Nov 18 23:12:08 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import sys
import numpy as np

import torch
from torch import nn
from torch.optim import Adam

sys.path.append("../")
from dl_models.embedding_models import EmbeddingNet

from misc_functions import preprocess_image, recreate_image, save_image
import argparse

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, layer_id, stop_index, out_folder):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.layer_id = layer_id
        self.selected_filter = selected_filter
        self.stop_index = stop_index
        self.conv_output = 0
        self.out_folder = out_folder
        # Create the folder to export images if not exists
        #if not os.path.exists('../generated'):
        #    os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.selected_layer.register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                print("At index ", index, " processing layer ", layer)
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.stop_index:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = self.out_folder + '/layer_vis_l' + str(self.layer_id) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN layer visualization')

    parser.add_argument('--model_path', type=str,
                        default='/data/therealgabeguo/most_recent_experiment_reports/jan_08_resnet18Final/weights_2023-01-07_11:06:28.pth',
                        help='Path to the model file')
    parser.add_argument('--output_folder', type=str,
                        default='/data/verifiedanivray/generated_end',
                        help='Path to the output folder')

    args = parser.parse_args()

    for filter_pos in range(0, 512):
        stop_index = 7
        # Perform network surgery to "flatten" the layer heirarchy of the model
        pretrained_model = EmbeddingNet()
        pretrained_model.load_state_dict(torch.load(args.model_path))
        print(pretrained_model)
        cnn_layer = pretrained_model.feature_extractor[7][1].conv2
        print(cnn_layer)
        pretrained_model = pretrained_model.feature_extractor
        '''
        pretrained_model = pretrained_model.feature_extractor
        modules = []
        for layer in pretrained_model:
            if isinstance(layer, nn.Sequential):
                layer = list(layer)
                for sublayer in layer[0]:
                    # nodes
                for sublayer in layer[1]:
                    # nodes
                modules = modules + list(layer[0].children()) + list(layer[1].children())
            elif isinstance(layer, ):
                modules.append(layer)
        print(modules)
        '''
        layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos, "7.1.conv2", stop_index, args.output_folder)

        # Layer visualization with pytorch hooks
        layer_vis.visualise_layer_with_hooks()
        
        # Layer visualization without pytorch hooks
        # layer_vis.visualise_layer_without_hooks()
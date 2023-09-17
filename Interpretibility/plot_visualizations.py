import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
from PIL import Image
import math

def plot_visualizations(img_directory, num_visualizations=(8, 8), desired_iter=30, the_title='Layer Visualizations', \
        zoom_factor=1, the_figsize=(10, 10), save_dir='/data/therealgabeguo/updated_fingerprint_results_fall23/nn_filter_visualizations/'):
    imgs = list()
    for item in os.listdir(img_directory):
        if ('.jpg' in item.lower() or '.png' in item.lower()) \
                and '_iter{}.'.format(desired_iter) in item:
            curr_img = Image.open(os.path.join(img_directory, item))
            # zoom in
            w, h = curr_img.size
            curr_img = curr_img.crop((w / 2 - w / zoom_factor, h / 2 - h / zoom_factor, \
                w / 2 + w / zoom_factor, h / 2 + h / zoom_factor))
            curr_img = curr_img.resize((w, h))
            imgs.append(curr_img)
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=the_figsize)
    side_len = int(math.sqrt(len(imgs)))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=num_visualizations,  # creates grid of axes
                    axes_pad=0.05,  # pad between axes in inch.
                    )

    print('num images:', len(imgs))
    print('num spots in grid:', num_visualizations[0] * num_visualizations[1])

    for ax, im in zip(grid, imgs[:num_visualizations[0] * num_visualizations[1]]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')
    fig.suptitle(the_title)

    # save image
    src_name = img_directory[:-1 if img_directory[-1] == '/' else len(img_directory)].split('/')[-1]
    print('save file name:', src_name)
    print('save dir:', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '{}.pdf'.format(src_name + "_4x4")), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, '{}.png'.format(src_name + "_4x4")), bbox_inches='tight')
    
    # show image
    plt.show()

    return

if __name__ == "__main__":
    plot_visualizations('/data/verifiedanivray/generated_early2', num_visualizations=(4, 4), zoom_factor=4, the_figsize=(8, 8),\
        the_title='ResNet-18, Layer 5: Visualizations')
    plot_visualizations('/data/verifiedanivray/generated_middle2', num_visualizations=(4, 4), zoom_factor=4, the_figsize=(8, 8),\
        the_title='ResNet-18, Layer 11: Visualizations')
    plot_visualizations('/data/verifiedanivray/generated_end2', num_visualizations=(4, 4), zoom_factor=4, the_figsize=(8, 8),\
        the_title='ResNet-18, Layer 17: Visualizations')
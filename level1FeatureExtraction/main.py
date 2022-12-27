
import argparse
from multiprocessing.pool import ThreadPool
import levelOneExtraction
import cv2
import os
import time


def editImage(inputImg, root_dir):
    try:
        imgName = os.path.relpath(inputImg, root_dir)#inputImg.split('/')[-1]
        #print('processing {}'.format(inputImg))
        enhancedPath = inputImg # NOTE: Should have passed enhanced images
        orientPath = os.path.join(orientDir, imgName)
        ridgePath = os.path.join(freqDir, imgName)
        if not os.path.exists(orientPath) or not os.path.exists(ridgePath):
            orient, orientList = levelOneExtraction.findOrientationPhase(enhancedPath)
        if not os.path.exists(ridgePath):
            freq, ridgeCount = levelOneExtraction.findRidgeFlowCount(enhancedPath, orientList)
            os.makedirs(os.path.dirname(ridgePath), exist_ok=True)
            cv2.imwrite(ridgePath, freq)
        if not os.path.exists(orientPath):
            os.makedirs(os.path.dirname(orientPath), exist_ok=True)
            cv2.imwrite(orientPath, orient)
        #print("{} Finished Level One Extraction".format(enhancedPath))
    except Exception as e:
        print(inputImg, "doesn't work")
        pass
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Master Run File')
    parser.add_argument('--src', dest='input', help='Path to folder containing images (should have been enhanced beforehand)', type=str)
    args = parser.parse_args()
    inputPath = args.input
    outputPath = os.path.join(inputPath, '../img_l1_feature_extractions')
    os.makedirs(outputPath, exist_ok=True)
    orientDir = os.path.join(outputPath, 'orient')
    os.makedirs(orientDir, exist_ok=True)
    freqDir = os.path.join(outputPath, 'freq')
    os.makedirs(freqDir, exist_ok=True)

    imageFiles = []
    for root, dirs, files in os.walk(inputPath, topdown=False):
        for name in files:
            relPath = os.path.join(root, name)
            if relPath.endswith('.png') or relPath.endswith('.jpg') or relPath.endswith('.jpeg') or relPath.endswith('.pneg'):
                imageFiles.append(relPath)
                # print(relPath)

    # TODO: find way to preserve folder structure

    print('start level 1 feature extraction')
    
    since = time.time()

    for img in imageFiles:
        editImage(img, inputPath)
    
    elapsed = time.time() - since

    print('took {}m{}s'.format(int(elapsed//60), int(elapsed % 60)))

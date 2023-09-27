import cv2
import csv
import os
import numpy as np
import argparse

def levelTwoExtraction(inputPath, outputPath):
    """this uses the NBIS MINDTCT algorithm to extract minutiae from the original image
    once extracted the minutiae are laid on a blank canvas they could also be overlayed onto the original image
    if so chosen"""
    img = cv2.imread(inputPath, 0)
    imgBase = inputPath[inputPath.rfind("/"):inputPath.rfind(".")]
    outputImageName = outputPath+imgBase
    outputImagePath = outputImageName + ".png"
    cv2.imwrite(outputImagePath, img)
    # call mindtct object with outputImagePath and outputImageName
    #cmd = "./mindtct " + outputImagePath + " " + outputImageName
    #os.system(cmd)
    outputXYT = outputImagePath + ".xyt"
    data = []
    with open(outputXYT) as minutiae:
        reader = csv.reader(minutiae, delimiter=' ')
        for row in reader:
            x,y,t,q = row
            data.append([int(x),int(y),int(t)])
    # put all minutiae on empty numpy array
    canvas = np.zeros(img.shape)
    canvas.fill(255)
    for minut in data:
        x, y, theta = minut
        cv2.rectangle(canvas, (x - 3, y - 3), (x + 3, y + 3), 0, 1)
        cv2.line(canvas, (x, y), (int(x + (8 * np.cos(theta))), int(y + (8 * np.sin(theta)))), 0, 1)
    cv2.imwrite(outputImageName+"_minutiae.png", canvas)
    print("{} Finished Level Two Extraction".format(outputImagePath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameterized_multiple_finger_tester.py')
    parser.add_argument('--dataset_name', '-n', help='Dataset name', \
        const="sd302_split", default="sd302_split", type=str)
    parser.add_argument('--dataset_root', '-d', help='Dataset root', \
        const="/data/therealgabeguo/fingerprint_data/", default="/data/therealgabeguo/fingerprint_data/", type=str)
    parser.add_argument('--mindtct_root', '-o', nargs='?', help='Root directory for output', \
        const="/data/verifiedanivray/mindtct_output/", default="/data/verifiedanivray/mindtct_output/", type=str)

    args = parser.parse_args()
    datasetName = args.dataset_name
    inputRoot = os.path.join(args.dataset_root, datasetName)
    outputRoot = os.path.join(args.mindtct_root, datasetName)
    for root, dirs, files in os.walk(inputRoot):
        for file in files:
            if file.endswith(".png"):
                personId =  os.path.split(root)[1]
                inputPath = os.path.join(root, file)
                outputPath = os.path.join(outputRoot, personId)
                if os.path.exists(os.path.join(outputPath, file + ".xyt")):
                    levelTwoExtraction(inputPath, outputPath)

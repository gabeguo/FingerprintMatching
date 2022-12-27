''' 
generate_minutiae_diagram.py

Given an image file name of a fingerprint and a threshold for the sensitivity of
minutiae, enhances and extracts the level 2 termination and bifurcation features
and returns a diagram showing the minutiae superimposed on the fingerprint

'''

import sys, getopt, os

def main(argv):
    image_file = ''
    threshold = 10

    try:
        opts, args = getopt.getopt(argv,"h?t:i:",["help", "threshold=", "img="])
    except getopt.GetoptError:
        print("Usage: generate_minutiae_diagram.py --img <img_file> [-t threshold(default 10)]")
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '-?', '--help'):
            print("Usage: generate_minutiae_diagram.py --img <img_file> [-t threshold(default 10)]")
            sys.exit(0)
        elif opt in ('--img', '-i'):
            image_file = arg
        elif opt in ('--threshold', '-t'):
            threshold = int(arg)

    if not image_file or not os.path.exists(image_file):
        print("Usage: generate_minutiae_diagram.py --img <img_file> [-t threshold(default 10)]")
        sys.exit(1)

    from FingerprintImageEnhancer import FingerprintImageEnhancer
    from FingerprintFeatureExtractor import FingerprintFeatureExtractor
    import cv2
    import numpy as np

    image_enhancer = FingerprintImageEnhancer()
    feature_extractor = FingerprintFeatureExtractor()

    img = cv2.imread(image_file)
    # Convert to grayscale
    if(len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print ("Enhancing image...")
    enhanced = image_enhancer.enhance(img)
    enhanced_file = "{0}_{2}{1}".format(*os.path.splitext(image_file) + ('enhanced',))

    image_enhancer.save_enhanced_image(os.path.join(os.getcwd(), enhanced_file))
    img = cv2.imread(enhanced_file, 0)

    print("Extracting minutiae...")
    FeaturesTerminations, FeaturesBifurcations = feature_extractor.extractMinutiaeFeatures(img, threshold)

    result_matrix = np.empty(feature_extractor._skel.shape)

    for feat in FeaturesTerminations:
        result_matrix[feat.locX][feat.locY] = 1.0
    for feat in FeaturesBifurcations:
        result_matrix[feat.locX][feat.locY] = 1.0

    from scipy.ndimage import gaussian_filter

    result_matrix = gaussian_filter(result_matrix, sigma=5)

    from matplotlib import pyplot as plt

    print("Superimposing images...")
    minutiae_file = "{0}_{2}{1}".format(*os.path.splitext(image_file) + ('out',))
    plt.axes([0,0,1,1])
    plt.axis("off")
    plt.imsave(minutiae_file, result_matrix, cmap='coolwarm')

    img1 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img2 = cv2.imread(minutiae_file)
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    dst = cv2.addWeighted(img1, 0.5, img2_resized, 0.5, 0)

    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
   main(sys.argv[1:])
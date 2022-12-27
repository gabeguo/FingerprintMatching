import cv2
import numpy as np
import math
import imutils


def overlapping1D(line1, line2):
    return line1[1] >= line2[0] - 5 and line2[1] >= line1[0] - 5


def overlapping2D(box1, box2):
    return overlapping1D(box1["x"], box2["x"]) and overlapping1D(box1["y"], box2["y"])


def crop_minAreaRect(img, rect):
    # Source: https://stackoverflow.com/questions/37177811/
    img = (255 - img)
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, matrix, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), matrix))[0]
    pts[pts < 0] = 0
    img_rot = (255 - img_rot)

    # crop and return
    return img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]


def cropCoordinates(image, inputPath, outputPath):
    img = image
    gray = cv2.cvtColor(cv2.imread(inputPath), cv2.COLOR_BGR2GRAY)
    retval, thresh_gray = cv2.threshold(gray, 230, maxval=255,
                                        type=cv2.THRESH_BINARY_INV)
    
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    graphNodes = {}
    graphEdges = {}
    contourDict = {}
    subGraphList = []
    explored = []
    # build nodes of graph
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = {"x": (x, x + w), "y": (y, y + h)}
        graphNodes[str(contour)] = box
        graphEdges[str(contour)] = []
        contourDict[str(contour)] = contour
    # build edges of graph
    for contour in graphNodes:
        for contour2 in graphNodes:
            if contour != contour2 and overlapping2D(graphNodes[contour2], graphNodes[contour]):
                if contour not in graphEdges[contour2]:
                    graphEdges[contour2].append(contour)
                if contour2 not in graphEdges[contour]:
                    graphEdges[contour].append(contour2)
    # find sub graphs in graph
    for contour in graphNodes:
        exploredMini = [contour]
        if contour not in explored:
            toExplore = [contour]
            explored.append(contour)
            while len(toExplore) != 0:
                nodeToExplore = toExplore.pop(0)
                for node in graphEdges[nodeToExplore]:
                    if node not in exploredMini:
                        toExplore.append(node)
                        exploredMini.append(node)
                        explored.append(node)
        subGraphList.append(exploredMini)
    subGraphList.sort(key=len, reverse=True)
    fingerPrint = subGraphList[0]
    hull = []
    for i in range(len(fingerPrint)):
        hull += contourDict[fingerPrint[i]].tolist()
    box = cv2.minAreaRect(np.asarray(hull))
    return box


def createMask(img):
    # This code takes a binarized image and build a convex hull mask
    arr = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] == 0:
                arr.append([x, y])
    # Create a big Contour from all small contours
    ctr = np.array(arr)
    # Create a Convex Hull from the Contour
    hull1 = cv2.convexHull(ctr).tolist()
    hull2 = []
    for point in hull1:
        hull2.append(point[0])
    hull = np.array(hull2, np.int32)
    hull = hull.reshape((-1, 1, 2))
    # Create a Mask from the Hull
    mask = np.zeros(img.shape[:2])
    cv2.drawContours(mask, [hull], 0, 255, -1)
    # print(np.count_nonzero(mask))
    return mask


def getColor(coordinates, img):
    kernelSize = 3
    inter = int(kernelSize / 2)
    x = coordinates[0]
    y = coordinates[1]
    height, width = img.shape[:2]
    x0 = max(x - inter, 0)
    x1 = min(x + inter + 1, width)
    y0 = max(y - inter, 0)
    y1 = min(y + inter + 1, height)
    img = img[y0:y1, x0:x1]
    avg = []
    for yInner in range(img.shape[0]):
        for xInner in range(img.shape[1]):
            pixel = float(img[yInner, xInner])
            avg.append(pixel)
    if len(avg) != 0:
        return sum(avg) / len(avg)
    else:
        return 255


def findOrient(img):
    numerator = 0
    denominator = 0
    for y in range(1, img.shape[0]):
        for x in range(1, img.shape[1]):
            bx1, bx2 = img[y, x], img[y, x - 1]
            by1, by2 = img[y, x], img[y - 1, x]
            gx = int(bx1) - int(bx2)
            gy = int(by1) - int(by2)
            numerator += (2 * gx * gy)
            denominator += (math.pow(gx, 2) - math.pow(gy, 2))
    percent = 0
    theta = np.pi
    if denominator != 0.00000000:
        base = numerator / denominator
        theta = .5 * np.arctan(base)
        if theta > 0:
            theta -= np.pi / 2
        theta += np.pi
        percent = 1 - (np.count_nonzero(img) / (img.shape[0] * img.shape[1]))
        if theta == np.pi:
            img2 = np.rot90(img)
            numerator = 0
            denominator = 0
            for y in range(1, img2.shape[0]):
                for x in range(1, img2.shape[1]):
                    bx1, bx2 = img2[y, x], img2[y, x - 1]
                    by1, by2 = img2[y, x], img2[y - 1, x]
                    gx = int(bx1) - int(bx2)
                    gy = int(by1) - int(by2)
                    numerator += (2 * gx * gy)
                    denominator += (math.pow(gx, 2) - math.pow(gy, 2))
            theta = np.pi
            if denominator != 0.00000000:
                base = numerator / denominator
                theta = .5 * np.arctan(base)
                if theta > 0:
                    theta -= np.pi / 2
                theta += np.pi
                theta += np.pi / 2
    return theta, 2 * percent


def findOrientationPhase(inputPath):
    """finds the orientation map of the fingerprint"""
    img = cv2.imread(inputPath, 0)
    height, width = img.shape[:2]
    boxSize = 16
    overlap = 16
    orientArray = np.zeros(img.shape)
    orientArray.fill(255)
    i = 0
    orientation = [[[0, 0] for w in range(width)] for h in range(height)]
    for y in range(0, height - boxSize, overlap):
        for x in range(0, width - boxSize, overlap):
            x1, y1 = x + boxSize + 1, y + boxSize + 1
            angle, strength = findOrient(img[y:y1, x:x1])
            orientation[y + 8][x + 8] = [angle, strength]
            if strength > 0:
                yd = int(8 * np.sin(angle) * strength)
                xd = int(8 * np.cos(angle) * strength)
                x1, x2 = x + xd + int(boxSize / 2), x - xd + int(boxSize / 2)
                y1, y2 = y + yd + int(boxSize / 2), y - yd + int(boxSize / 2)
                cv2.line(orientArray, (x2, y2), (x1, y1), color=0, thickness=1)
                i += 1
    return orientArray, orientation


def getRidgeCount(img, center, orientation):
    ridges = []
    ridgeCount = 0
    x, y = center
    angle, strength = orientation
    theta = angle / 2 / np.pi * 360
    block = img[max(y - 16, 0):min(y + 16, img.shape[0]), max(x - 16, 0):min(x + 16, img.shape[1])]
    if block.shape[0] == 32 and block.shape[1] == 32:
        rotated = imutils.rotate(block, theta)
        rotatedCrop = rotated[:, 4:20]
        rotatedCrop = np.rot90(rotatedCrop)
        res, rotatedCrop = cv2.threshold(rotatedCrop, 127, 255, cv2.THRESH_BINARY)
        for y in range(rotatedCrop.shape[0]):
            if np.count_nonzero(rotatedCrop[y, :]) < 32:
                blk = []
                wte = []
                current = rotatedCrop[y, 0]
                count = 1
                for x in range(1, rotatedCrop.shape[1]):
                    if current == rotatedCrop[y, x]:
                        count += 1
                    else:
                        if current == 0:
                            blk.append(count)
                            count = 1
                            current = 255
                        else:
                            wte.append(count)
                            count = 1
                            current = 0
                ridges.append(len(blk))
        if len(ridges) > 0:
            ridgeCount = sum(ridges) / len(ridges)
        if ridgeCount % 1 >= .5:
            ridgeCount = int(ridgeCount) + 1
        else:
            ridgeCount = int(ridgeCount)
    return ridgeCount


def findRidgeFlowCount(inputPath, orientation):
    """finds the ridge frequency at different locations in the fingerprint
     this is based off of the orientation map"""
    img = cv2.imread(inputPath, 0)
    blockSize = 16  # must be multiple of 4
    overlap = 16
    height, width = img.shape[:2]
    counts = []
    ridgeArray = np.zeros((height, width), dtype=np.uint8)
    for y0 in range(0, height - blockSize, overlap):
        for x0 in range(0, width - blockSize, overlap):
            xc = x0 + int(blockSize / 2)
            yc = y0 + int(blockSize / 2)
            count = getRidgeCount(img, (xc, yc), orientation[yc][xc])
            if count != 0:
                counts.append(count)
            for ycor in range(y0, y0 + blockSize):
                for xcor in range(x0, x0 + blockSize):
                    ridgeArray[ycor, xcor] = count * 51
    average = 0
    if len(counts) > 0:
        average = sum(counts) / len(counts)
    return ridgeArray, average

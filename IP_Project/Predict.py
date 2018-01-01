import cv2
import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import segmentation


def extractHog(image):
    image = cv2.resize(image, (32, 32))
    winSize = (32, 32)  # Image Size
    cellSize = (4, 4)  # Size of one cell
    blockSizeInCells = (2, 2)  # will be multiplies by No of cells

    blockSize = (blockSizeInCells[1] * cellSize[1], blockSizeInCells[0] * cellSize[0])
    blockStride = (cellSize[1], cellSize[0])
    nbins = 12  # Number of orientation bins
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)  #
    h = hog.compute(image)
    h = h.flatten()
    return h.flatten()


def showImage(image):
    b, g, r = cv2.split(image)  # get b,g,r
    image = cv2.merge([r, g, b])  # switch it to rgb
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('Image')
    ax.axis('off')
    ax.set_adjustable('box-forced')
    plt.show()
    cv2.waitKey(0)                 # Waits forever for user to press any key
    cv2.destroyAllWindows()        # Closes displayed windows


image1Path = 'C:\Users\Sherif\Desktop\IP_Project\test8.jpg'
image1 = cv2.imread(image1Path)

image_g = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(image_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
it = int(ret / 2)

sure_bg = cv2.dilate(opening, kernel, iterations=25)
sure_fg = cv2.erode(sure_bg, kernel, iterations=25)

unknown = cv2.subtract(sure_bg, sure_fg)

ret2, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(image1, markers)
image1[markers == -1] = [255, 0, 0]

outputImages = list()

for region in range(ret2 - 1):
    image_o = np.copy(image1)
    image_o[markers != region + 2] = [0, 0, 0]
    gray = cv2.cvtColor(image_o, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    image2 = image1[y:y + h, x:x + w]
    outputImages.append(image2)

for i in range(len(outputImages)):
    image1 = outputImages[i]
    showImage(image1)
    image1Features = extractHog(image1)
    testImages = list()
    testImages.append(image1Features)

    NN_model = joblib.load('C:\Users\Sherif\Desktop\IP_Project\Model')
    outputMat = NN_model.predict_proba(testImages)
    ind = int(np.argmax(outputMat[0]) / 4)

    if ind == 0:
        print('\nThe received bill was 5 pounds')
    elif ind == 1:
        print('\nThe received bill was 10 pounds')
    elif ind == 2:
        print('\nThe received bill was 20 pounds')
    elif ind == 3:
        print('\nThe received bill was 50 pounds')
    elif ind == 4:
        print('\nThe received bill was 100 pounds')
    else:
        print('\nThe received bill was 200 pounds')
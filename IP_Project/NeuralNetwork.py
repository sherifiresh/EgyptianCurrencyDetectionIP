from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from imutils import paths
import numpy as np
from sklearn.neural_network import MLPClassifier
import cv2
import os

path = 'C:\\Users\\Ahmed\\Desktop\\IP_Project\\Notes'
testPaths = 'C:\\Users\\Ahmed\\Desktop\\IP_Project\\Test'
t_size = 0.01  # test_Size
r_state = 42   # Random Seed
sizeAfterResize = (32, 32)  # Size of images after resize
jobs = 1
kNeighbours = 7
imagePaths = list(paths.list_images(path))
testImagePaths = list(paths.list_images(testPaths))
features = []
labels = []
testImages = list()


def extractHog(image):
    image = cv2.resize(image, sizeAfterResize)
    winSize = (32, 32)  # Image Size
    cellSize = (4, 4)  # Size of one cell
    blockSizeInCells = (2, 2)  # will be multiplies by No of cells

    blockSize = (blockSizeInCells[1] * cellSize[1], blockSizeInCells[0] * cellSize[0])
    blockStride = (cellSize[1], cellSize[0])
    nbins = 12  # Number of orientation bins
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    h = hog.compute(image)
    h = h.flatten()
    return h.flatten()


for (i, imagePath) in enumerate(imagePaths):
    # Load image and labels
    # path format: path/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    sampleFeatures = extractHog(image)

    features.append(sampleFeatures)

    labels.append(label)


for (j, imagePath) in enumerate(testImagePaths):
    image = cv2.imread(imagePath)

    imageFeatures = extractHog(image)

    testImages.append(imageFeatures)


features = np.array(features)

labels = np.array(labels)

(trainFeatures, testImages, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=t_size, random_state=r_state)

NN_model = MLPClassifier(hidden_layer_sizes=(500, ), solver='sgd', alpha=1e-5, random_state=100, max_iter=4000, verbose=1)

NN_model.fit(trainFeatures, trainLabels)

print('\nTraining finished.')

acc = NN_model.score(testImages, testLabels)

print("\n[NN] Accuracy: {:.2f}%".format(acc * 100))

joblib.dump(NN_model, 'C:\Users\Sherif\Desktop\IP_Project\Model')

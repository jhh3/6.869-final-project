# Author: John Holliman
from scipy import ndimage as ndi
from scipy.misc import imread, imresize
from skimage.filters import gabor_kernel
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from time import time
import numpy as np
import os

################################################################################
# Options

im_dir = os.getcwd() + "/images/im-jit"
#im_dir = os.getcwd() + "/images/im-reg"
fundus_data_file = os.getcwd() + "/data/fundus_dataset.svm"
target_names = np.array(['healthy', 'glaucoma', 'diabetic retinopathy'])
scale_factor = 0.1
test_size = 0.25
gamma = 0.001
kernel = 'poly'
degree = 3

################################################################################
# Prepare filter bank kernels

kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, 
                                          theta=theta,
                                          sigma_x=sigma, 
                                          sigma_y=sigma))
            kernels.append(kernel)


def compute_feats(image, kernels):
    feats = np.zeros((16, 198*297), dtype=np.double)
    #feats = np.zeros((16, 233*350), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        filtered = np.reshape(filtered, (198*297))
        #filtered = np.reshape(filtered, (233*350))
        feats[k] = filtered
    return feats

################################################################################
# Load dataset (create it if necessary)

print("Loading dataset")
t0 = time()
fundus_dataset = {}
image_data = []
spat_data = []
target = []
for fname in os.listdir(im_dir):
    # Load Image and transform to a 1D numpy array.
    im_fundus = imread(im_dir + "/" + fname, flatten=True)
    im_fundus = imresize(im_fundus, scale_factor)
    im_fundus = np.array(im_fundus, dtype=np.float64) / 255
    w, h = original_shape = tuple(im_fundus.shape)
    print("\t" + str(original_shape))

    # extract features
    fun_feats = compute_feats(im_fundus, kernels)
    fun_feats = np.reshape(fun_feats, (16*198*297))
    #fun_feats = np.reshape(fun_feats, (16*233*350))
    spat_data.append(fun_feats)

    im_fundus = np.reshape(im_fundus, (w * h))
    image_data.append(im_fundus)
    if 'h' in fname:
        target.append(0)
    elif 'g' in fname:
        target.append(1)
    else: # 'dr'
        target.append(2)
    print("\t" + fname)

spat_data = np.array(spat_data)
image_data = np.array(image_data)

print("Done in %0.3fs.\n" % (time() - t0))

################################################################################
# Extract features from the data and create the best estimator

print("Fitting the classifier to the training set")
t2 = time()

pca = PCA(n_components=3)
X_imtrans = pca.fit(image_data).transform(image_data)
X_gbtrans = pca.fit(spat_data).transform(spat_data)
X_new = np.concatenate((X_imtrans, spat_data), axis=1)

################################################################################
# Split data into a training and a test set

X_train, X_test, y_train, y_test = train_test_split(X_new, target,
                                                    test_size=test_size,
                                                    random_state=42)

svm = SVC(kernel='linear', gamma=gamma)
neigh = KNeighborsClassifier(n_neighbors=5)
classifier = svm
#classifier = neigh

param_grid = {'gamma':[0.001, 0.0005, 0.0001], 
              'kernel':['linear', 'poly', 'rbf']}

#param_grid = {'n_neighbors':[3, 4, 5, 6, 7, 8]}

grid_search = GridSearchCV(classifier, param_grid=param_grid, verbose=10)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)
classifier = grid_search.best_estimator_
classifier = svm

################################################################################
# Train SVM classification model

classifier.fit(X_train, y_train)
print("Done in %0.3fs." % (time() - t2))
print("")

################################################################################
# Predict value on test set 

predicted = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, classification_report(y_test, predicted)))
print("Confusion matrix:\n%s" % confusion_matrix(y_test, predicted))


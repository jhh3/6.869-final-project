# Author: John Holliman
from scipy import ndimage as ndi
from scipy.misc import imread, imresize
from skimage.filters import gabor_kernel
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from time import time
import numpy as np
import os

################################################################################
# Options

#im_dir = os.getcwd() + "/images/im-jit"
im_dir = os.getcwd() + "/images/im-reg"
#im_dir = os.getcwd() + "/images/im-unp"
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
    #feats = np.zeros((16, 198*297), dtype=np.double)
    feats = np.zeros((16, 233*350), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        #filtered = np.reshape(filtered, (198*297))
        filtered = np.reshape(filtered, (233*350))
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
    #fun_feats = np.reshape(fun_feats, (16*198*297))
    fun_feats = np.reshape(fun_feats, (16*233*350))
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

pca = PCA(n_components=50)
X_imtrans = pca.fit(image_data).transform(image_data)
pca = PCA(n_components=50)
X_gbtrans = pca.fit(spat_data).transform(spat_data)
#X_new = np.concatenate((X_imtrans, spat_data), axis=1)
#X_new = np.concatenate((spat_data, X_imtrans), axis=1)
#X_new = X_imtrans
#X_new = X_gbtrans

################################################################################
# Split data into a training and a test set

#class1 = SVC(kernel='linear', gamma=gamma, probability=True)
#class2 = SVC(kernel='linear', gamma=gamma, probability=True)
#class3 = SVC(kernel='linear', gamma=gamma)
class1 = KNeighborsClassifier(n_neighbors=5)
class2 = KNeighborsClassifier(n_neighbors=5)
class3 = KNeighborsClassifier(n_neighbors=5)

n_samples = len(X_imtrans)
X_imtrain = X_imtrans[:int(n_samples * (1 - test_size))]
y_train = target[:int(n_samples * (1 - test_size))]
X_imtest = X_imtrans[int(n_samples * (1 - test_size)):]
y_test = target[int(n_samples * (1 - test_size)):]

class1.fit(X_imtrain, y_train)
impred = class1.predict_proba(X_imtest)

X_trans1 = class1.predict_proba(X_imtrain)

X_gbtrain = X_gbtrans[:int(n_samples * (1 - test_size))]
X_gbtest = X_gbtrans[int(n_samples * (1 - test_size)):]

class2.fit(X_gbtrain, y_train)
gbpred = class2.predict_proba(X_gbtest)

X_trans2 = class2.predict_proba(X_gbtrain)

x_newnew = np.concatenate((X_trans1, X_trans2), axis=1)

################################################################################
# Train SVM classification model

class3.fit(x_newnew, y_train)
print("Done in %0.3fs." % (time() - t2))
print("")

################################################################################
# Predict value on test set 

predicted = class3.predict(np.concatenate((impred, gbpred), axis=1))

print("Classification report for classifier %s:\n%s\n"
      % (classifier, classification_report(y_test, predicted)))
print("Confusion matrix:\n%s" % confusion_matrix(y_test, predicted))


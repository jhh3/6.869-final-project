# TODO : write doc
# Author: John Holliman
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.cross_validation import train_test_split
#from sklearn.svm import SVC
from sknn.mlp import Classifier, Layer, Convolution
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from scipy.misc import imread, imresize
from time import time
import os

################################################################################
# Options
im_dir = os.getcwd() + "/images"
fundus_data_file = os.getcwd() + "/data/fundus_dataset.svm"
target_names = np.array(['healthy', 'glaucoma', 'diabetic retinopathy'])
scale_factor = 0.1
test_size = 0.2
gamma = 0.0005
kernel = 'poly'
degree = 3

################################################################################
# Load dataset (create it if necessary)
print("Loading dataset")
t0 = time()
if not os.path.isfile(fundus_data_file):
    t1 = time()
    print("\tGenerating dataset")
    fundus_dataset = {}
    image_data = []
    target = []
    for fname in os.listdir(im_dir):
        # Load Image and transform to a 1D numpy array.
        im_fundus = imread(im_dir + "/" + fname)
        im_fundus = imresize(im_fundus, scale_factor)
        im_fundus = np.array(im_fundus, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(im_fundus.shape)
        print("\t" + str(original_shape))
        assert d == 3
        im_fundus = np.reshape(im_fundus, (w * h * d))
        image_data.append(im_fundus)
        if 'h' in fname:
            target.append(0)
        elif 'g' in fname:
            target.append(1)
        else: # 'dr'
            target.append(2)
        print("\t" + fname)
    dump_svmlight_file(np.array(image_data), np.array(target), fundus_data_file)
    print("\tDone in %0.3fs." % (time() - t1))

image_data, target = fundus_dataset = load_svmlight_file(fundus_data_file)
print("Done in %0.3fs." % (time() - t0))
print("Total dataset size:")
print("n_samples: %d" % target.shape[0])
print("n_features: %d" % image_data.shape[1])
print("n_classes: %d" % target_names.shape[0])
print("")

################################################################################
# Split data into a training and a test set

X_train, X_test, y_train, y_test = train_test_split(image_data, target,
        test_size=test_size, random_state=42)

################################################################################
# Extract features from the data

# TODO

################################################################################
# Train NN

x_tnew = []
for im in X_train:
    x_tnew.append(im.toarray().reshape((233,350,3)))

x_testnew = []
for im in X_test:
    x_testnew.append(im.toarray().reshape((233,350,3)))

X_train = np.array(x_tnew)
X_test = np.array(x_testnew)

cnn = Classifier(
    layers=[
            Convolution('Rectifier', channels=12, kernel_shape=(3, 3),
                border_mode='full', pool_shape=(2,2), pool_type='max'),
            Layer('Softmax')],
    learning_rate=0.001, 
    n_iter=25,
    verbose=True)
mm_scaler = MaxAbsScaler()
nn_class= Classifier( 
    layers=[
        Layer("Maxout", units=100, pieces=2), 
        Layer("Softmax")], 
    learning_rate=0.001, 
    n_iter=25,
    verbose=True)

pipeline = Pipeline([
        ('min/max scaler', mm_scaler),
        ('neural network', nn_class)])

################################################################################
# Train SVM classification model

print("Fitting the classifier to the training set")
t2 = time()
classifier = cnn
classifier.fit(X_train, y_train)
print("Done in %0.3fs." % (time() - t2))
print("")

################################################################################
# Predict values on test set and plot predictions

predicted = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, classification_report(y_test, predicted)))
print("Confusion matrix:\n%s" % confusion_matrix(y_test, predicted))

for index, prediction in enumerate(predicted):
    plt.subplot(3, 3, index + 1)
    plt.axis('off')
    image = X_test[index].toarray().reshape((233, 350, 3)) * 255
    #image = X_test[index] * 255
    image = np.array(image, dtype=np.uint8)
    plt.imshow(image)
    plt.title('Prediction: %s' % target_names[int(prediction)])

plt.show()

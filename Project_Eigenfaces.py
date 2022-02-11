# Eigenfaces
# Pillow installation required

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Chart #1 (Far left chart in report)

# Load human face images
dataset1 = fetch_lfw_people(min_faces_per_person=50)

x1 = dataset1.data
y1 = dataset1.target
names_1 = dataset1.target_names
_, h1, w1 = dataset1.images.shape

# split into a training and testing set
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.3)

# Compute Principal Component Analysis (PCA)
N_1 = 50
pca_1 = PCA(n_components=N_1, whiten=True).fit(x_train1)

# Apply Principal Component Analysis (PCA)
x_train_PCA_1 = pca_1.transform(x_train1)
x_test_PCA_1 = pca_1.transform(x_test1)

# Train neural network
classifier1 = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(x_train_PCA_1, y_train1)

y_predictor1 = classifier1.predict(x_test_PCA_1)

# Results pertaining to accuracy of recognition
print(classification_report(y_test1, y_predictor1, target_names=names_1))

# Chart #2 (Middle chart in report)

dataset2 = fetch_lfw_people(min_faces_per_person=50)

x2 = dataset2.data
y2 = dataset2.target
names_2 = dataset2.target_names
_, h2, w2 = dataset2.images.shape

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.8)

N_2 = 50
pca_2 = PCA(n_components=N_2, whiten=True).fit(x_train2)

x_train_PCA_2 = pca_2.transform(x_train2)
x_test_PCA_2 = pca_2.transform(x_test2)

classifier2 = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(x_train_PCA_2, y_train2)

y_predictor2 = classifier2.predict(x_test_PCA_2)

print(classification_report(y_test2, y_predictor2, target_names=names_2))

# Chart #3 (Far right chart in report)

dataset3 = fetch_lfw_people(min_faces_per_person=80)

x3 = dataset3.data
y3 = dataset3.target
names_3 = dataset3.target_names
_, h3, w3 = dataset3.images.shape

x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.3)

N_3 = 80
pca_3 = PCA(n_components=N_3, whiten=True).fit(x_train3)

x_train_PCA_3 = pca_3.transform(x_train3)
x_test_PCA_3 = pca_3.transform(x_test3)

classifier3 = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(x_train_PCA_3, y_train3)

y_predictor3 = classifier3.predict(x_test_PCA_3)

print(classification_report(y_test3, y_predictor3, target_names=names_3))
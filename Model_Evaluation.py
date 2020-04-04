#Model Evalution
import pandas as pd
import numpy as np
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

model = load_model(r'Fashion_Model.hdf5',compile = False)
fashion_test_df = pd.read_csv('fashion-mnist_test.csv', sep = ',')
testing = np.array(fashion_test_df, dtype = 'float32')
X_test = testing[:, 1:]/255
y_test = testing[:, 0]
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))

#Result Visualization
predicted_classes = model.predict_classes(X_test)
W = 5
L = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()
for i in np.arange(0, W * L):
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i],y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(hspace = 0.5)
plt.show()

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True)

#Classification Report
from sklearn.metrics import classification_report

num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test,predicted_classes, target_names = target_names))


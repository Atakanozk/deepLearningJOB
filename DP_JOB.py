# -*- coding: utf-8 -*-
"""
@author: ataka
"""


import pandas as pd 
import numpy as np 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
sw_english = stopwords.words("english")
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import time 
import nvidia_smi
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None)#checking is tensorflow use gpu or cpu /// True ----> gpu
tf.config.experimental.list_physical_devices('GPU')#[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 

print(device_lib.list_local_devices())

#nvidia results
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

#Text cleaning
dataset = dataset[["description","fraudulent"]]
dataset.isnull().values.any()#There is NO na value
dataset = dataset.dropna()
dataset.reset_index(inplace = True)
dataset = dataset.drop(["index"],axis=1)
corpus = []#this list will include all different jobs words
for i in range(0, 17879):
    description = re.sub('[^a-zA-Z]', ' ', dataset['description'][i])
    description = description.replace("'","")
    description = description.replace("|","")
    description = description.replace("-","")
    description = description.replace(".","")
    description = description.replace("(","")
    description = description.replace(")","")
    description = description.lower()
    description = description.split()
    ps = PorterStemmer()
    description = [ps.stem(word) for word in description if not word in set(stopwords.words("english"))]
    description = " ".join(description)
    corpus.append(description)
    
#bag of word model 
cv = CountVectorizer(max_features = 47000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

#spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Modelling
"""
citation:"Artificial Intelligence for Humans, Volume 3: Deep Learning and Neural Networks" ISBN: 1505714346
The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer.
"""
model = tf.keras.models.Sequential()
#input layer
model.add(tf.keras.layers.Dense(units = 12, activation = "relu"))
#hidden layer
model.add(tf.keras.layers.Dense(units = 9, activation = "relu"))
#output layer
model.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))

model.compile(optimizer = "adam" ,loss = "binary_crossentropy" ,metrics = ["accuracy"])
tann= time.clock()
model.fit(X_train,y_train, batch_size = 512, epochs = 100)
predict = model.predict(X_test)
anntime = time.clock() - tann 
print("Execution Time in seconds =", anntime)	
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print(f'mem: {mem_res.used / (1024**2)} (GiB)') # usage in GiB
print(f'mem: {100 * (mem_res.used / mem_res.total):.3f}%')
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

predict = np.rint(predict)
cm = confusion_matrix(y_test, predict)
print(cm)
plt.style.use("classic")
plot_confusion_matrix(cm, classes=["Real", "Fraudulent"],
                      title="ANN")
"""
Citiation
---------
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

"""

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#  %97 accuracy

#Model cnn 
embedding_train, embedding_test, y_embedding_train, y_embedding_test = train_test_split(corpus, y, test_size = 0.2, random_state = 0)

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=47000)
tokenizer.fit_on_texts(embedding_train)
X_train_token = tokenizer.texts_to_sequences(embedding_train)
X_test_token = tokenizer.texts_to_sequences(embedding_test)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(embedding_train[2])
print(X_train_token[2])


maxlen = 100

X_train_token = pad_sequences(X_train_token, padding='post', maxlen=maxlen)
X_test_token = pad_sequences(X_test_token, padding='post', maxlen=maxlen)

print(X_train_token[0, :])

embedding_dim = 50
tcnn= time.clock()
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
cnn.add(tf.keras.layers.Conv1D(128, 5, activation="relu"))
cnn.add(tf.keras.layers.GlobalMaxPooling1D())
cnn.add(tf.keras.layers.Dense(10, activation="relu"))
cnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))
cnn.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
cnn.summary()
history = cnn.fit(X_train_token, y_embedding_train,
                    epochs=15,
                    validation_data=(X_test_token, y_embedding_test),
                    batch_size=512)
predict_cnn = cnn.predict(X_test_token)
cnntime = time.clock() - tcnn 
print("Execution Time in seconds =", cnntime)	
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print(f'mem: {mem_res.used / (1024**2)} (GiB)') # usage in GiB
print(f'mem: {100 * (mem_res.used / mem_res.total):.3f}%')
loss, accuracy = cnn.evaluate(X_test_token, y_embedding_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

predict_cnn = np.rint(predict_cnn)
cm = confusion_matrix(y_embedding_test, predict_cnn)
print(cm)
plot_confusion_matrix(cm, classes=["Real", "Fraudulent"],
                      title="CNN")

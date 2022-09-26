#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from tensorflow.keras.regularizers import l2


# In[2]:


train_dir = r"C:\Users\Lenovo\Desktop\Thesis\main folder\dataset\New folder\train/"
test_dir = r"C:\Users\Lenovo\Desktop\Thesis\main folder\dataset\New folder\test/"


# In[3]:


train_datagen = ImageDataGenerator(rescale=(1/255.),shear_range = 0.2,zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory(directory = train_dir,target_size=(64,64),
                                                batch_size=32,
                                                class_mode = "binary")
test_datagen = ImageDataGenerator(rescale=(1/255.))
test_set = test_datagen.flow_from_directory(directory = test_dir,target_size=(64,64),
                                                batch_size=32,
                                                class_mode = "binary")


# In[4]:


model = Sequential()
model.add(Conv2D(filters = 32, padding = "same",activation = "relu",kernel_size=3, strides = 2,input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2,2),strides = 2))

model.add(Conv2D(filters = 32, padding = "same",activation = "relu",kernel_size=3))
model.add(MaxPool2D(pool_size=(2,2),strides = 2))

model.add(Flatten())
model.add(Dense(128,activation="relu"))

#Output layer
model.add(Dense(1,kernel_regularizer=l2(0.01),activation = "linear"))


# In[5]:


model.compile(optimizer = 'adam', loss = "hinge", metrics = ['accuracy'])


# In[6]:


# model.add(Dense(number_of_classes,kernel_regularizers = l2(0.01),activation= "softmax"))
# model.compile(optimizer="adam",loss="squared_hinge", metrics = ['accuracy'])


# In[7]:


history = model.fit(x = training_set, validation_data = test_set, epochs=5)


# In[8]:


from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('train acc')
plt.xlabel('val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[9]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('train loss')
plt.xlabel('val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# In[10]:


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)

predictions = model.predict_generator(test_set , steps=test_steps_per_epoch)

# Get most likely class
predicted_classes = [1 * (x[0]>=0.5) for x in predictions]

# 2.Get ground-truth classes and class-labels
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys()) 
import sklearn
# 3. Use scikit-learn to get statistics
from sklearn.metrics import confusion_matrix,classification_report

print(class_labels)

print(confusion_matrix(test_set.classes, predicted_classes))

report = sklearn.metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report) 


# In[ ]:





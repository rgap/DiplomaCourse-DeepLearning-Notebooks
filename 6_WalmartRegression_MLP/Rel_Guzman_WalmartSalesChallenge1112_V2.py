#!/usr/bin/env python
# coding: utf-8

# # Regresión Polinomial

# Predicción de las ventas de la siguiente semana para Walmart.
# Base de datos de 45 tiendas tipo A, B. Varían en tamaño

# In[1]:


from numpy.random import seed
seed(1)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Lectura del los datos del 2011

# In[3]:


walmart2011 = pd.read_csv('walmart2011e.csv')
walmart2012 = pd.read_csv('walmart2012e.csv')


# In[4]:


# Visualiza los primeros registros
walmart2011.head()


# In[5]:


# Visualiza los nombres de las variables
walmart2011


# In[8]:


# Concatenación de los dataframes
walmart = pd.concat([walmart2011, walmart2012])


# ## Pre-procesamiento de los datos

# In[10]:


#Estandarización
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(walmart)
walmartStd = scaler.transform(walmart)
walmartStd


# In[11]:


# Grafica de las ventas
plt.scatter(walmartStd[0:50, 0], walmartStd[0:50, -1])
plt.xlabel('tiempo')
plt.xlabel('ventas')
plt.show()


# Creación de conjuntos de entrenamiento y test

# In[14]:


caracteristicas = walmartStd[:,:-1]
target = walmartStd[:, -1]
caracteristicas.shape


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(caracteristicas, target, test_size=0.1)


# In[17]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# Generación del modelo

# In[20]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers, backend, callbacks
from sklearn.metrics import r2_score


# In[21]:


backend.clear_session()

# Modelo
modelo1 = Sequential()
modelo1.add(Dense(22, activation='sigmoid', input_shape=(11,)))
modelo1.add(Dense(1, activation='linear'))
modelo1.summary()
# Fin del modelo


# In[22]:


#Define el optimizador Adam
Adam = optimizers.Adam(learning_rate = 0.001)

#Compila el modelo
modelo1.compile(optimizer=Adam, loss='mean_squared_error', metrics=['mse'])

#Checkpointer para guardar el mejor modelo (en alguna epoca)
checkpointer = callbacks.ModelCheckpoint('walmart1.h5', monitor = 'val_mse', save_best_only = True,
                                         mode = 'min')


# In[24]:


#Entrena el modelo
M = modelo1.fit(x_train, y_train, validation_data=(x_test, y_test),
               epochs=1000, batch_size=10,
               verbose = False, callbacks = [checkpointer])


# In[27]:


# Evaluación del modelo
# Grafica del error
plt.plot(M.history['loss'], label='train')
plt.plot(M.history['val_loss'], label='test')
plt.legend(loc='best')
plt.show()

# Evaluación en entrenamiento
plt.scatter(x_train[:, 0], y_train, label='real')
plt.scatter(x_train[:, 0], modelo1.predict(x_train), label='pred')
plt.legend(loc='best')
plt.show()
print('r2: ', r2_score(y_train, modelo1.predict(x_train)))
print('error (mse): ', modelo1)

# Evaluación en validación
plt.scatter(x_test[:, 0], y_test, label='real')
plt.scatter(x_test[:, 0], modelo1.predict(x_test), label='pred')
plt.legend(loc='best')
plt.show()
print('r2: ',r2_score(y_test, modelo1.predict(x_test)))
print('error (mse): ', modelo1)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ---

# ---

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image


# In[3]:


# Rutas proporcionadas
labels_path = '/datasets/faces/labels.csv'
images_path = '/datasets/faces/final_files/'

try:
    labels = pd.read_csv(labels_path)
    
    # 1. Tamaño del conjunto de datos
    dataset_size = labels.shape
    print(f"Tamaño del conjunto de datos: {dataset_size}")
    
    # Primeras filas del conjunto de datos
    print("\nPrimeras filas del conjunto de datos:")
    print(labels.head())
    
    # 2. Distribución de edad
    plt.figure(figsize=(10, 6))
    plt.hist(labels['real_age'], bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Distribución de edades en el conjunto de datos')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.show()
    
    # 3. Imprimir de 10 a 15
    sample_images = labels.sample(n=10, random_state=42)  # Seleccionar 10 imágenes aleatorias
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (index, row) in enumerate(sample_images.iterrows()):
        image_path = os.path.join(images_path, row['file_name'])
        try:
            image = Image.open(image_path)
            axes[i].imshow(image)
            axes[i].set_title(f"Edad: {row['real_age']}")
            axes[i].axis('off')
        except Exception as e:
            axes[i].set_title("No cargó")
            axes[i].axis('off')
            print(f"No se pudo cargar la imagen: {image_path}. Error: {e}")
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(f"No se pudo cargar el archivo: {e}")
except Exception as e:
    print(f"Error durante el análisis: {e}")


# ## EDA

# In[4]:


# Verificar valores nulos
print("\nValores nulos por columna:")
print(labels.isnull().sum())

# Resoluciones de imágenes
image_dimensions = []

for _, row in labels.iterrows():
    image_path = os.path.join(images_path, row['file_name'])
    try:
        with Image.open(image_path) as img:
            image_dimensions.append(img.size)  # Agregar dimensiones (ancho, alto)
    except Exception as e:
        print(f"No se pudo cargar la imagen: {image_path}. Error: {e}")

# Convertir a DataFrame para análisis
dimensions_df = pd.DataFrame(image_dimensions, columns=['Width', 'Height'])
print("\nEstadísticas de resoluciones de imágenes:")
print(dimensions_df.describe())

# Distribución por edad
age_distribution = labels['real_age'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
age_distribution.plot(kind='bar', color='green', edgecolor='black', alpha=0.7)
plt.title('Distribución exacta de edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()


# In[5]:


# Verificar si hay otras columnas para análisis adicionales
print("\nColumnas disponibles en el conjunto de datos:")
print(labels.columns)

# 1. Calcular correlaciones si existen más columnas numéricas
if labels.select_dtypes(include=['float64', 'int64']).shape[1] > 1:
    print("\nMatriz de correlación entre variables numéricas:")
    print(labels.corr())
    
    # Visualizar matriz de correlación si hay suficientes columnas numéricas
    plt.figure(figsize=(8, 6))
    corr_matrix = labels.corr()
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
    plt.title("Matriz de correlación")
    plt.show()

# 2. Si existe una columna de género u otra categórica
if 'gender' in labels.columns:
    print("\nDistribución por género:")
    print(labels['gender'].value_counts())
    
    plt.figure(figsize=(6, 4))
    labels['gender'].value_counts().plot(kind='bar', color='purple', edgecolor='black')
    plt.title('Distribución por género')
    plt.xlabel('Género')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=0)
    plt.show()

# 3. Visualización de densidad de edad
plt.figure(figsize=(10, 6))
labels['real_age'].plot(kind='kde', color='green')
plt.title('Densidad de edad')
plt.xlabel('Edad')
plt.ylabel('Densidad')
plt.grid(True)
plt.show()

# Matriz de correlación
if labels.select_dtypes(include=['float64', 'int64']).shape[1] > 1:
    print("\nMatriz de correlación entre variables numéricas:")
    print(labels.corr())

# Duplicados
duplicados = labels.duplicated().sum()
print(f"\nNúmero de registros duplicados: {duplicados}")

# Balance del conjunto de datos
bins = [0, 18, 30, 45, 60, 100]
labels['age_group'] = pd.cut(labels['real_age'], bins=bins, labels=['0-18', '19-30', '31-45', '46-60', '61+'])

# Balance del conjunto de datos por rangos de edad
print("\nBalance de datos por rangos de edad:")
print(labels['age_group'].value_counts(normalize=True) * 100)

# Visualización del balance por rangos de edad
plt.figure(figsize=(10, 6))
labels['age_group'].value_counts().sort_index().plot(kind='bar', color='orange', edgecolor='black')
plt.title('Distribución por rangos de edad')
plt.xlabel('Rangos de edad')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.show()


# In[13]:


import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# In[14]:


def load_data(path, subset='training'):
    labels = pd.read_csv(os.path.join(path, 'labels.csv'))

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=15
    )

    data_gen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=os.path.join(path, 'final_files/'),
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset=subset,
        seed=12345
    )

    return data_gen_flow


# In[15]:


def create_model(input_shape):
    backbone = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
    backbone.trainable = False

    model = Sequential([
        backbone,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='relu')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model


# In[16]:


def train_model(model, train_data, test_data, epochs=10):
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        validation_steps=test_data.samples // test_data.batch_size,
        verbose=2
    )
    return model, history


# In[17]:


# prepara un script para ejecutarlo en la plataforma GPU

init_str = """
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
"""

import inspect

with open('run_model_on_gpu.py', 'w') as f:
    
    f.write(init_str)
    f.write('\n\n')
        
    for fn_name in [load_train, load_test, create_model, train_model]:
        
        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')


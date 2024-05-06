import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Cargar el modelo
model = tf.keras.models.load_model(r'C:\Users\antdu\OneDrive\Escritorio\Master\Técnicas de Desarrollo Avanzado de Aplicaciones Big Data\Actividad 2\modelo.h5')

# Función para preprocesar la imagen cargada por el usuario
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Función para realizar la predicción
def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

# Configuración de la página de la aplicación
st.title('Detector de perros y gatos')
st.write('Sube una imagen para determinar si es un perro o un gato')

# Seleccionar una imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen cargada por el usuario
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen aportada', use_column_width=True)

    # Realizar la predicción
    prediction = predict(image)

    # Mostrar el resultado de la predicción
    if prediction[0][0] > 0.5:
        st.write('La imagen es de un perro')
    else:
        st.write('La imagen es de un gato')

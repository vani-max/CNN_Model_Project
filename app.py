import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_utils import get_gradcam_ultra_manual
from tensorflow.keras import layers, models

def create_model():
    # Replace this with the ACTUAL layers from your Colab training
    model = models.Sequential([
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

@st.cache_resource
def load_model():
    # Ensure the weights file is in the same directory as app.py
    model = create_model() 
    model.load_weights('model_weights.weights.h5')
    return model
    
model = load_model()
st.title("Grad-CAM Interpretability Demo")

# 2. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Process image
    img = tf.keras.utils.load_img(uploaded_file, target_size=(32, 32))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Generate Heatmap
    heatmap = get_gradcam_ultra_manual(img_array, model, 'conv2d_1')
    
    # Display
    col1, col2 = st.columns(2)
    col1.image(img, caption='Original Image')
    col2.image(heatmap, caption='Grad-CAM Heatmap', use_container_width=True)
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)


with st.expander("See Technical Report: How this Heatmap was generated"):
    st.write("""
    ### Grad-CAM Methodology Report
    To interpret the model's decision-making process, we implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)**:
    
    1. **Targeting Convolutional Features:** We extracted the activations from the `conv2d_4` layer, which captures high-level spatial patterns before flattening.
    2. **Gradient Backpropagation:** We calculated the gradient of the predicted class score with respect to the output of `conv2d_4`. This measures how much each neuron in the layer contributed to the final prediction.
    3. **Global Average Pooling:** We computed the importance weights by taking the mean of these gradients across the spatial dimensions.
    4. **Weighted Activation Map:** We performed a weighted sum of the layer's activations using these gradient importance weights.
    5. **ReLU Activation:** We applied a ReLU function to the resulting heatmap to discard negative influences, highlighting only the features that **positively contributed** to the model's confidence.
    """)

import tensorflow as tf
import numpy as np

def get_gradcam_ultra_manual(img_array, model, layer_name):
    # 1. Identify the layer
    target_layer = model.get_layer(layer_name)
    
    # 2. Define a function to get activations AND predictions
    # We must run the full model to ensure the gradient tape tracks the connection
    with tf.GradientTape() as tape:
        # Pass the input through all layers up to the target
        x = img_array
        for layer in model.layers:
            x = layer(x)
            if layer.name == layer_name:
                conv_outputs = x
                break # Stop at target layer
        
        # Now continue the pass to the end (the predictions)
        # We start from the target layer's output and continue to the end
        target_index = -1
        for i, layer in enumerate(model.layers):
            if layer.name == layer_name:
                target_index = i
                break
        
        # Continue the forward pass from the next layer
        final_x = conv_outputs
        for i in range(target_index + 1, len(model.layers)):
            final_x = model.layers[i](final_x)
            
        tape.watch(conv_outputs)
        
        # Get the class channel
        pred_index = tf.argmax(final_x[0])
        class_channel = final_x[:, pred_index]

    # 3. Calculate gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        raise ValueError(f"Gradients are None. Layer '{layer_name}' might not be contributing to the output.")

    # 4. Compute heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.reduce_max(heatmap)
    return (heatmap / max_val).numpy() if max_val > 0 else heatmap.numpy()
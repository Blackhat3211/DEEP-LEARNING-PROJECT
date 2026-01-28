import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_and_prep_data():
    """Download and normalize the CIFAR-10 dataset."""
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels

def build_model():
    """Create a Convolutional Neural Network."""
    model = models.Sequential([
        
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10) # 10 output classes
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def plot_results(history):
    """Visualize training progress."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title("Model Accuracy Over Time")
    plt.show()


if __name__ == "__main__":
    
    train_img, train_lbl, test_img, test_lbl = load_and_prep_data()
    
    
    print("🚀 Training starting... this may take a few minutes.")
    model = build_model()
    history = model.fit(train_img, train_lbl, epochs=10, 
                        validation_data=(test_img, test_lbl))
    
    
    plot_results(history)
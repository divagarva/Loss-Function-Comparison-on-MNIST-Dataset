import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Function to build a simple neural network
def build_model(loss_function):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    return model

# Train with Mean Squared Error
model_mse = build_model('mse')
history_mse = model_mse.fit(x_train, y_train_cat, epochs=5, validation_split=0.2, verbose=2)

# Train with Categorical Cross-Entropy
model_ce = build_model('categorical_crossentropy')
history_ce = model_ce.fit(x_train, y_train_cat, epochs=5, validation_split=0.2, verbose=2)

# Evaluate models
loss_mse, acc_mse = model_mse.evaluate(x_test, y_test_cat, verbose=0)
loss_ce, acc_ce = model_ce.evaluate(x_test, y_test_cat, verbose=0)

# Display results
print(f'MSE Loss: {loss_mse:.4f}, Accuracy: {acc_mse:.4f}')
print(f'Cross-Entropy Loss: {loss_ce:.4f}, Accuracy: {acc_ce:.4f}')

# Plot training loss comparison
plt.plot(history_mse.history['loss'], label='MSE Loss')
plt.plot(history_ce.history['loss'], label='Cross-Entropy Loss')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
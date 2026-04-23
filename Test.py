import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  # For multi-class classification, it's best to use a softmax activation on the output layer.
  tf.keras.layers.Dense(10, activation='softmax')
])

# The model needs to be compiled and trained before it can make meaningful predictions.
# 1. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 2. Train the model
print("--- Training Model ---")
model.fit(x_train, y_train, epochs=5)
print("--- Training Complete ---\n")

# Now let's make a prediction on the first test image
predictions_probabilities = model.predict(x_test[:1])
predicted_class = np.argmax(predictions_probabilities[0])

print(f"Predicted class: {predicted_class}")
print(f"Actual class:    {y_test[0]}")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

image_size = (48, 48)
batch_size = 32
epochs = 15
model_name = "model_emotion"

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "emotions/train",
    target_size=image_size,
    color_mode="grayscale",
    shuffle=True,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    "emotions/test",
    target_size=image_size,
    shuffle=True,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical')

print(train_generator.class_indices)
if os.path.exists(model_name):
    print("Load: " + model_name)
    classifier = load_model(model_name)
else:
    classifier = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights=None, input_tensor=None, input_shape=image_size + (1,), pooling=None, classes=7)
    classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch=train_generator.samples//batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=validation_generator.samples//batch_size)
model.save(model_name)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
plt.legend(['loss', 'acc'])
plt.savefig("learning-emotion.png")
plt.show()
plt.close()

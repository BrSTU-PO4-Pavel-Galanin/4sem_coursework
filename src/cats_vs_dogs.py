import numpy as np # массив
import tensorflow as tf # tensorflow
import tensorflow_datasets as tfds # mnist
from tensorflow.keras.preprocessing.image import load_img, img_to_array # загрузка изображения, изображение в массив
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # для создания слоёв
import matplotlib.pyplot as plt # для графиков
from google.colab import files # для загрузки файлов

train, _ = tfds.load(
    'cats_vs_dogs',
    split=['train[:100%]'],
    with_info=True,
    as_supervised=True
)

for img, label in train[0].take(10):
    plt.figure()
    plt.imshow(img)
    class_in_img = 'Dog' if label else 'Cat'
    plt.title(class_in_img)

SIZE = 224
def resize_image(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (SIZE, SIZE))
    img = img / 255.0
    return img, label

train_resized = train[0].map(resize_image)
train_batches = train_resized.shuffle(1000).batch(16)

base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False)
base_layers.trainable = False

model = tf.keras.Sequential([
    base_layers,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_batches, epochs=1)

f = files.upload()

list_img_path = []
for key in f:
  list_img_path.append(key)

for i in range(len(list_img_path)):
    path = list_img_path[i]
    img = load_img(path)
    img_array = img_to_array(img)
    img_resized, _ = resize_image(img_array, _)
    img_expended = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'Cat' if prediction < 0.5 else 'Dog'
    plt.figure()
    plt.imshow(img)
    plt.title(f'{path}\n{pred_label} {prediction}')

model.fit(train_batches, epochs=1)

for i in range(len(list_img_path)):
    path = list_img_path[i]
    img = load_img(path)
    img_array = img_to_array(img)
    img_resized, _ = resize_image(img_array, _)
    img_expended = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'Cat' if prediction < 0.5 else 'Dog'
    plt.figure()
    plt.imshow(img)
    plt.title(f'{path}\n{pred_label} {prediction}')

model.fit(train_batches, epochs=1)

for i in range(len(list_img_path)):
    path = list_img_path[i]
    img = load_img(path)
    img_array = img_to_array(img)
    img_resized, _ = resize_image(img_array, _)
    img_expended = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'Cat' if prediction < 0.5 else 'Dog'
    plt.figure()
    plt.imshow(img)
    plt.title(f'{path}\n{pred_label} {prediction}')
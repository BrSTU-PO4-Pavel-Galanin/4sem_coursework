# = = = = = библиотеки = = = = =

# При запуске программы TensorFlow 2+ пытается запустить GPU
# Нам нужен CUDA для GPU TensorFlow
# Чтобы не выводились предупреждения при запуске программы
# пропишем две строчки:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore

import numpy
import tensorflow
import tensorflow_datasets
import matplotlib.pyplot
#from google.colab import files # для загрузки файлов в Google Colabs

train, _ = tensorflow_datasets.load(
    'cats_vs_dogs',
    split=['train[:1%]'],
    with_info=True,
    as_supervised=True
)

for img, label in train[0].take(10):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(img)
    class_in_img = 'Dog' if label else 'Cat'
    matplotlib.pyplot.title(class_in_img)
matplotlib.pyplot.show()

SIZE = 224
def resize_image(img, label):
    img = tensorflow.cast(img, tensorflow.float32)
    img = tensorflow.image.resize(img, (SIZE, SIZE))
    img = img / 255.0
    return img, label

train_resized = train[0].map(resize_image)
train_batches = train_resized.shuffle(1000).batch(16)

base_layers = tensorflow.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False)
base_layers.trainable = False

model = tensorflow.keras.Sequential([
    base_layers,
    tensorflow.keras.layers.GlobalAveragePooling2D(),
    tensorflow.keras.layers.Dropout(0.2),
    tensorflow.keras.layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_batches, epochs=1)

f = []
#f = files.upload()

list_img_path = []
for key in f:
  list_img_path.append(key)

for i in range(len(list_img_path)):
    path = list_img_path[i]
    img = tensorflow.keras.preprocessing.imag.load_img(path)
    img_array = tensorflow.keras.preprocessing.imag.img_to_array(img)
    img_resized, _ = resize_image(img_array, _)
    img_expended = numpy.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'Cat' if prediction < 0.5 else 'Dog'
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(img)
    matplotlib.pyplot.title(f'{path}\n{pred_label} {prediction}')
    matplotlib.pyplot.show()

model.fit(train_batches, epochs=1)

for i in range(len(list_img_path)):
    path = list_img_path[i]
    img = tensorflow.keras.preprocessing.imag.load_img(path)
    img_array = tensorflow.keras.preprocessing.imag.img_to_array(img)
    img_resized, _ = resize_image(img_array, _)
    img_expended = numpy.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'Cat' if prediction < 0.5 else 'Dog'
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(img)
    matplotlib.pyplot.title(f'{path}\n{pred_label} {prediction}')
    matplotlib.pyplot.show()

model.fit(train_batches, epochs=1)

for i in range(len(list_img_path)):
    path = list_img_path[i]
    img = tensorflow.keras.preprocessing.imag.load_img(path)
    img_array = tensorflow.keras.preprocessing.imag.img_to_array(img)
    img_resized, _ = resize_image(img_array, _)
    img_expended = numpy.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'Cat' if prediction < 0.5 else 'Dog'
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(img)
    matplotlib.pyplot.title(f'{path}\n{pred_label} {prediction}')
    matplotlib.pyplot.show()
# = = = = = библиотеки = = = = =

# При запуске программы TensorFlow 2+ пытается запустить GPU
# Нам нужен CUDA для GPU TensorFlow
# Чтобы не выводились предупреждения при запуске программы
# пропишем две строчки:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore

import numpy
import matplotlib.pyplot
import tensorflow

c = numpy.array( [-40, -10,  0,  8, 15, 22,  38] )  # градусы цельсий
f = numpy.array( [-40,  14, 32, 46, 59, 72, 100] )  # градусы фаренгейт

model = tensorflow.keras.Sequential()

model.add(
    tensorflow.keras.layers.Dense(
        units=1,
        input_shape=(1,),
        activation='linear'
    )
)

model.compile(
    loss='mean_squared_error',
    optimizer=tensorflow.keras.optimizers.Adam(0.1)
)

history = model.fit(
    c,
    f,
    epochs=500,
    verbose=0
)
print("Обучение завершено")

mySample = [100, 101, 102]
print(f'Прогноз при выборке : {mySample}')
print(f'Результат           : {model.predict(mySample)}')
print()

print(f'Веса модели         : {model.get_weights()}')

matplotlib.pyplot.plot(history.history['loss'])
matplotlib.pyplot.grid(True)
matplotlib.pyplot.ylabel('Ошибка')
matplotlib.pyplot.xlabel('Итерации')
matplotlib.pyplot.title('Зависимость ошибки от итераций')
matplotlib.pyplot.show()
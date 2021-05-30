import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore

import numpy
import matplotlib.pyplot
import tensorflow

c = numpy.array([-40, -10, 0, 8, 15, 22, 38])
f = numpy.array([-40, 14, 32, 46, 59, 72, 100])

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

print( model.predict([100, 101, 102]) )
print( model.get_weights() )

matplotlib.pyplot.plot(history.history['loss'])
matplotlib.pyplot.grid(True)
matplotlib.pyplot.show()
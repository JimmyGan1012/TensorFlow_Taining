import tensorflow as tf
import tensorflow.keras.datasets as datasets


# GPU configs
# Working without memory growth setting cause memory error in my PC
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)




(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data() # m=60000, h,w=28


#Network structure using LeNet-5 (with size modification)
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28,28)))
model.add(tf.keras.layers.Reshape((28,28,1)))
model.add(tf.keras.layers.Conv2D(6,5,strides=1))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Conv2D(16,5,strides=1))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120,activation="relu"))
model.add(tf.keras.layers.Dense(84,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=5)

# eval = model.evaluate(x_test,y_test)

# Example to get output from a certain layer
from keras import backend as K
# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[5].output])
layer_output = get_3rd_layer_output(x_test)[0]

print("Accuracy:{}, Loss:{}".format(eval[0],eval[1]))





import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras

img_height = 180
img_width = 180

img = keras.preprocessing.image.load_img(
    "test_images/sunflower.jpg", target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

loaded_model = tf.keras.models.load_model("model/myFlower2.h5")

predictions = loaded_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


print("className : " + class_names[np.argmax(score)])

plt.xlabel("This image most likely belongs to '{}' with a '{:.2f}' percent confidnce.".format(
    class_names[np.argmax(score)], 100 * np.max(score)), color="black")

plt.grid(False)
plt.xticks([])
plt.yticks([])

plt.imshow(img)
plt.show()
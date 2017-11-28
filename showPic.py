from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train = mnist.train.images

print(train.shape)

for i in range(2):
    image = train[i].reshape([28, 28])
    plt.imshow(image, cmap="gray")
    plt.show()

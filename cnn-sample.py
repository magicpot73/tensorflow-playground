import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

batch_size = 128
num_classes = 10
epochs = 10

dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = dataset['train'], dataset['test']

def convert_types(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(batch_size)
mnist_test = mnist_test.map(convert_types).batch(batch_size)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(filters=32, kernel_size=3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(units=128, activation='relu')
    self.d2 = Dense(units=10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
  
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(image, label):
  with tf.GradientTape() as tape:
    predictions = model(image)
    loss = loss_object(label, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(label, predictions)

@tf.function
def test_step(image, label):
  predictions = model(image)
  t_loss = loss_object(label, predictions)
  
  test_loss(t_loss)
  test_accuracy(label, predictions)

EPOCHS = epochs

for epoch in range(EPOCHS):
  for image, label in mnist_train:
    train_step(image, label)
  
  for test_image, test_label in mnist_test:
    test_step(test_image, test_label)
  
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         test_loss.result(), 
                         test_accuracy.result()*100))

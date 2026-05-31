from nn import MLP
import tensorflow as tf
from keras.datasets import mnist

# prepare the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# vectorise the data
x_train_vectorised = x_train.reshape(x_train.shape[0], -1)
x_test_vectorised = x_test.reshape(x_test.shape[0], -1)

# normalise the data        (q. why do we normalise the dataset)
x_train_finalised = x_train_vectorised.astype("float32") / 255.0
x_test_finalised = x_test_vectorised.astype("float32") / 255.0


n = MLP(784, [8, 8, 10])
batchsize = 10
epochs = 22
learning_rate = 0.05

for epoch in range(epochs):

    #make batches then run the model through each batch, do shi, then shuffle the batches
    for ind in range(0, 60000, batchsize):
        batch = x_train_finalised[ind : ind+batchsize]
        batch_losses = []
        counter = ind
        for counter in range(ind, ind+batchsize):
            img = batch[counter-ind]
            ypred = n(img)
            ytarget = [1.0 if i == y_train[counter] else -1.0 for i in range(10)]

            img_loss = sum((ygt - yp) ** 2 for ygt, yp in zip(ytarget, ypred))
            batch_losses.append(img_loss)
        
        total_loss = sum(batch_losses) * (1.0/batchsize)

        for p in n.parameters():
            p.grad = 0.0

        total_loss.backward()


        for p in n.parameters():
            p.data += -(learning_rate) * p.grad

        print(f"batch: {ind/batchsize}  epoch: {epoch}  loss: {total_loss.data}")

        










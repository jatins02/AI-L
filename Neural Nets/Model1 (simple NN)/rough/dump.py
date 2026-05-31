# import graph_maker
import math
import random

# import os
# import logging

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# tf.get_logger().setLevel(logging.ERROR)
# tf.autograph.set_verbosity(0)

from keras.datasets import mnist


# 1. import and automatically load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(f"original training shape: {X_train.shape}")
# print(f"test dataset shape: {X_test.shape}")


# 2. vectorise (flatten) the 28 * 28 images into 784 size 1d arrays
# -1 tells numpy to automatically calculate the number of dimensions for the number of samples
X_train_vectorised = X_train.reshape(X_train.shape[0], -1)
X_test_vectorised = X_test.reshape(X_test.shape[0], -1)

# 3. normalise the pixel values (optional but highly recommended)
X_train_vectorised = X_train_vectorised.astype("float32") / 255.0
X_test_vectorised = X_test_vectorised.astype("float32") / 255.0 

# print(X_train_vectorised.shape)             # 60000 rows and 784 columns
# print(y_train.shape)

# class takes in the number of inputs while intialising it, then while calling it you pass in the input as list
# it calculates and returns the activation
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]     

"""
x = [2.0, 2.2, 2.1]         // list of inputs
n = Neuron(3)               // define a neuron taking 3 inputs
neuron_activation = n(x)                        // "call" the neuron with its inputs

"""

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]           # a Neuron in this layer will be connected with nin no. of neurons in the previous layer
                                                                    # this layer will contain nout no. of neurons in it

    def __call__(self, x):                                          # x is the list of inputs, then layer makes and returns list on nout no. of neurons that takes in nin number of inputs
        outs = [n(x) for n in self.neurons]                         # outs is the list of activation of neurons, at input x
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        params = [p for neuron in self.neurons for p in neuron.parameters()]

        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        
        return params

"""
now instead of calling the Neuron call each time, you just define the layers and the Neurons will be automatically made in the background
x = [2.0, 2.1, 4.4]         // list of inputs
n = Layer(3, 4)             // layer taking in 3 inputs and has 4 neurons in it
activation_list = n(x)      // list of activation of neurons in this layer 

"""

class MLP:
    def __init__(self, nin, nouts):         # nouts is the size of each "hidden layer", i.e. the number of neurons in each hidden layer
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)                    # x is the list of activation of neurons of that layer
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class Value:
    def __init__(self, data, _children = (), _op = "", label = ""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad                  # out.grad is the d(root)/d(out), now self.grad will be d(root)/d(self), because out = self + other. 
                                                        # so d(out)/d(self) = 1, so d(root)/d(self) = (d(root)/d(out)) * (d(out)/d(self)) = 1.0 * out.grad
            other.grad += 1.0 * out.grad        
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))          # only supports int/float powers for now
        out = Value(self.data ** other, (self, ), label = f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out


    def __truediv__(self, other):
        return self * (other**-1)
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), label = "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), label = "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    
    def backward(self):
        topo = []
        visited = set()

        # building the topological list
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # topological list is made, so make the grad of the node on which grad is called to be 1
        self.grad = 1.0

        #call _backward() function on all the nodes in the topo list
        for node in reversed(topo):
            node._backward()
        

# x1 = Value(2.0)
# w1 = Value(-3.0)

# x2 = Value(0.0)
# w2 = Value(1.0)

# b = Value(6.881373)

# x1w1 = x1 * w1
# x2w2 = x2 * w2

# x1w1x2w2 = x1w1 + x2w2
# n = x1w1x2w2 + b
# o = n.tanh()

# o.grad = 1.0

# o._backward()
# n._backward()
# b._backward()
# x1w1x2w2._backward()
# x2w2._backward()
# x1w1._backward()

"""
now you can only calculate the grad at a previous layer, if you know the grad at the next layer
so we HAVE TO CALL o._backward first and then move backwards, thus the name BACKPROPAGATION

now automating this fucntion calling, 
this will always be done in the above order, the result layer first then move backwards.

this can be implemented using topological sort: here all the edges go only one way
"""

# o.backward()
# print(x1.grad)

# a = Value(4.4)
# print(2.2-a)


# ---------------------------------------------------------
# using above model

# xs = [[2.0, 3.0, -1.0],             # list containing different inputs that will be fed into the nn
#       [3.0, -1.0, 0.5],
#       [0.5, 1.0, 1.0], 
#       [1.0, 1.0, -1.0]]

# ys = [1.0, -1.0, -1.0, 1.0]         # desired outputs

# n = MLP(3, [4, 4, 1])               # n is the MLP that takes inputs, 3 in size, has two layers with 4 neurons and the last layer with 1 neuron

# ypreds = [n(x) for x in xs]

# losses1 = [(ygt - yp) ** 2  for ygt, yp in zip(ypreds, ys)]

# loss1 = sum(losses1)
# print(loss1)
# loss1.backward()

# # print(n.layers[0].neurons[0].w[0].grad)

# for p in n.parameters():
#     p.data += -0.01 * p.grad                # 0.01 is the step size



# ypreds2 = [n(x) for x in xs]
# losses2 = [(ygt - ypred)**2 for ygt, ypred in  zip(ys, ypreds2)]
# loss2 = sum(losses2)
# print(loss2)



#training loop
# epochs = 1222
# for epoch in range(epochs):
#     #forward pass
#     ypreds = [n(x) for x in xs]
#     losses = [(ygt - ypred)**2 for ygt, ypred in zip(ys, ypreds)]
#     loss = sum(losses)

#     #backward pass
#     for p in n.parameters():
#         p.grad = 0.0
#     loss.backward()

#     #update
#     for p in n.parameters():
#         p.data += -0.01 * p.grad

#     print(f"{epoch}, {loss.data}")


n = MLP(784, [4, 4, 1])

#training loop
epochs = 100
for epoch in range(epochs):
    #forward pass
    ypreds = [n(x) for x in X_train_vectorised]
    losses = [(ygt - ypred) ** 2 for ygt, ypred in zip(y_train, ypreds)]
    loss = sum(losses)

    #backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.01 * p.grad

    print(f"epochs: {epoch}, {loss.data}")
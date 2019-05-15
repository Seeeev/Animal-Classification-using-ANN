""" 
Animal Classification through Artificial Neural Network with 
Back Propagation 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Define Architecture of NN
n_in = 16
n_h1 = 18
n_h2 = 14
n_out = 7
tp = []
eta = 0.1  # Learning Rate

# Pre-allocate storage and initialize weights + biases
x_in = np.zeros((n_in, 1))
w_h1 = -0.1 + (0.1 + 0.1) * np.random.rand(n_h1, n_in)
b_h1 = -0.1 + (0.1 + 0.1) * np.random.rand(n_h1, 1)
w_h2 = -0.1 + (0.1 + 0.1) * np.random.rand(n_h2, n_h1)
b_h2 = -0.1 + (0.1 + 0.1) * np.random.rand(n_h2, 1)
w_out = -0.1 + (0.1 + 0.1) * np.random.rand(n_out, n_h2)
b_out = -0.1 + (0.1 + 0.1) * np.random.rand(n_out, 1)
d_out = np.zeros((n_out, 1))
# Training Data
train_instances = 69  # 70
test_instances = 30  # 31
#X = np.array([[0, 0, 0] ,[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1 ,0 ,1], [1 ,1 ,0], [1, 1, 1]])
#Y = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0]])

# Training Data
X_dataset = pd.read_csv('Data Set/x_train_in.csv').values
X = X_dataset[:]
Y_dataset = pd.read_csv('Data Set/y_train_out.csv').values
Y = Y_dataset[:]

# Testing Data
x_dataset = pd.read_csv('Data Set/x_test_in.csv').values
x = x_dataset[:]
y_dataset = pd.read_csv('Data Set/y_test_out.csv').values
y = y_dataset[:]
max_epoch = 10000
# TRAINING PHASE
totalerr = np.zeros((max_epoch, 1))
#print("total1 err  :",totalerr)
for q in range(1, max_epoch):
    p = np.random.permutation(train_instances)
    # shuffle patterns
    for n in range(train_instances):
        nn = p[n]
        # read data
        x_in = X[[nn]].conj().T
        d_out = Y[[nn]].conj().T
        # forward pass
        # hidden layer 1
        v_h1 = np.dot(w_h1, x_in) + b_h1
        y_h1 = 1. / (1 + np.exp(-v_h1))
        # hidden layer 2
        v_h2 = np.dot(w_h2, y_h1) + b_h2
        y_h2 = 1. / (1 + np.exp(-v_h2))
        # output layer
        v_out = np.dot(w_out, y_h2) + b_out
        out = 1. / (1 + np.exp(-v_out))
        # error backpropagation %
        # compute error
        err = d_out - out
        # compute gradient in output layer
        delta_out = err * out * (1 - out)
        # compute gradient in hidden layer 2
        delta_h2 = y_h2 * (1 - y_h2) * np.dot(w_out.conj().T, delta_out)
        # compute gradient in hidden layer 1
        delta_h1 = y_h1 * (1 - y_h1) * np.dot(w_h2.conj().T, delta_h2)
        # update weights and biases in output layer
        w_out = w_out + eta * np.dot(delta_out, y_h2.conj().T)
        b_out = b_out + eta * delta_out
        # update weights and biases in hidden layer 2
        w_h2 = w_h2 + eta * np.dot(delta_h2, y_h1.conj().T)
        b_h2 = b_h2 + eta * delta_h2
        # update weights and biases in hidden layer 1
        w_h1 = w_h1 + eta * np.dot(delta_h1, x_in.conj().T)
        b_h1 = b_h1 + eta * delta_h1
    totalerr[q] = totalerr[q] + np.sum(err*err)
    if np.mod(q, 500) == 0:
        print('iteration:', q, 'Error: ', totalerr[q])
        # if termination condition is satisfied save weights and exit
    # if totalerr[q] < 0.001:
        # break
# Testing Phase
nn_output = np.zeros(np.shape(y))  # edit for test and train
for n in range(test_instances):  # edit for test and train
    # read data
    x_in = x[[n], :].conj().T  # edit for test and train
    d_out = y[[n], :].conj().T  # edit for test and train
    # hidden layer 1
    v_h1 = np.dot(w_h1, x_in) + b_h1
    y_h1 = 1. / (1 + np.exp(-v_h1))
    # hidden layer 2
    v_h2 = np.dot(w_h2, y_h1) + b_h2
    y_h2 = 1. / (1 + np.exp(-v_h2))
    # output layer
    v_out = np.dot(w_out, y_h2) + b_out
    out = 1. / (1 + np.exp(-v_out))
    for i in range(len(out)):
        if out[i] == max(out):
            nn_output[n][i] = 1  # return 1 if index is the highest value
        elif out[i] != max(out):
            # return 0 if index is not equal to the highet value
            nn_output[n][i] = 0
    #nn_output[[n],:]= np.greater_equal(out.conj().T, 0.5)
    print("Concatenate ", np.concatenate(
        (x_in.conj().T, nn_output[[n], :]), axis=1))
    # edit for test and train
    print(tp.append(np.array_equal(nn_output[n], y[n])))
print('Total Bits with error: ', np.sum(
    np.sum(np.abs(y - nn_output))))  # edit for test and train
print('Total epochs: ', q)
print('Network Error at termination:', totalerr[q])
plt.plot(totalerr[1:q])
plt.show()

# Things to try
# 1) Do not shuffle the patterns. Observe the error curve. Did the network learn ?
# 2) Change learning rate to 0.5 or 0.01. Observe the error curve. Did the network learn ?
# 3) Change the architecture (e.g. 3-2-2-3; 3-20-35-3)

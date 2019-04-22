# -*- coding: utf-8 -*-
"""
    Group LASSO regularization for neural networks (Theano/Lasagne)
    Author: Simone Scardapane
    Preprint: https://arxiv.org/abs/1607.00485
"""
# TODO: RUN THIS SHIT ON UBUNTU
# Necessary imports
import lasagne
from lasagne.nonlinearities import leaky_rectify, softmax
import theano, theano.tensor as T
import numpy as np
import sklearn.datasets, sklearn.preprocessing, sklearn.cross_validation
import matplotlib.pyplot as plt
from tabulate import tabulate
import time


# Define the group lasso penalty
def groupl1(x):
    return T.sum(T.sqrt(x.shape[1]) * T.sqrt(T.sum(x ** 2, axis=1)))


# Number of simulations
N_runs = 1

# Maximum number of epochs
max_epochs = 1500

# Define number of layers and number of neurons
H_layers = np.asarray([40, 20])

# Minibatch size
batch_size = 300

# Lasagne Regularizers to be tested
regularizers = [lasagne.regularization.l2,
                lasagne.regularization.l1,
                groupl1,
                lambda x: lasagne.regularization.l1(x) + groupl1(x),  # Sparse group LASSO
                ]

# Define the regularization factors for each algorithm
reg_factors = [10 ** -3.5, 10 ** -3.5, 10 ** -3.5, 10 ** -3.5]

# Define the names (for display purposes)
names = ['L2', 'L1', 'Group L1', 'Sparse GL1']

# Load the dataset (DIGITS)
digits = sklearn.datasets.load_digits()
X = digits.data
y = digits.target

# MNIST
# mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='C:/Users/ISPAMM/Downloads')
# X = mnist.data
# y = mnist.target

# Preprocessing (input)
scaler = sklearn.preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

# Output structures
tr_errors = np.zeros((len(regularizers), N_runs))
tst_errors = np.zeros((len(regularizers), N_runs))
tr_times = np.zeros((len(regularizers), N_runs))
tr_obj = np.zeros((len(regularizers), N_runs, max_epochs))
sparsity_weights = np.zeros((len(regularizers), N_runs, len(H_layers) + 1))
sparsity_neurons = np.zeros((len(regularizers), N_runs, len(H_layers) + 1))

# Define the input and output symbolic variables
input_var = T.matrix(name='X')
target_var = T.ivector(name='y')


# Utility function for minibatches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


for k in np.arange(0, N_runs):

    print("Run ", k + 1, " of ", N_runs, "...\n")

    # Split the data
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.25)

    # Define the network structure
    network = lasagne.layers.InputLayer((None, X.shape[1]), input_var)
    for h in H_layers:
        network = lasagne.layers.DenseLayer(network, h, nonlinearity=leaky_rectify, W=lasagne.init.GlorotNormal())
    network = lasagne.layers.DenseLayer(network, len(np.unique(y)), nonlinearity=softmax, W=lasagne.init.GlorotNormal())
    params_original = lasagne.layers.get_all_param_values(network)
    params = lasagne.layers.get_all_params(network, trainable=True)

    # Define the loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

    # Define the test function
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    test_fn = theano.function([input_var, target_var], test_acc, allow_input_downcast=True)

    for r in np.arange(0, len(regularizers)):

        # Set to original parameters
        lasagne.layers.set_all_param_values(network, params_original)

        # Define the regularized loss function
        loss_reg = loss.mean() + reg_factors[r] * lasagne.regularization.regularize_network_params(network,
                                                                                                   regularizers[r])

        # Update function
        # updates_reg = lasagne.updates.nesterov_momentum(loss_reg, params,learning_rate=0.01)
        updates_reg = lasagne.updates.adam(loss_reg, params)

        # Training function
        train_fn = theano.function([input_var, target_var], loss_reg, updates=updates_reg, allow_input_downcast=True)

        # Train network
        print("\tTraining with ", names[r], " regularization, epoch: ")
        start = time.time()
        for epoch in range(max_epochs):
            loss_epoch = 0
            batches = 0
            if np.mod(epoch, 10) == 0:
                print(epoch, "... ")
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                input_batch, target_batch = batch
                loss_epoch += train_fn(input_batch, target_batch)
                batches += 1
            tr_obj[r, k, epoch] = loss_epoch / batches
        end = time.time()
        tr_times[r, k] = end - start
        print(epoch, ".")

        # Final test with accuracy
        print("\tTesting the network with ", names[r], " regularization...")
        tr_errors[r, k] = test_fn(X_train, y_train)
        tst_errors[r, k] = test_fn(X_test, y_test)

        # Check sparsity
        params_trained = lasagne.layers.get_all_param_values(network, trainable=True)
        sparsity_weights[r, k, :] = [1 - (x.round(decimals=3).ravel().nonzero()[0].shape[0] / x.size) for x in
                                     params_trained[0::2]]
        sparsity_neurons[r, k, :] = [x.round(decimals=3).sum(axis=1).nonzero()[0].shape[0] for x in
                                     params_trained[0::2]]

tr_obj_mean = np.mean(tr_obj, axis=1)

# Plot the average loss
plt.figure()
plt.title('Training objective')
for r in np.arange(0, len(regularizers)):
    plt.semilogy(tr_obj_mean[r, :], label=names[r])
plt.legend()

# Print the results
print(tabulate([['Tr. accuracy [%]'] + np.mean(tr_errors, axis=1).round(decimals=4).tolist(),
                ['Test. accuracy [%]'] + np.mean(tst_errors, axis=1).round(decimals=4).tolist(),
                ['Tr. times [secs.]'] + np.mean(tr_times, axis=1).round(decimals=4).tolist(),
                ['Sparsity [%]'] + np.mean(sparsity_weights, axis=1).round(decimals=4).tolist(),
                ['Neurons'] + np.mean(sparsity_neurons, axis=1).round(decimals=4).tolist()],
               headers=[''] + names))
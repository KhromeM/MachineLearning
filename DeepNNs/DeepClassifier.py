import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    return 1/(1+np.exp(-Z)), Z

def relu(Z):
    A = Z
    A[A<=0] = 0
    return A,Z

def sigmoid_backward(dA,activation_cache):
    return dA * (sigmoid(activation_cache)[0]*(1-sigmoid(activation_cache)[0]))

def relu_backward(dA,activation_cache):

    def reluDerivative(activation_cache):
        activation_cache[activation_cache<=0] = 0
        activation_cache[activation_cache>0] = 1
        return activation_cache
    activation_cache = reluDerivative(activation_cache)
    return dA*activation_cache




def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)       

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) *.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
 
    return parameters


def linear_forward(A, W, b):

    Z = np.dot(W,A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2         

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
            
    return AL, caches


def compute_cost(AL, Y):

    m = Y.shape[1]

    cost = ((AL - Y)**2).mean(axis=1) #np.sum(((Y*np.log(AL)) + (1-Y) * np.log(1-AL))) * (-1/m)
    
    cost = np.squeeze(cost)   
    
    return cost

def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T) * (1/m)
    db = np.sum(dZ,axis=1,keepdims=True) * (1/m)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
   
    linear_cache, activation_cache = cache
    
    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

        
    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    
    return dA_prev, dW, db



def model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    dA_prev_temp = grads["dA"+str(L-1)]

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads



def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters








def model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

    costs = []                        
    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = model_forward(X, parameters)
        cost = compute_cost(AL, Y)

        grads = model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters



import math
x = np.array([[math.pi,math.pi/2,1,2,3,4,5,6,9,10]]).T


def predict(x,parameters): # use to make predictions
    AL, caches = model_forward(x, parameters)
    cost = compute_cost(AL, x*0)
    print(cost,'cost')



layers_dims = [10, 20, 7, 5, 1] # (n_x, L1, L2, L3 ...)
train_x = np.random.randn(10,1000) # (n_x,m)         # of features, # of examples
train_y = np.random.randn(1,1000) # (AL,m)           # of outputs, # of features

train_y = np.sin(train_y)

#parameters = model(train_x, train_y, layers_dims, num_iterations = 1000, print_cost = True)
#predict(x,parameters)




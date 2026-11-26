"""Train the handwritten dense ANN on the Bart/Homer image dataset."""

import time
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

from dataset_loader import load_split

#Constants
flag = False #Numerical gradient
e = 10**(-4)

#Data parameters & hyperparameters
scale = 35
num_labels = 2
n_hiddenLayers = 2
hidden_layer_size = 100
lamda = 1.6 #5\0.0001 4\0.05 3\0.3 2\1.6 1\8
MaxIter = 50

#Importing Train Data-------------------------------------------------------------------------------------------
dataset_root = (
    Path(__file__).resolve().parent
    / "data"
    / "archive"
    / "dataset_personagens"
    / "dataset_personagens"
)
X, y = load_split(dataset_root, "training_set", scale)
X_test, y_test = load_split(dataset_root, "test_set", scale)
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(f"Training images: {len(y)}")
print(f"Test images: {len(y_test)}")

y_eye = np.eye(num_labels, dtype=int)

#network parameters
input_layer_size = X.shape[1]
print("Input layer size-> ",input_layer_size)

def RandInitW(L_in, L_out):
    ep_init = np.sqrt(6)/np.sqrt(L_in+L_out)
    W = (np.random.rand(L_out, L_in+1)*2*ep_init) - ep_init
    return W

def sigmoid(z):
    g = 1/(1 + np.e**(-z))
    return g

def sigGradient(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return g
#Fine till here--------------------------------------------------------------------------------------------------------------------------------------------
def nnCostFunction(nn_params,*args):

    ils,hls,num_labels,X,y,lmd = args

    Theta_list = []
    #Input layer to hidden layer
    Theta_list += [nn_params[:(hls*(ils+1))].reshape((hls,(ils+1)))]
    #Hidden layers to output layer
    if n_hiddenLayers == 1: #One hidden layers
        Theta_list += [nn_params[hls*(ils+1):].reshape((num_labels,(hls+1)))]
    elif n_hiddenLayers > 1: #More than one hidden layers
        Theta_list += [nn_params[hls*(ils+1):hls*(ils+1)+(hls*(hls+1))].reshape((hls,(hls+1)))]
        if n_hiddenLayers>2:
            for i in range(2,n_hiddenLayers): #if hl = 3 i-> 2 hl = 5, i-> 2,3,4
                Theta_list += [nn_params[hls*(ils+1)+((i-1)*hls*(hls+1)):hls*(ils+1)+((i)*(hls*(hls+1)))].reshape((hls,(hls+1)))]
            Theta_list += [nn_params[hls*(ils+1)+((n_hiddenLayers-1)*(hls*(hls+1))):].reshape((num_labels,(hls+1)))]
        else:
            Theta_list += [nn_params[hls*(ils+1)+(hls*(hls+1)):].reshape((num_labels,(hls+1)))]

    #Initializations
    J = 0
    m = int(X.shape[0])
    bias = np.ones((m,1),dtype=int)
    X = np.concatenate((bias,X),axis=1)
    #Delta1, Delta2 = 0, 0

    for i in range(m):

        #   FORWARD PROPAGATION(1hidden) -> trying 2 hidden

        #Ith training example data
        xi = np.array([X[i]])
        yi = int(y[i, 0])

        zList = []
        aList = []
        #layer1 (i/p layer)
        #a1 = xi.transpose()
        aList += [xi.transpose()]

        #layer2 (hidden layer-1)
        #z2 = Theta1 @ a1
        #a2 = sigmoid(z2)
        #a2 = np.concatenate((np.array([[1]]),a2),axis=0) #if axis is 0(by default) concatenation is vertical, 1 -> horizontal, None-> flattented

        #layer3 (hidden layer-2)
        #z3 = Theta2 @ a2
        #a3 = sigmoid(z3)
        #a3 = np.concatenate((np.array([[1]]),a3),axis=0)

        for i in range(n_hiddenLayers):
            zList += [Theta_list[i] @ aList[-1]]
            aList += [sigmoid(zList[-1])]
            aList[-1] = np.concatenate((np.array([[1]]),aList[-1]),axis=0)

        #layer4 (o/p layer)
        zList += [Theta_list[-1] @ aList[-1]]
        #z4 = Theta3 @ a3
        #a4 = sigmoid(z4)
        aList += [sigmoid(zList[-1])]

        h_x = np.array(aList[-1], dtype=float)
        check1 = np.all(np.isfinite(np.log(h_x)))
        if check1 == False:
            print("Nan value found at h_x\n")
            break

        Y = np.array([y_eye[:,yi-1]]).transpose()

        #compute cost
        check2 = np.all(np.isfinite(np.log(h_x)))
        if check2 == False:
            print("Nan value found at log\n")
            break

        J += ((-Y.transpose() @ np.log(h_x)) - ((1-Y.transpose()) @ np.log(1-h_x))).item()

    RegTerm=0
    for theta in Theta_list:
        RegTerm += sum(sum(np.power(theta[:,1:],2)))
    RegTerm = (lmd/(2*m))*RegTerm
    J = (J/m) + RegTerm

    return J
#Fine till here--------------------------------------------------------------------------------------------------------------------------------------------
def ComputeNumericalGradient(nn_params, iter=5, *args):

    e = 10*(-30)
    numgrad = np.zeros(nn_params.shape)
    perturb = np.zeros(nn_params.shape)
    for p in range(iter):
        perturb[p] = e
        temp1, temp2 = nn_params-perturb, nn_params+perturb
        loss1 = nnCostFunction(temp1, *args)
        loss2 = nnCostFunction(temp2, *args)
        numgrad[p] = (loss2 - loss1)/(2*e)
        perturb[p] = 0
    return numgrad[:iter]



def norm(vec):
    return float(np.sqrt(sum(np.power(np.array(vec, dtype=float),2))))


def GradnnCostFunction(nn_params,*args):

    global flag
    ils,hls,num_labels,X,y,lmd = args

    Theta_list = []
    #Input layer to hidden layer
    Theta_list += [nn_params[:(hls*(ils+1))].reshape((hls,(ils+1)))]
    #Hidden layers to output layer
    if n_hiddenLayers == 1: #One hidden layers
        Theta_list += [nn_params[hls*(ils+1):].reshape((num_labels,(hls+1)))]
    elif n_hiddenLayers > 1: #More than one hidden layers
        Theta_list += [nn_params[hls*(ils+1):hls*(ils+1)+(hls*(hls+1))].reshape((hls,(hls+1)))]
        if n_hiddenLayers>2:
            for i in range(2,n_hiddenLayers): #if hl = 3 i-> 2 hl = 5, i-> 2,3,4
                Theta_list += [nn_params[hls*(ils+1)+((i-1)*hls*(hls+1)):hls*(ils+1)+((i)*(hls*(hls+1)))].reshape((hls,(hls+1)))]
            Theta_list += [nn_params[hls*(ils+1)+((n_hiddenLayers-1)*(hls*(hls+1))):].reshape((num_labels,(hls+1)))]
        else:
            Theta_list += [nn_params[hls*(ils+1)+(hls*(hls+1)):].reshape((num_labels,(hls+1)))]

    #Initializations
    J = 0
    m = int(X.shape[0])
    bias = np.ones((m,1),dtype=int)
    X = np.concatenate((bias,X),axis=1)
    DeltaList = list(np.zeros(n_hiddenLayers+1))

    for i in range(m):

        #   FORWARD PROPAGATION(1hidden) -> trying 2 hidden

        #Ith training example data
        xi = np.array([X[i]])
        yi = int(y[i, 0])

        zList = []
        aList = []
        #layer1 (i/p layer)
        #a1 = xi.transpose()
        aList += [xi.transpose()]

        #layer2 (hidden layer-1)
        #z2 = Theta1 @ a1
        #a2 = sigmoid(z2)
        #a2 = np.concatenate((np.array([[1]]),a2),axis=0) #if axis is 0(by default) concatenation is vertical, 1 -> horizontal, None-> flattented

        #layer3 (hidden layer-2)
        #z3 = Theta2 @ a2
        #a3 = sigmoid(z3)
        #a3 = np.concatenate((np.array([[1]]),a3),axis=0)

        for i in range(n_hiddenLayers):
            zList += [Theta_list[i] @ aList[-1]]
            aList += [sigmoid(zList[-1])]
            aList[-1] = np.concatenate((np.array([[1]]),aList[-1]),axis=0)

        #layer4 (o/p layer)
        zList += [Theta_list[-1] @ aList[-1]]
        #z4 = Theta3 @ a3
        #a4 = sigmoid(z4)
        aList += [sigmoid(zList[-1])]

        h_x = np.array(aList[-1], dtype=float)
        check1 = np.all(np.isfinite(np.log(h_x)))
        if check1 == False:
            print("Nan value found at h_x\n")
            break

        Y = np.array([y_eye[:,yi-1]]).transpose()

        #compute cost
        check2 = np.all(np.isfinite(np.log(h_x)))
        if check2 == False:
            print("Nan value found at log\n")
            break

        J += ((-Y.transpose() @ np.log(h_x)) - ((1-Y.transpose()) @ np.log(1-h_x))).item()

        #   BACK PROPAGATION

        #Error vectors
        #d4 = a4 - Y
        #d3 = (Theta3.transpose() @ d4)[1:] * (sigGradient(z3))
        #d2 = (Theta2.transpose() @ d3)[1:] * (sigGradient(z2))

        dList = []
        dList += [aList[-1] - Y]
        for i in range(n_hiddenLayers,0,-1):
            temp_dval = (Theta_list[i].transpose() @ dList[-1])[1:] * sigGradient(zList[i-1])
            dList += [temp_dval]
        #Fine till here----------------------

        #Error matrix accumulater
        #Delta3 += d4 @ a3.transpose()
        #Delta2 += d3 @ a2.transpose()
        #Delta1 += d2 @ a1.transpose()

        for i in range(len(dList)-1,-1,-1):
            d_index = len(dList)-i-1
            a_index = i
            DeltaList[i] += dList[len(dList)-i-1] @ aList[i].transpose()
        #Fine till here-----------------

    RegTerm=0
    for theta in Theta_list:
        RegTerm += sum(sum(np.power(theta[:,1:],2)))
    RegTerm = (lmd/(2*m))*RegTerm
    J = (J/m) + RegTerm

    #temp1 = np.array([(1/m)*Delta1[:,0]]).transpose()
    #temp2 = (1/m)*(Delta1 + (lmd * Theta1))[:,1:]
    #Theta1_grad = np.concatenate((temp1,temp2),axis=1)


    #temp1 = np.array([(1/m)*Delta2[:,0]]).transpose()
    #temp2 = (1/m)*(Delta2 + (lmd * Theta2))[:,1:]
    #Theta2_grad = np.concatenate((temp1,temp2),axis=1)

    #temp1 = np.array([(1/m)*Delta3[:,0]]).transpose()
    #temp2 = (1/m)*(Delta3 + (lmd * Theta3))[:,1:]
    #Theta3_grad = np.concatenate((temp1,temp2),axis=1)

    Theta_gradList = []
    for i in range(len(DeltaList)):
        temp1 = np.array([(1/m)*DeltaList[i][:,0]]).transpose()
        temp2 = (1/m)*(DeltaList[i] + (lmd * Theta_list[i]))[:,1:]
        Theta_gradList += [np.concatenate((temp1,temp2),axis=1)]
    #Theta_gradList.reverse()
    Theta_gradTuple = tuple(Theta_gradList)

    grad = np.concatenate(Theta_gradTuple,axis=None)

    if flag == True:
        #-------Actual----------------------------------------------------------------



        iter = 100
        grad.size
        print("\nComputing Numerical Gradient...\n")
        NumericalGrad = ComputeNumericalGradient(initial_nn_params, iter, *args)
        #-----------------------------------------------------------------------------
        print("Numerical Gradient computed successfully.")
        print(f"Numerical Gradient-> \n{NumericalGrad}") if iter < 100 else print(f"Numerical Gradient-> \n{NumericalGrad[0:100]}")
        print(f"Shape of Numerical Gradient-> \n{NumericalGrad.shape}")
        print(f"\nGrad-> \n{grad[0:iter]}") if iter < 100 else print(f"\nGrad-> \n{grad[0:100]}")
        print(f"Shape of Grad-> {grad[0:100].shape}\n")
        temp_diff = norm(NumericalGrad - np.array([grad[0:iter]]).T)
        temp_sum = norm(NumericalGrad + np.array([grad[0:iter]]).T)
        rel_diff = temp_diff/temp_sum
        print(f"\nRelative Difference-> {np.format_float_scientific(rel_diff)}")
        while True:
            cont = (input("Continue? (Enter 'y' or 'n')-> ")).lower()
            if cont == 'y':
                flag = False
                break
            elif cont == 'n':
                print('Exiting...')
                sys.exit()
            else:
                print("Invalid Input")

    return grad
#Error at gradnnCostfunciton---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def predict(X,theta):

    #h1 = sigmoid(np.concatenate((np.ones((X.shape[0],1)),X), axis=1) @ theta1.T)
    #h2 = sigmoid(np.concatenate((np.ones((h1.shape[0],1)),h1), axis=1) @ theta2.T)
    #h3 = sigmoid(np.concatenate((np.ones((h2.shape[0],1)),h2), axis=1) @ theta3.T)

    h = X
    for i in range(len(theta)):
        h = sigmoid(np.concatenate((np.ones((h.shape[0],1)),h), axis=1) @ theta[i].T)

    p = np.array([[list(example).index(max(example))+1 for example in h]]).T
    return p

#__main__

#Initialize theta
init_Theta_List = []
init_Theta_List += [RandInitW(input_layer_size, hidden_layer_size)]
for i in range(n_hiddenLayers-1):
    init_Theta_List += [RandInitW(hidden_layer_size, hidden_layer_size)]
init_Theta_List += [RandInitW(hidden_layer_size, num_labels)]
init_Theta_tuple = tuple(init_Theta_List)

#Unroll parameters
initial_nn_params = np.array([np.concatenate(init_Theta_tuple,axis=None)]).transpose()

#Testing cost and grad funcitons
args = (input_layer_size,hidden_layer_size,num_labels,X,y,lamda)

Init_Cost = nnCostFunction(initial_nn_params, *args)
Grad = GradnnCostFunction(initial_nn_params, *args)

#Optimizing

args = (input_layer_size,hidden_layer_size,num_labels,X,y,lamda)

#Countdown
print("Optimizing...\n")
for i in range(3,0,-1):
    print(f'{i}...',end='')
    time.sleep(0.5)
print()

optimized_parameters, _, _, _, optimization_status = optimize.fmin_cg(
    nnCostFunction,
    initial_nn_params,
    fprime=GradnnCostFunction,
    args=args,
    maxiter=MaxIter,
    full_output=True,
)
optimized_parameters = np.array([optimized_parameters]).transpose()
if optimization_status == 0:
    print("Optimization converged.\n")
else:
    print(
        "Optimization stopped without confirmed convergence "
        f"(SciPy warning code {optimization_status}).\n"
    )

optimized_Cost = nnCostFunction(optimized_parameters, *args)
print("\nInitial value of Cost Function-> {}".format(Init_Cost))
print("Optimized value of Cost Function-> {}\n".format(optimized_Cost))

#Theta1 = optimized_parameters[0:(hidden_layer_size*(input_layer_size+1))].reshape((hidden_layer_size,(input_layer_size+1)))
#Theta2 = optimized_parameters[hidden_layer_size*(input_layer_size+1):hidden_layer_size*(input_layer_size+1)+(hidden_layer_size*(hidden_layer_size+1))].reshape((hidden_layer_size,(hidden_layer_size+1)))
#Theta3 = optimized_parameters[hidden_layer_size*(input_layer_size+1)+(hidden_layer_size*(hidden_layer_size+1)):].reshape((num_labels,(hidden_layer_size+1)))

Theta_list = []
#Input layer to hidden layer
Theta_list += [optimized_parameters[:(hidden_layer_size*(input_layer_size+1))].reshape((hidden_layer_size,(input_layer_size+1)))]
#Hidden layers to output layer
if n_hiddenLayers == 1: #One hidden layers
    Theta_list += [optimized_parameters[hidden_layer_size*(input_layer_size+1):].reshape((num_labels,(hidden_layer_size+1)))]
elif n_hiddenLayers > 1: #More than one hidden layers
    Theta_list += [optimized_parameters[hidden_layer_size*(input_layer_size+1):hidden_layer_size*(input_layer_size+1)+(hidden_layer_size*(hidden_layer_size+1))].reshape((hidden_layer_size,(hidden_layer_size+1)))]
    if n_hiddenLayers>2:
        for i in range(2,n_hiddenLayers): #if hl = 3 i-> 2 hl = 5, i-> 2,3,4
            Theta_list += [optimized_parameters[hidden_layer_size*(input_layer_size+1)+((i-1)*hidden_layer_size*(hidden_layer_size+1)):hidden_layer_size*(input_layer_size+1)+((i)*(hidden_layer_size*(hidden_layer_size+1)))].reshape((hidden_layer_size,(hidden_layer_size+1)))]
        Theta_list += [optimized_parameters[hidden_layer_size*(input_layer_size+1)+((n_hiddenLayers-1)*(hidden_layer_size*(hidden_layer_size+1))):].reshape((num_labels,(hidden_layer_size+1)))]
    else:
        Theta_list += [optimized_parameters[hidden_layer_size*(input_layer_size+1)+(hidden_layer_size*(hidden_layer_size+1)):].reshape((num_labels,(hidden_layer_size+1)))]
Theta_tuple = tuple(Theta_list)

#Making predictions
theta = Theta_tuple
p_train = predict(X, theta)
p_test = predict(X_test, theta)

print(f"Lamda = {lamda}")
print(f"No. Hidden layers = {n_hiddenLayers}")
print(f"Hidden layer size = {hidden_layer_size}")
train_accuracy = np.mean(p_train == y)*100
test_accuracy = np.mean(p_test == y_test)*100
print(f'Train Accuracy-> {train_accuracy}')
print(f'Test Accuracy-> {test_accuracy}\n')

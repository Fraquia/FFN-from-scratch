# FFN-from-scratch
Implementation of a simple two-layer neural network and its training algorithm based on back-propagation using first matrix operations and then PyTorch library, to understand the advantages offered in using such tools. This involves:

1. Implementing the feedforward model (Part 1).
2. Implementing the backpropagation algorithm (gradient computation) (Part 2).
3. Training the model using stochastic gradient descent and improving the model training with better hyper- parameters (Part 3).
4. Using the PyTorch Library to implement the above and experiment with deeper networks (Part 4).

## Dataset and Task

We developed the same architecture in different ways so as benchmark to test our models, we consider an image classification task using the widely used CIFAR-10 dataset. This dataset consists of 50000 training images of 32x32 resolution with 10 object classes, namely airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. **The task is to code and train a parameterised model for classifying those images**

## Part 1: FFN Architecture

In this part we will implement a two-layered a neural network architecture as well as the loss function to train it. Starting from the main file **ex2_FCnet.py**, we specified the required code in the **two_layernet.py** to complete this question.

Our architecture is shown in the image that follows:

<img src="https://github.com/Fraquia/FFN-from-scratch/blob/main/architecture.png" width="70%" height="100%">


The model has:
1. An input layer. 
2. Two model layers – a hidden and an output layer. 

In the hidden layer, there are 10 units. The input layer and the hidden layer are connected via linear weighting matrix W(1) ∈ R10x4 and the bias term b(1) ∈ R10. The parameters W(1) and b(1) are to be learnt later on. 

A linear operation is performed, W (1)x + b(1), resulting in a 10-dimensional vector z(2). It is then followed bya ReLU non-linear activation φ, applied element-wise on each unit, resulting in the activations a(2) = φ(z(2)).


A similar linear operation is performed on a(2), resulting in z(3) = W(2)a(2), where W(2) ∈ R3x10 and b(2) ∈ R3; it is followed by the softmax activation to result in a(3) = ψ(z(3)).

**The final functional form of our model is thus defined by:**

  a(1) = x
  
  z(2) =W(1)a(1) +b(1) 
  
  a(2) = φ(z(2))
  
  z(3) =W(2)a(2) +b(2)
  
  fθ(x) := a(3) = ψ(z(3))
  
We indicate all the network k parameters by **θ=(W(1) ,b(1) ,W(2) ,b(2))**. We later guide the neural network parameters θ to fit to the given data and label pairs. We do so by minimising the loss function. A popular choice of the loss function for training neural network for multi-class classification is the **cross-entropy loss**.

## Part 2: Backpropagation Algorithm 


We train the model by solving the **Objective Function** via **Stochastic Gradient Descent**. We therefore need an efficient computation of the gradients ∇J. We use **Backpropagation** of top layer error signals to the parameters θ at different layers.
In this part, we implemented the backpropagation algorithm from scratch in the **two_layernet.py** file.

### Backpropagation 


The backpropagation algorithm is simply a sequential application of **chain rule**. It is applicable to any (sub-) differentiable model that is a composition of simple building blocks. In this part, we focus on the architecture with stacked layers of linear transformation + ReLU non-linear activation.

The intuition behind the backpropagation algorithm is as follows. Given a training example (xi, yi), we first run the feedforward to compute all the activations throughout the network, including the output value of the model fθ(xi) and the loss J. Then, for each parameter in the model we want to compute the effect that parameter has on the loss. This is done by computing the derivatives of the loss w.r.t. each model parameter.

The backpropagation algorithm is performed from the top of the network (loss layer) towards the bottom. Itsequentially computes the gradient of the loss function with respect to each layer activations and parameters.

## Stochastic Gradient Descend

We have implemented the backpropagation algorithm for computing the parameter gradients and have verified that it indeed gives the correct gradient. We are now ready to train the network. We solve 
<img src="https://github.com/Fraquia/FFN-from-scratch/blob/main/min.png" width="70%" height="100%">
with the stochastic gradient descent.

# FFN-from-scratch
Implementation of a simple two-layer neural network and its training algorithm based on back-propagation using first matrix operations and then PyTorch library, to understand the advantages offered in using such tools. This involves:

1. Implementing the feedforward model (Part 1).
2. Implementing the backpropagation algorithm (gradient computation) (Part 2).
3. Training the model using stochastic gradient descent and improving the model training with better hyper- parameters (Part 3).
4. Using the PyTorch Library to implement the above and experiment with deeper networks (Part 4).

**To be able to train the above model on large datasets, with larger layer widths, the code has to be very efficient. To do this we  avoided using any python for loops in the forward pass and instead we used matrix and vector multiplication routines in the Numpy library.**

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

We have implemented the backpropagation algorithm for computing the parameter gradients and have verified that it indeed gives the correct gradient. We are now ready to train the network. We solved 
<img src="https://github.com/Fraquia/FFN-from-scratch/blob/main/min.png" width="10%" height="5%">
with the stochastic gradient descent.

We implemented the stochastic gradient descent algorithm in **two_layernet.py** and run the training on the toy data. Pur model should be able to obtain loss = 0.02 on the training set and the training curve should look similar to this:

<img src="https://github.com/Fraquia/FFN-from-scratch/blob/main/loss.png" width="30%" height="50%">

## FFN, Backpropagation and SGD 

Completed parts 1,2 and 3 we are now ready to train our model on real image dataset. For this we will use the CIFAR-10 dataset. Since the images are of size 32 × 32 pixels with 3 color channels, this gives us 3072 input layer units, represented by a vector x ∈ R3072. The code to load the data and train the model is provided with some default hyper-parameters in **ex2_FCnet.py**. 

With default hyper-parametres we got validation set accuracy of about 29%. This is very poor. So we tried to debug the model training and come up with better hyper-parameters to improve the performance on the validation set also visualizing the training and validation performance curves to help with this analysis. 

## Part 4 

So far we have implemented a two-layer network by explicitly writing down the expressions for forward, backward computations and training algorithms using simple matrix multiplication primitives from the NumPy library.

However there are many libraries available designed make experimenting with neural networks faster, by abstracting away the details into reusable modules. One such popular open-source library is **PyTorch** (https: //pytorch.org/). 

In this final question we will use PyTorch library to implement the same two-layer network we did before and train it on the CIFAR-10 dataset. However, extending a two-layer network to a three or four layered one is a matter of changing two-three lines of code using PyTorch. We will take advantage of this to experiment with deeper networks to improve the performance on the CIFAR-10 classification. 

In this part we:

1. Implemented a multi-layer perceptron network in the class **MultiLayerPerceptron** in **ex2_pytorch.py**. This includes instantiating the required layers from **torch.nn** and writing the code for forward pass. Initially we wrote the code for the same two-layer network we have seen before.

2. Completed the code to train the network using loss function **torch.nn.CrossEntropyLoss** to compute the loss and **loss.backward()** to compute the gradients. Once gradients are computed, **optimizer.step()** was invoked to update the model. 

3. We trained the two layer network to achieve reasonable performance, also increasing the network depth to see if we can improve the performance.

## Report 
We also produced a PDF report to analyse the results we obtained in each different part. 

## Authors and Credits 

### Authors 
1. Caterina Alfano (@cat-erina)
2. Angelo Berardi (@AngeloBerardi) 
3. Emanuele Fratocchi (@Fraquia)
4. Dario Cappelli (@Capp18)

### Credits 
This project was develped as an exercise of Advanced Machine Learning Course at Sapienza University of Rome. This exercise credit goes to Prof. Fabio Galasso.

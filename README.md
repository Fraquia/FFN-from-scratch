# FFN-from-scratch
Implementation of a simple two-layer neural network and its training algorithm based on back-propagation using first matrix operations and then PyTorch library, to understand the advantages offered in using such tools. This involves:

1. Implementing the feedforward model (Part 1).
2. Implementing the backpropagation algorithm (gradient computation) (Part 2).
3. Training the model using stochastic gradient descent and improving the model training with better hyper- parameters (Part 3).
4. Using the PyTorch Library to implement the above and experiment with deeper networks (Part 4).

## Dataset and Task

We developed the same architecture in different ways so as benchmark to test our models, we consider an image classification task using the widely used CIFAR-10 dataset. This dataset consists of 50000 training images of 32x32 resolution with 10 object classes, namely airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. **The task is to code and train a parameterised model for classifying those images**

## FFN Architecture
Our architecture is shown in the image that follows:

<img src="https://github.com/Fraquia/FFN-from-scratch/blob/main/architecture.png" width="70%" height="100%">


The model has:
1. An input layer. 
2. Two model layers – a hidden and an output layer. 

We start with randomly generated toy inputs of 4 dimensions and number of classes K = 3 to build our model in Questions 1, 2 and 3. Then we would use images from CIFAR-10 dataset to test our model on a real-world task in Question 3. Hence input layer is 4 dimensional for now. In the hidden layer, there are 10 units. The input layer and the hidden layer are connected via linear weighting matrix W(1) ∈ R10x4 and the bias term b(1) ∈ R10. The parameters W(1) and b(1) are to be learnt later on. A linear operation is performed, W (1)x + b(1), resulting in a 10-dimensional vector z(2). It is then followed by

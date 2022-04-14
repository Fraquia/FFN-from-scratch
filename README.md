# FFN-from-scratch
Implementation of a simple two-layer neural network and its training algorithm based on back-propagation using first matrix operations and then PyTorch library, to understand the advantages offered in using such tools.

## FFN Architecture

Our architecture is shown in 1. It has an input layer, and two model layers – a hidden and an output layer. We start with randomly generated toy inputs of 4 dimensions and number of classes K = 3 to build our model in Questions 1, 2 and 3. Then we would use images from CIFAR-10 dataset to test our model on a real-world task in Question 3. Hence input layer is 4 dimensional for now.
In the hidden layer, there are 10 units. The input layer and the hidden layer are connected via linear weighting matrix W(1) ∈ R10x4 and the bias term b(1) ∈ R10. The parameters W(1) and b(1) are to be learnt later on. A linear operation is performed, W (1)x + b(1), resulting in a 10-dimensional vector z(2). It is then followed by

# General-Regression-Neural-Network
General Regression Neural Network is a variant of radial basis function neural network and a powerful tool for nonlinear function approximation.
GRNN was developed by Specht (1991) based on idea that each training observation is assigned a neuron in the first hidden layer called pattern layer [22]. Compared to RBFNN, GRNN has one additional hidden layer called summation layer. There are no input weights connecting the input layer nodes to the pattern layer neurons. The traditional GRNN assumes that the pattern layer can have as many neurons as the given number of training examples. The pattern layer neurons use radial basis function (RBF) as activation function, thus these neurons are called radial basis neurons. The first training observation is selected as the center of the first radial basis neuron, the second training observation is the center of the second radial basis neuron and so on. All the radial basis neurons have the same width or spread. 

![image](https://user-images.githubusercontent.com/86943102/156288870-c597f909-dbee-46f4-a7a0-eb74252710fc.png)

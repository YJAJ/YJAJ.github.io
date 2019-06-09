---
layout: post
author: YJ Park
title:  "Basic neural network Part 1 - Math involved in forward and backward functions (assignment with Python)"
date:   2019-06-09 10:50:00 +1000
categories: jekyll update
tags: neural network, neural network mathematics, backpropagation, backpropagation mathematics
---
<head>
	<!-- Global site tag (gtag.js) - Google Analytics -->
	<script async src="https://www.googletagmanager.com/gtag/js?id=UA-127453746-1"></script>
	<script>
		  window.dataLayer = window.dataLayer || [];
		  function gtag(){dataLayer.push(arguments);}
		  gtag('js', new Date());

		  gtag('config', 'UA-127453746-1');
	</script>
</head>

This is Part 1 of the two series of basic neural network posts, covering explanations of mathematics involved, in particular how backpropagation works with one hidden layer and one output layer.
Part 2 addresses a vectorisation implementation for a neural network in Python based on the formulas covered in this post.
If you are not fond of the mathematics behind neural networks, you can skip to [Part 2](http://yjpark.me/blog/jekyll/update/2019/06/09/basic-neural-network-part2.html).

## Illustration of a small neural network
Let us assume a neural network with one hidden layer and one output layer with weights and biases initialised as above for the illustration purpose.
We have a data set with a batch size 2 and its two data points are X1 = (x1: 0.1, x2: 0.1) and X2 = (x1: 0.1, x2: 0.2).
Ground truth labels for X1 and X2 is 0 and 1, respectively. A learning rate is set to 0.1.

![An example of a small neural network](../../../../../../assets/images/Example_of_a_small_neural_network.png)

We will follow the following process step by step:
1. Forward function for a hidden layer;
2. Forward function for an output layer;
3. Quadratic loss function;
4. Backpropagation of an output layer;
5. Backpropagation of a hidden layer; and finally,
6. Adjustment of weights and biases- Gradient Descent

This process can be loosely visualised as a red arrow below:

![Whole_process](../../../../../../assets/images/Whole process.png)

### 1. Forward function for a hidden layer
First, the following forward function needs to be calculated:

![Forward formula](../../../../../../assets/images/Forward_formula.png)

This will result in the following hidden layer:

![Hidden layer](../../../../../../assets/images/Hidden_layer.png)

Then, a sigmoid function is calculated to produce the result of the hidden layer:

![Hidden layer sigmoid](../../../../../../assets/images/Hidden_layer_sigmoid.png)

### 2. Forward function for an output layer
Similarly, the following forward function needs to be calculated for the output layer based on the result from the hidden layer:

![Output layer](../../../../../../assets/images/Output_layer.png)

The final result of output layers is then calculated through a sigmoid function again:

![Output layer sigmoid](../../../../../../assets/images/Output_layer_sigmoid.png)

### 3. Quadratic loss function
Based on the final result of the output layers, the loss is calculated. Here, a quadratic loss function is used.

![Quadratic](../../../../../../assets/images/Quadratic_loss.png)

![Quadratic_cal](../../../../../../assets/images/Quadratic_loss_cal.png)

### 4. Backpropagation of an output layer
The loss calculated above is then broken down into the two output losses:

![Backpropagation_output](../../../../../../assets/images/Backprop_output.png)

where:

![Backpropagation_output_chain_rule](../../../../../../assets/images/Backprop_output_chain.png)

Then, a chain rule in a backward function is:

![Backpropagation_output_cal](../../../../../../assets/images/Backprop_output_cal.png)

### 5. Backpropagation of a hidden layer
I find the backward process for a hidden layer is a bit more heavy.
Essentially, we will need to use what was calculated previously for the derivative of an individual loss and an output sigmoid function, which are highlighted in yellow below.

![Backpropagation_hidden](../../../../../../assets/images/Backprop_hidden.png)

For a more concrete example, I will use the weight W1 below:

![Backpropagation_hidden_chain_rule](../../../../../../assets/images/Backprop_hidden_chain.png)

If we calculate all weights and biases for a hidden layer in a similar manner, we get:

![Backpropagation_hidden_cal](../../../../../../assets/images/Backprop_hidden_cal.png)

### 6. Adjustment of weights and biases- Gradient Descent
With a learning rate 0.1, we can now perform first gradient descent of our small neural network:

![Gradient_descent](../../../../../../assets/images/Gradient_descent.png)

Applying this, we will eventually get all adjusted weights and biases as follow:

![Gradient_descent_cal](../../../../../../assets/images/Gradient_descent_cal.png)

That's it! This is only one batch gradient descent performed with a batch size 2 data points.
Now, it is time to implement this hairy process into codes! You can see this implementation here, [Part 2](http://yjpark.me/blog/jekyll/update/2019/06/09/basic-neural-network-part2.html).
Otherwise, if you have any questions on the process and calculation, feel free to contact me.
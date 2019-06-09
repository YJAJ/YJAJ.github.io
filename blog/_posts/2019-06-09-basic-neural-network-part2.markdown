---
layout: post
author: YJ Park
title:  "Basic neural network - Implementation (assignment with Python)"
date:   2019-06-09 10:50:00 +1000
categories: jekyll update
tags: neural network, vectorisation, neural network python code
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

In this post, I will go through how a basic neural network with one hidden layer can be implemented with its vectorisation.
This is Part 2 of the basic neural network series so if you are interested in the mathematics involved in neural networks, you can have a look at [Part 1](http://yjpark.me/blog/jekyll/update/2019/06/09/basic-neural-network-part1.html).

The data set used in this NN example is based on [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/).

## Basic points of this NN implementation
The implementation of the neural network is done through matrix calculation for each batch. 

While an element-wise multiplication is used in some cases, a dot product is mainly utilised during the forward and backpropagation process where two examples in a batch need to be multiplied and added together.

With the MNIST data set, input matrix’s first dimension is a batch size (e.g. default batch size used is 20 in MNIST). Input matrix’s second dimension is 784 (28x28) representing pixels to make up a digit. The example of the input can be visualised by using the “visualise_individual_data” function in the source code below, displaying the digit in a plot as follow.


![An example of the input data](../../../../../../assets/images/Example_of_the_input_data.png)

{% highlight  Python%}
#utility function to visualise a csv row into a digit
def visualise_individual_data(data):
    img_sz = 28
    img_index = 20
    image = data[img_index,:]
    im_sq = np.reshape(image, (img_sz, img_sz))
    plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
    plt.show()
{% endhighlight %}

Output matrix’s first dimension is also a batch size. Output matrix’s second dimension is the number of classes/labels (i.e. 10 digits in MNIST).

Input values are shuffled so that different digit numbers can be mixed within a training batch. However, a testing set does not get shuffled. Without standardising input values, the learning process is too slow. Thus, input values are standardised around mean 0.0 and standard deviation 1.0 for each batch prior to forwarding through the network.

Two loss functions are implemented: Quadratic cost function and Cross entropy cost function to compare the performance between two.

Similarly, without standardising weights and biases, the learning process is tedious. Therefore, random weights and biases are initialised with mean 0.0 and standard deviation 1.0. To aim for a better initialisation, [Kaiming initialisation](https://arxiv.org/abs/1502.01852) is implemented by multiplying √(2/(input size)) to the random initialisation. The comparison of the performance is listed in the result section.

A function to reduce the learning rate as the epoch increases is implemented to ensure that the gradient steps are reduced when it is closer to the optimum. Output labels are vectorized through an one-hot encoding method to compare it with the prediction and obtain accuracy.

## The implementation of NN class
The NN class is implemented with:
1. Forward function;
2. Backward function (backpropagation);
3. Random and Kaiming initialisation of weights and biases; and
4. Quadratic and Cross Entropy cost function;

### Forward function of NN class
As this NN class consists of only one hidden layer and one output layer, a forward function is quite straight forward.

![Forward formula](../../../../../../assets/images/Forward_formula.png)

{% highlight  Python%}
#main function - forward implemented with vectorisation
def forward(self, x):
    #hidden layer
    self.neth = np.dot(x, self.w1.T) + np.dot(self.b, self.b1.T)
    self.outh = self.sigmoid(self.neth)
    #output layer
    self.neto = np.dot(self.outh, self.w2.T) + np.dot(self.b, self.b2.T)
    self.y_hat = self.sigmoid(self.neto)
{% endhighlight %}

Activation function in this case is a sigmoid function:

![Sigmoid formula](../../../../../../assets/images/Example_of_sigmoid.png)

{% highlight  Python%}
#sigmoid activation
def sigmoid(self, x):
    return 1 / (1+np.exp(-x))
{% endhighlight %}

### Backward function (backpropagation) of NN class
Backpropagation uses a chain rule to step backward and adjusts weights and biases of the current NN layers.
The example of the chain rule implemented is:

![Example_backprop](../../../../../../assets/images/Example_of_backprop.png)

When the formula is implemented with codes, it does not look too bad.

{% highlight  Python%}
#main function - back propagation implemented with vectorisation
def backward(self, x, y, lr):
    #last batch may have lesser samples than the specified batch size
    d_w2 = np.dot(self.outh.T, ((1/x.shape[0]) * (self.y_hat - y) * self.d_sigmoid(self.y_hat)))
    d_b2 = np.dot(self.b.T, ((1/x.shape[0]) * (self.y_hat - y) * self.d_sigmoid(self.y_hat)))
    d_w1 = np.dot(x.T, (np.dot((1/x.shape[0])  * (self.y_hat - y) * self.d_sigmoid(self.y_hat), self.w2) *
             self.d_sigmoid(self.outh)))
    d_b1 = np.dot(self.b.T, (np.dot((1/x.shape[0])  * (self.y_hat - y) * self.d_sigmoid(self.y_hat), self.w2) *
             self.d_sigmoid(self.outh)))
    #update the weights with a learning rate
    self.w1 -= lr * d_w1.T
    self.w2 -= lr * d_w2.T
    self.b1 -= lr * d_b1.T
    self.b2 -= lr * d_b2.T
{% endhighlight %}

To apply a chain rule here, the beautiful derivative of a sigmoid function needs to be defined as:

![Derivative_sigmoid](../../../../../../assets/images/Derivative_sigmoid.png)

{% highlight  Python%}
#sigmoid derivative
def d_sigmoid(self, out):
    return np.multiply(out, (1 - out))
{% endhighlight %}

### Initialisation of weights and biases of NN class
Before starting to train, we will need to initialise weights and biases of a hidden layer.
There are two initialisation methods implemented here: 1) Random initalisation; and 2) Kaiming initialisation.

Random initalisation is set with mean 0.0 and standard deviation 1.0.

{% highlight  Python%}
#random initialisation with mean 0. and std 1.
def random_init(self, bs):
    #make sure self.w.T first dimension aligns with self.x second dimension
    assert self.n_input == self.x.shape[1], "input numbers not equal to matrix dimension"
    mean = 0.0
    std = 1.0
    self.w1 = np.random.normal(mean, std, (self.n_hidden, self.n_input))
    self.w2 = np.random.normal(mean, std, (self.n_output, self.n_hidden))
    self.b = np.ones((bs, 1))
    self.b1 = np.random.normal(mean, std, (self.n_hidden, 1))
    self.b2 = np.random.normal(mean, std, (self.n_output, 1))
{% endhighlight %}

Kaiming initialisation has an additional step by multiplying √(2/(input size)) to each randomised weights.

{% highlight  Python%}
#kaiming initialisation
def kaiming_init(self, bs):
    #initialise with mean 0 and std 1 multiplied by square root two divided by input size
    mean = 0.0
    std = 1.0
    self.w1 = np.random.normal(mean, std, (self.n_hidden, self.n_input))*math.sqrt(2./self.n_input)
    self.w2 = np.random.normal(mean, std, (self.n_output, self.n_hidden))*math.sqrt(2./self.n_hidden)
    self.b = np.ones((bs, 1))
    self.b1 = np.random.normal(mean, std, (self.n_hidden, 1))
    self.b2 = np.random.normal(mean, std, (self.n_output, 1))
{% endhighlight %}

### Cost functions of NN class
There are two loss/cost functions that are implemented: Quadratic and Cross entropy.

![Quad_loss](../../../../../../assets/images/Quadratic_loss.png)

{% highlight  Python%}
#quadratic loss function
def quad_loss(self, y):
    #1/2 as part of matrix mean operation
    return np.square(self.y_hat - y).mean()
{% endhighlight %}

Cross entropy loss is implemented as follow:

![Cross_entropy_loss](../../../../../../assets/images/Cross_entropy.png)

{% highlight  Python%}
#cross entropy loss
def cross_entropy_loss(self, x, y):
    return -(y*np.log(self.y_hat)+(1.-y)*np.log(1.-self.y_hat)).mean()
{% endhighlight %}

## Results of a small neural network on the MNIST data
Given all other parameter are equivalent, Kaiming initialisation provides marginally better accuracy. The network with Kaiming initialisation starts at higher accuracy and improves more quickly. However, regardless of different initialisation methods, the gap between the training and test accuracy becomes wider as the number of epochs increases. This may be due to overfitting starting to occur with the training data.

![Result_quad](../../../../../../assets/images/Result_quad.png)

![Result_quad_plot](../../../../../../assets/images/Result_quad_plot.png)

### Different learning rates
With a small learning rate (lr) such as 0.001, the learning process is very slow (in particular, for the model with random initialisation) and it still displays underfitting symptoms after 30 epochs for both models. The lr 0.1 improves slower than the performance with the learning rate 3.0 but this learning rate seems to be appropriate (95.18%) as the accuracy reaches to a higher performance compared to the performance of 94.81% with lr = 3.0 for the network with Kaiming initialisation. The lr 1.0 is similarly effective while the lr 10 performs as worse as lr = 0.001.

When lr is 100, the network performance does not improve better than that of the smaller learning rates for both models. 

![lr_table](../../../../../../assets/images/lr_changes_table.png)

Overall, the learning rate of 0.1 and 1 provides the better accuracy for the model with Kaiming initialisation while the learning rate of 1 and 10 works better for the model with random initialisation.

![lr_table](../../../../../../assets/images/lr_changes_plot.png)

Lr of 100 performs terribly because the network cannot perform the gradient descent properly. With the too large learning rate, the model’s convergence cannot be made as it only goes back and forth near the similar points instead of descending towards the optimum.

![Large_learning_rate](../../../../../../assets/images/Large_lr.png)

### Cross entropy loss function
The both networks achieve a similar accuracy with the cross-entropy cost function. Despite small improvements in the accuracy for the model with Kaiming initialisation, according to [Nielsen (2018)](http://neuralnetworksanddeeplearning.com/chap3.html) , the cross-entropy cost function can be resilient to the issues associated with the quadratic cost function’s slowdown of learning when using many-layer and multi-neuron network. Hence, using the cross-entropy cost function makes sense when there is a neural network with many layers and nodes.

![Result_cross_entropy](../../../../../../assets/images/Result_cross_entropy.png)

### Combination of different hyper-parameters
The increased hidden node of 90 provides better accuracy of 97.20% on the test dataset. The accuracy of 97.20% is reached within 30 epochs, thus, the number of epochs over 30 does not make a difference.

Without increasing the number of nodes in a hidden layer, the maximum accuracy is 95.74% with epochs=30, lr=1.0, bs=20, and lr decay with the cross-entropy cost function.

![Result_different_hyper_parameters](../../../../../../assets/images/Result_different hyper-parameters.png)

Overall, it seems to be important to identify appropriate hyper-parameters for a given data set and a model for a task through experiments. 

For the mnist dataset, the neural network with one hidden layer and one output layer performs better with a cross-entropy cost function, Kaiming initialisation, learning rate decay, and the settings of epochs=30, lr=1.0, bs=20, and hidden nodes=90 in my experiments.
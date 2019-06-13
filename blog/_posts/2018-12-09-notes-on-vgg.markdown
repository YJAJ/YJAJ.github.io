---
layout: post
author: YJ Park
title:  "Notes on VGG"
date:   2018-12-09 10:50:00 +1000
categories: jekyll update
tags: VGG, Karen Simonyan, Andrew Zisserman, 2014
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

While our small deep learning study group was playing with visualisation techniques of layers, we found that many methods were still based on VGG-like architecture. 
This week, therefore, we decided to read through ["Very deep convolutional networks for large-scale image recognition" by Simonyan and Zisserman (2014)](https://arxiv.org/abs/1409.1556) to study a VGG model.

Many studies around this time focused on improving hyper-parameters of Convolutional networks (ConvNets). This paper also investigated one of the most important hyper-parameter - the effect of the ConvNets depth.

## Why is this paper important and interesting?

* VGG won the first and second place in the localisation and classification challenges from ILSVRC-2014.
* VGG experiments on the ConvNets depth by changing and comparing ConvNets with different layers.

## What does VGG(s) look like?

ConvNet layer configuration was inspired by [Ciresan et al. (2011)](https://arxiv.org/abs/1102.0183) and [Krizhevsky et al. (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

The base VGG has the similar configuration from AlexNet, but with the deeper layers (11 layers) and the smaller filters (3 x 3). The smaller filters were sized as 3 x 3 because this was the smallest size to capture the notion of left/right, up/down, and centre. 

To experiment of the effect of the depth, the paper compare the six different models below:

* VGG model A: 8 conv layers, 3 fc layers
* VGG model A-LRN: 8 conv layers (but with **Local Response Normalisation** from AlexNet), 3 fc layers
* VGG model B: 10 conv layers, 3 fc layers
* VGG model C: 13 conv layers (but with **2 1 x 1 linear transformation with ReLU**), 3 fc layers
* VGG model D: 13 conv layers (all 3 X 3 conv layers), 3 fc layers
* VGG model E: 16 conv layers (all 3 x 3 conv layers), 3 fc layers

![models Table 1](../../../../../../assets/expressions/VGG-config.png)

Source: [Simonyan and Zisserman (2014), p. 3, Table 1](https://arxiv.org/abs/1409.1556)

An interesting point I want to highlight here is that the activation maps' size gone through the larger sizes of filters (e.g. 11 X 11 and 7 x 7) from AlexNet and ZFNet were equivalent with the activation gone through a stack of 3 x 3 filters in VGG. As illustrated below, a stack of two 3 x 3 conv layers has an effective receptive field of 5 x 5 (and the three of such layers are equivalent with one 7 x 7 layer).

![image of filter size](../../../../../../assets/images/VGG-filter-size.png)

So why do we need a stack of 3 x 3 filters instead of one larger filter?

A greater number of layers could provide more discriminative features and help to decrease the number of parameters. 
As you could see below, the largest number of parameters is around 144 millions from VGG E (19 layers), of which size is similar to the parameters of the shallow model with larger filters.

![models Table 2](../../../../../../assets/expressions/VGG-parameters.png)

Source: [Simonyan and Zisserman (2014), p. 3, Table 2](https://arxiv.org/abs/1409.1556)

## How did the authors train VGG?

### Training methods
VGG used a mini-batch size of 256, momentum 0.9, weight-decay penalty multiplier 5e-4, and drop out for the first two fc layers with the probability of 0.5. As usual, the learning rate started from 1e-2 and was decreased by a factor of 10.

Another interesting point in VGG was around the initialisation method. Because this model was developed prior to the batch normalisation adoption, the model was initially trained with only 11 layers (i.e. VGG A) to stabilise the gradients. The first four conv layers and the last three fc layers were then taken out and other intermediate layers were slided into the models with deeper layers. Intermediate layers were initialised randomly with the zero mean and the standard deviation of 1e-2.

### Training image size

* Single-scale training: two fixed crop scales were used - 224 and 384 to compare the results.

* Multi-scale training: each training image was rescaled between _S_ min (256) and _S_ max (512). This could be seen as training set data augmentation by scale jittering.

The difference between the two method can be highlighted as the second approach makes the model to be trained to classify an object a broader range of scales.

The model was trained with 4 NVIDIA Titan Black GPUs, taking about 2-3 weeks for a single net training (subject to the type of the architecture).

## What were the results of VGG?
Dataset used was the ILSVRC-2012 (images of 1000 classes, training set of 1.3m, validation set of 50k, and test set of 100k).

Top-1 error (proportion of incorrectly classified images) and Top-5 error (proportion of images such that the ground-truth category is outside the top-5 predicted category) were compared to see the results of VGG.

### Single-scale evaluation

* It turned out that A-LRN (11 layers with local response normalisation) was not performing better than the model without LRN (i.e. model A) so the authors decided not to use LRN.
* A classification error decreases with the increased ConvNet depth, meaning two 3 x 3 conv layers perform better than one 5 x 5.
* As seen below, the deeper model (model C) with non-linearity additions help (1 x 1 conv layer with ReLU) but capturing more spatial contexts (model D) using 3 x 3 filters performs better than just 1 x 1 linear transformation and ReLU (model C). Overall, this is why the model D (deep, 3 x 3 filter) performs better than the model C (deep, 1 x 1 filter) that performs better than the model B (shallower).
* Scale jittering at training time also helps to improve the performance.

![models Table 4](../../../../../../assets/expressions/VGG-single-performance.png)

Source: [Simonyan and Zisserman (2014), p. 6, Table 4](https://arxiv.org/abs/1409.1556)

### Multi-scale evaluation

* Scale jittering at test time turned out to be helpful (ranged between _S_ min, 0.5(_S_ min+_S_ max), _S_ max) rather than just using fixed sizes at test time.

![models Table 5](../../../../../../assets/expressions/VGG-multi-performance.png)

Source: [Simonyan and Zisserman (2014), p. 7, Table 5](https://arxiv.org/abs/1409.1556)

### Comparison with other models

When compared to other models with good performance, VGG ensembled of two best-performing multi-scale models (model D - 16 layers and Model E - 19 layers) outperformed previous state-of-the-art models, other than GooLeNet (another 2014 ILSVRC-2014 winner). Top-5 validation error and test error was at 6.8% on the classification challenge.

![models Table 7](../../../../../../assets/expressions/VGG-comparison.png)

Source: [Simonyan and Zisserman (2014), p. 8, Table 7](https://arxiv.org/abs/1409.1556)

## Lessons learnt and future to-do-list
This paper was where famous VGG16 and VGG19 were born. Even now in 2018, there are many papers that use VGG-like models to experiment previously-unexplored research areas.
Unfortunately, this week I was not able to implement VGG. 

- [ ] I want to implement VGG in Pytorch later.
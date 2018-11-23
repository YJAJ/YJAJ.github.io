---
layout: post
author: YJ Park
title:  "Notes on ZFNet (Paper title: Visualizing and Understanding Convolutional Networks)"
date:   2018-11-18 10:50:00 +1000
categories: jekyll update
tags: ZFNet, Zeiler and Furgus, 2013, Matthew Zeiler, Rob Furgus, Visualization of CNN
---

This post focuses on understanding [the paper](https://arxiv.org/abs/1311.2901) "Visualizing and Understanding Convolutional Networks". To explore core concepts presented in the paper, some better explanations have been adopted from [fast.ai](https://www.fast.ai/) and [Standford CS231N](http://cs231n.stanford.edu/2017/).

This paper was interesting because authors 1) placed efforts to visualise convolutional layers (which was a kind of a black box) and 2) used this visualisation to find better hyperparameters of the existing architecture, AlexNet. In the abstract, the motivations were articulated as identifying 'why large Covlutional Network (CNN) work well' and 'how they might be improved', so this paper focuses on the details around 'how' and 'why' through a good range of experiments.

## Main characters of ZFNet
As ZFNet (Zeiler Furgus Net) seems to be an enhanced modification of AlexNet through investigation of internal operations and behaviours of the model, many characters were addressed in comparison with AlexNet.
* Visualisation techniques used as a diagnostic role
* Ablation study to identify performance contribution from different model layers (e.g. cnn vs fc layer - which contributes more)
* Sensitivity analysis through occluding portions of an image
* Improved hyperparameters as a result of futher investigations above
* Reduced Top 5 error rate to 14.8% with 6 convnets compared to AlexNet (best result 15.3% with 7 convnets) on ILSVRC-2012
* Transfer learning to datasets other than ImageNet displaying generalisation ability of the model

## How did convolutional layers get visualised prior to this paper?
In the past, first layers tended to be more frequently visualised because outputs could be more easily projected to the pixel space. The optimal stimulus for each unit needed to be found by performing gradient descent so that activations were maximised. 

To do so, this required a careful initialisation and made sure it does not give large invariances. This meant that higher layers with greater invariances were considered extremely complex to be visualised.

## Why was this paper different from previous visualisation techqniues?
Through [Deconvolutional Network (Zeiler et al., 2011)](http://www.matthewzeiler.com/wp-content/uploads/2017/07/iccv2011.pdf), this paper visualised features to pixels to map feature activities back to the input pixel space for a relevant layer. There are three components to be understood in deconvnet.

### 1. Unpooling

Maxpooling is not invertible, but it can be approximated by recording the locations of maxima, preserving structure of the stimulus.

![image of unpooling](../../../../../../assets/expressions/Unpooling.png)

### 2. Rectification

Unpooled feature reconstructions (that are always postive) go through ReLU.

### 3. Filtering

Conv layer uses defined filters to convolve output from the previous input. This could be simplifed as:

> Input @ Filter = Output 

where @ is matrix multiplication.

Therefore, deconvnet used the transposed versions of the same filters on the output from Rectification above (2. Rectification), which was simplifed as:

> Reconstructed Input = Output @ Transposed Filter

With these three components of deconvnet, authors were able to visuallise all layers in AlexNet.

## What were main changes to AlexNet?

* First layer filters were changed their size from 11 * 11 to 7 * 7 and its stride became 2 instead of 4.
* Dense connections were used on layer 3, 4, and 5 from AlexNet because ZFNet was trained on a single GTX580 GPU.
* After visualising the first layer filters, authors realised that a few of the layer filters dominated so they renormlise each filter in the conv layers where the root mean square value exceeds a fixed radius of 1/10 to the current fixed radius.

## How does ZFNet look like?

The size of each input, filter, and output per layer is visualised on Excel.

The number below each conv layer, ReLU, maxpool, and fc layer indicates the corresponding line from the codes below.
Excel version is located [here](https://github.com/YJAJ/Deep_learning_studies/blob/master/ZFNet.xlsx).

![image of ZFNetSize](../../../../../../assets/expressions/ZFNetSize.png)

The codes below represent my attempt to create ZFNet through Pytorch based on AlexNet. The full version with an example is located [here](https://github.com/YJAJ/Deep_learning_studies/blob/master/ZFNet-babies.ipynb).

{% highlight  Python%}
class ZFNet(nn.Module):
    # num_classes changed based on ILSBRC class name
    def __init__(self, num_classes=1000):
        super(ZFNet,self).__init__()
        # conv layers
        self.feature = nn.Sequential(
            #first set
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1), #1
            nn.ReLU(inplace=True), #2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #3
            #second set
            nn.Conv2d(96, 256, kernel_size=5, stride=2), #4
            nn.ReLU(inplace=True), #5
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #6
            #thrid set
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), #7
            nn.ReLU(inplace=True), #8
            #fourth set
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), #9
            nn.ReLU(inplace=True), #10
            #fifth set
            nn.Conv2d(384, 256, kernel_size=5, stride=1, padding=1), #11
            nn.ReLU(inplace=True), #12
            nn.MaxPool2d(kernel_size=3, stride=2), #13
        )
        # fully-connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            
            #sixth set
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(),
            
            # seventh set
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            # last layer
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), 256 * 6 * 6) # to resize to match to matrix shape for the next linear layer
        x = self.classifier(x)
        return x
{% endhighlight %}


## What did we learn from visualisation of layers?

### Feature Visualisation

Layer 2 visualised corners and other edge/color conjunctions while Layer 3, 4, and 5 displayed more complex features with larger invariances, such as similar textures and text, class-specific, and entire objects, respectively. I wanted to include the visualisation of layers from the paper here, but I was not sure about the license for these figures. For your information, it is _Figure 2. Visualization of features in a fully trained model._

### Feature Evolution during Training

Authors found that lower layers converged within a few epochs while upper layers had convergence after a considerable number of epochs. This indicates that recognising an object/class takes a greater time than simple features.

### Feature Invariance

Interestingly, small transforms affected lower layers greatly whilst upper layers got affected lesser.

## How did it affect performance when some portions of images are occluded?

Where an object was occluded, probability of the corret class dropped significantly, indicating that the model really focused on the location of the object in an image.

## What was the result on ILSVRC-2012?

ZFNet reduced Top-5 error rate on the test set by 0.5%, making it 14.8% (6 convnets) compared to AlexNet's best result 15.3% (7 convnets). Through experiments in changing structures, authors found that adding a middle conv layer gained in performance whilst adding an fc layer made little difference. Adding both layers resulted in over-fitting.

## How did it go with other datasets?

Authors also experimented feature generalisation by using pre-trained seven layers on the ImageNet dataset and adding the last softmax layer (i.e. classification layer) individually trained on a relevant data set.

### Caltech-101 and Caltech-256

The pre-trained model outperformed previous best results while the model trained from scratch performed poorly.

### Pascal 2012

Pascal 2012 images can have multiple objects in an image while ImageNet datasets focus on a single classification per image, indicating images from these two sources are quite different in nature. Potentially because of this, the pre-trained model performed worse than the best results avoailable for Pascal 2012 images.

### Feature Analysis

As it was implied in AlexNet, authors found that deeper feature hierarchies tended to learn more powerful features.

## Lessons learnt and future to-do-list
Through reading the paper, I became more curious about visualising each output - disecting the structure and looking at what each layer produces would enable better understanding of the current model that I would use for problem solving. I looked around more to identify these techniques and found that there is a plenty of papers out there. For the next task, I would like to focus on this visualisation by reading through [Zeiler et al., 2011](http://www.matthewzeiler.com/wp-content/uploads/2017/07/iccv2011.pdf) and [Selvaraju et al., 2016](https://arxiv.org/abs/1610.02391). In particular, Selvaraju et al., 2016 would be interesting to study further because their approach, Grad-CAM, enables to investigate Resnet-based models.
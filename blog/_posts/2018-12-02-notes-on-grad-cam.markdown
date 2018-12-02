---
layout: post
author: YJ Park
title:  "Notes on Grad-CAM (Paper title: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization)"
date:   2018-12-02 19:35:00 +1000
categories: jekyll update
tags: Grad-CAM, Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra, Visual Explanations, 2016
---

This week, our small deep learning study group decided to focus on a visualisation of layers since both of us wanted to see how the model sees through images when doing a classification task.


Computer vision tasks have been diversified into more than a labelling task, branching into 1) image classification (traditional single or multi labelling tasks); 2) object detection (localisation of an object); 3) semantic segmentation (pixel-wise localisation); 4) image captioning; and 5) visual question answering (VQA).

_Example of Semantic segmentation (3)_

![image of semantic segmentation](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRCAKpr3EHEEMooof6NzhnTvy-6aMND5rHHk53ymA-bOY478I9XkA)

Source: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRCAKpr3EHEEMooof6NzhnTvy-6aMND5rHHk53ymA-bOY478I9XkA 

_Example of Image captioning (4)_

![image of dense captioning](../../../../../../assets/images/Johnson_et_al_2015_Figure3.png)

Source: [Johnson et al., 2015](https://arxiv.org/abs/1511.07571) Figure 3 Example captions generated and localized by our model on test images. (from top-right corner)

_Example of Visual question answering (5)_

![image of visual question answering](../../../../../../assets/images/Ren_et_al_2015_Figure3.png)

Source: [Ren et al., 2015](https://arxiv.org/abs/1505.02074) Figure 3 Sample questions and responses from our system.

To effectively design, implement and deploy these models in real life, the authors for this paper articulates that it is important to focus on interpretability and transparency of the models:

1. to identify the failure modes (when the model performs worse than humans on a particular task(s));
2. to establish appropriate trust and confidence in users (when the model's performance is on par with humans); and
3. to teach how to make better decisions (when the model's performance is stronger than humans).

The interpretability and transparency of the models could provide 'why the models predict what they predict'.

## What would you consider a good visual explanation?
The authors defined two characteristics that make a good visual explanation: class discriminative and high resolution.

* Class discriminative means whether a visual explanation is provided for a localisation of an object of interest.

* High resolution in this case represents a visual explanation that could capture fine-grained details.

On this post, we will compare different visualisation techniques based on these two criteria to evaluate the extent of explanatory power.


## Why do we need Grad-CAM over other visualisation techniques?
In the past, visualisation was done through Guided Backpropagation ([Springenberg et al. (2014)](https://arxiv.org/abs/1412.6806)) or Deconvolution ([Zeiler and Furgus (2014)](https://arxiv.org/abs/1311.2901)). These visualisation techniques tended to provide a high resolution but they could not localise the object(s) of interest.

Grad-CAM is based on Class Activation Mapping (CAM) by [Zhou et al. (2015)](https://arxiv.org/abs/1512.04150) where CAM could identify discriminative regions for a particular class through CNNs.

Grad-CAM complements/generalises CAM because CAM cannot visualise localised regions through models other than CNNs whilst Grad-CAM could visualise regions not only through CNNs but also from other models such as CNNs with fully-connected layers (e.g. VGG), CNNs used for structured outputs (e.g. captioning) and CNNs used in tasks with multi-modal inputs (VQA) or reinforcement learning.

In addition, Guided Grad-CAM, combining Guided Backpropagation and Grad-CAM together, could provide both high resolution and localisation through point-wise multiplication of two different outputs.

_Comparison between Guided Backpropagation, Grad-CAM, and Guided Grad-CAM_

![image of comparison](../../../../../../assets/images/Selvaraju_et_al_2016_Figure1.png)

Source: [Selvaraju et al., 2016](https://arxiv.org/abs/1610.02391) Figure 1.

As you can see above, Guided Backpropagation (b) does not distinguish the object of interest from other objects from the image (i.e. a dog and a cat got visualised together).

Grad-CAM (c) focuses on cat for a cat classification but it does not provide any texture or feature of this cat (i.e. it only localise a cat).

Guided Grad-CAM (d) displays cat only but it also has a stripe-texture of the cat at the same time.

## How does Grad-CAM look-like?
![expression representing Grad-CAM](../../../../../../assets/expressions/Grad-CAM.png) expresses Grad-CAM where this expression represents "**the class-discriminative localisation map Grad-CAM**" with width _u_ and height _v_ for any class _c_.

There are two steps involved in calculating Grad-CAM.

> Step 1 ![expression for step1](../../../../../../assets/expressions/Grad-CAM_step1.png)


where represents a partial linearisation of the deep network downstream from A, and captures the importance of feature map _k_ for a target class _c_.

* In Step 1, the gradient of the score for class _c_, ![expression for prediction y for class c](../../../../../../assets/expressions/Grad-CAM_predy.png) before the softmax, is divided by feature map ![expression for feature map](../../../../../../assets/expressions/Grad-CAM_feature_map.png). 

* Then these will be global-average-pooled to get the neuron importance weight ![expression for the neuron importance](../../../../../../assets/expressions/Grad-CAM_neuron_importance.png).

In Step 2, to get Grad-CAM, do ReLU on this neuron importance weight is multiplied by feature maps. The authors applied ReLu to linear combination in order to increase the gradient of the score, ![expression for prediction y for class c](../../../../../../assets/expressions/Grad-CAM_predy.png), for a class of interest (i.e. pixels whose intensity should be increased).

> Step 2 ![expression for step2](../../../../../../assets/expressions/Grad-CAM_step2.png)

Source: [Selvaraju et al., 2016, pp. 3-4](https://arxiv.org/abs/1610.02391)

## Implementation of Grad-CAM through fast.ai library
The implementation code of Grad-CAM is adapted from Lecture 6 fast.ai course. The image dataset was collected from the web.

The full notebook can be found [here](https://github.com/YJAJ/Deep_learning_studies/blob/master/Insects_classification_with_Resnet_GradCAM.ipynb).

*Credit: A large part of this code is based on code from a fast.ai MOOC that will be publicly available in Jan 2019.*

![image of Grad-CAM on insect dataset](../../../../../../assets/images/Grad-CAM_on_insects.png)

As you can see from the notebook, heatmap indicates the object of interest within an image.

## Lessons learnt and future to-do-list
Guided Backpropagation and Grad-CAM explains where the model focuses on when labelling. However, Guided Backpropagation is missing localisation while Grad-CAM is missing high-resolution.

During the next week, I would like to implement Guided Grad-CAM to see both localisation and high-resolution of visualised layers.
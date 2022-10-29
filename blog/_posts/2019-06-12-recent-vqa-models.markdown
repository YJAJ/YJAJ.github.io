---
layout: post
author: YJ Park
title:  "Notes on recent VQA models"
date:   2019-06-10 10:50:00 +1000
categories: jekyll update
tags: Visual Question Answering models, VQA, models, tips, techniques
---
<head>
    <script defer data-domain="yjpark.me" src="https://plausible.io/js/plausible.js"></script>
</head>

_To get a brief look at VQA tasks and datasets, you can read through my previous post, ["Notes on VQA"](http://www.yjpark.me/blog/jekyll/update/2019/03/11/notes-on-vqa.html)._

The purpose of this post is to summarise trends in recent VQA models from:
- ["Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering" by Anderson et al. (2017)](https://arxiv.org/abs/1707.07998);
- ["Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge" by Teney et al. (2017)](https://arxiv.org/abs/1708.02711); and
- ["Pythia v0.1: the Winning Entry to the VQA Challenge 2018" by Jiang et al. (2018)](https://arxiv.org/abs/1807.09956).

These three papers address architectures and techniques of their selection of VQA models with the rationale behind them.

## Overview of the three papers
The first and second papers, Anderson et al. (2017) and Teney et al. (2017), seem to be built on the findings from each other.

### 1) Anderson et al. (2017)
The first paper, ["Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering" by Anderson et al. (2017)](https://arxiv.org/abs/1707.07998), highlights the integration bottom-up and top-down attention mechanisms to focus on salient regions in images. Interesting findings for this VQA model include:

* an object detector to get image features through bottom-up attention, utilising Faster R-CNN together with the Resnet-101 CNN , 
* bottom-up attention with the maximum number of regions to focus on up to 100, but selecting top 36 regions works well;
* bottom-up attention pre-trained on ImageNet, followed by Visual Genome data;
* bottom-up attention adding another training output for predicting attribute classes to enhance learning of good feature representations (in addition to an object prediction);
* soft top-down attention mechanism and gated tanh activations (which proved to be effective in Teney et al. (2017)); and
* when a candidate attention region is associated with related objects, all the visual instances relating to these objects are considered together.

The incorporation of attention mechanisms helps to identify a focus area of input questions. The images below displays one of the successful or failure cases with this VQA model:

![An example of the successful vqa Anderson et al. (2017)](../../../../../../assets/images/VQA_Anderson_S.png)

Successful case. Source: Anderson et al. (2017), p. 14. Figure 10. 

![An example of the failed vqa Anderson et al. (2017)](../../../../../../assets/images/VQA_Anderson_F.png)

Failure case. Source: Anderson et al. (2017), p. 15. Figure 11.

Although the model failed to count oranges correctly above, it still focuses on the correct object regions in the image above.

### 2) Teney et al. (2017)
The second paper, ["Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge" by Teney et al. (2017)](https://arxiv.org/abs/1708.02711), focuses on empirically exploring various architectural development techniques and hyper-parameters of VQA models, motivated by the question of "What makes a successful VQA model?". The main findings include using:

* sigmoid outputs instead of softmax outputs to enable multiple correct answers per question (removing competitions between answers, similar to an approach by [Mask R-CNN, p.3](https://arxiv.org/abs/1703.06870));
* soft target scores instead of binary targets to pass slightly richer information, which may be more helpful in the presence of a greater uncertainty;
* image features from bottom-up attention to highlight on salient regions;
* gated tanh activations for all non-linear layers;
* transfer learning through output embeddings initialised using 300-dimension Global Vectors for Word Representation (GloVe) and Google Images (similar to Anderson et al. (2017)); 
* a mid range size of mini-batches (e.g. 256, 384 or 512); and 
* balanced pairs of VQA v2.0 in the same mini-batches (i.e. pairs that have identical questions with different images and answers) to encourage stable learning and means to differentiate subtle dissimilarities. 

The VQA models with the effective techniques found during their experiments placed the model as the winner for the VQA Challenge in 2017.

### 3) Jiang et al. (2018)
The last paper, ["Pythia v0.1: the Winning Entry to the VQA Challenge 2018" by Jiang et al. (2018)](https://arxiv.org/abs/1807.09956), which is short (3 pages), documents a number of changes made to the model from the previous two papers to win the first place in the 2018 VQA Challenge. The changes include:

[Pythia](https://github.com/facebookresearch/pythia) by Facebook Artificial Intelligence researchers (FAIR) is based on the findings addressed in this paper and it seems to be continuously improved and updated in their github.

* minor architectural change: 1) using weight normalisation followed by ReLU instead of gated tanh when pretraining the classifier; and 2) element-wise multiplication than concatenation when joining image and question features for attention;
* learning schedule: Adamax, a variant of Adam with infinite norm, with a learning rate of 0.002 and adopting a learning rate increase;
* fine-tuning bottom-up features with Feature Pyramid Network (Pythia now uses Mask-R-CNN as part of their VQA model as well);
* data augmentation with the aid of Visual Genome and Visual Dialogue;
* combining grid-level image features with the original Anderson et al. (2017)'s object proposals, in order to further integrate a holistic spatial information; and
* ensembling.

## Interesting aspects from the three papers
There are two aspects that I would like to focus on this section.

First of all, these VQA models seem to be heavily influenced by the recent progress made in object classification, detection, segmentation, and natural language processing.
Since VQA models are based on the joint features of the input questions and images, any progress developed in each field of computer vision and natural language processing helps to advance in attaining better performance from the integrated models. For instance, most recent bottom-up attention mechanisms explored in computer vision are actively adopted in many VQA models, only slightly varying from each other.

Secondly, the models from Teney et al. (2017) and Anderson et al. (2017) appear to be the same. 
According to each overview of the proposed model from the two papers (see below), a difference seems to be the pre-trained linear classifiers split after the element-wise product operation on embeddings of images and questions. However, upon further reading through the papers, Anderson et al. (2017) asks readers to refer Teney et al. (2017)'s paper on detailed implementation. Thus, it is unclear whether soft target scores were adopted in Anderson et al. (2017) or a different approach was adopted after the joint embeddings.

![Overview of the proposed model Anderson et al. (2017)](../../../../../../assets/images/VQA_Anderson_arch.png)
Source: Anderson et al. (2017), p. 5. Figure 4.

![Overview of the proposed model Teney et al. (2017)](../../../../../../assets/images/VQA_Teney_arch.png)
Source: Teney et al. (2017), p. 3. Figure 2.

Therefore, I think it is reasonable to consider these two papers as part 1 and part 2 of the larger topic where the findings make up the SOTA model in the 2017 VQA Challenge.
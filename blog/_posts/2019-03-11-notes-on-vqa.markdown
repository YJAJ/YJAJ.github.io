---
layout: post
author: YJ Park
title:  "Notes on Visual Question Answering (Paper title: Visual Question Answering: Datasets, Algorithms, and Future Challenges)"
date:   2019-03-11 10:50:00 +1000
categories: jekyll update
tags: VQA, Kushal Kafle, Christopher Kanan, 2016
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

The purpose of this post is to summarise trends in Visual Question Answering (VQA) as to:
 * what datasets are available;
 * what models are used for exploring VQA; and
 * what gaps are arising from adopting available datasets and models.

## What is VQA?
VQA is a computer vision task where a model is expected to infer the answer from the provided images and associated text information.
To do so, VQA requires identifying objects, understanding spatial positions, inferring attributes and relationships, and localising objects with surrounding contexts.
VQA, therefore, is a means to human-computer interaction where the goal is to extract question-relevant semantic information from images.

Image captioning is the field close to VQA tasks, to produce a natural language description of a given image aiming to describe complex attributes and object relationships.
In image captioning, this is achieved through annotating an image with local regions such as bounding boxes and pixel-level local areas. 
In general, localisation of objects is addressed in image captioning, but relationships may have been overlooked in certain models. 
Unlike image captioning, VQA's level of granularity depends on the nature of the question asked. VQA could provide specific and unambiguous answers, making it more favourable to automated evaluation metric.

## Dataset for VQA
There are seven major datasets available for VQA tasks: 
1) Dataset for question answering on Real-word images (DAQUAR); 
2) Microsoft Common Objects in Context (COCO-QA);
3) The VQA Dataset;
4) Freestyle Multilingual Image Question Answering (FM-IQA);
5) Visual7W;
6) Visual Genome; and
6) SHAPES.

Apart from DAQUAR, all other dataset includes images from COCO dataset that has 328,000 images, 91 common objects categories, with over 2 million labelled instances, and average 5 captions per image.

### DAQUAR
DAQUAR dataset was the first pioneering main dataset released for VQA. It has a relatively small dataset (6,795 training, 5,673 testing QA pairs). The images are biased towards indoor scenes while some clutter or extreme lighting conditions are observed. Due to this, the accuracy of human annotator is around 50.2% on the full dataset.

### COCO-QA
This dataset's QA pairs were created from COCO image captions through an Natural Language Processing (NLP) algorithm. It has 78,736 training, 38,948 testing QA pairs, consisting of querying objects, colours, counts, and locations. However, there are many single word answers, awkwardly phrased questions with many containing grammatical errors, and some are unintelligible.

![image of grammatical errors](../../../../../../assets/images/Kafle_and_Kanan_2016_Figure2.png)

Source: [Kafle and Kanan, 2016](https://arxiv.org/abs/1610.01465) Figure 2. Sample images from DAQUAR and the COCO-QA datasets and the corresponding QA pairs.

### VQA dataset
VQA dataset includes real-world images and three questions per image with ten answers per each question. The uniqueness of this dataset involves synthetic images (cartoon images), which provides a more varied and balanced dataset. Considering diversity and balance in the dataset is important because it helps VQA models to learn in a less-biased way. For example, natural image datasets tend to have more consistent contexts and biases, therefore, a street scene is more likely to have a picture of a dog than a zebra - human annotators are still more likely to recognise a zebra regardless of the contexts provided.

Similar to other datasets, however, many of questions can be answered accurately without using the images. There are some subjective, opinion seeking questions while some human annotators unreliably said 'Yes' incorrectly to certain questions when building up the dataset. Also, the dataset consists of 38% of simple 'Yes/No' questions.

### FM-IQA
FM-IQA contains human-generated questions and answers. However, these answers are expected to be full sentences, which makes common machine-evaluated metrics intractable and impractical.

### Visual Genome
This large dataset includes 108,249 images, 1.7 million QA pairs, and average 17 QA per image. It also attempts to incorporate relatively complex questions (e.g. open-ended question), composed of: what, where, how, when, who, and why questions.
The dataset was developed by asking human annotators to focus on specific image regions so that it could provide greater answer diversity and specifically excluded binary questions. Due to this reason, the challenges associated with open-ended evaluations still exist. 

### Visual7W
This dataset is a subset of Visual Genome, specifically aiming to request region information. With 47,300 images, this dataset asks 'which' questions, requesting models to select a correct bounding box and asking multiple choice questions with plausible answers. The plausible answers were built from prompting annotators to answer questions without seeing the image.

![image of Visual Genome and Visual7W](../../../../../../assets/images/Kafle_and_Kanan_2016_Figure4.png)

Source: [Kafle and Kanan, 2016](https://arxiv.org/abs/1610.01465) Figure 4. A sample of Visual7W and Visual Genome images and QA pairs.

### SHAPES
SHAPES are made up of all synthetic images and QA pairs, which displays shapes of varying arrangements, types, and colours. Although images shown are simple shapes, the questions are quite complex, asking attributes, relationships, and position of shapes. Because they are synthetic, vast amount of data is available and free of biases.

Algorithm that cannot perform well on SHAPES but performs well on other datasets may indicate that it is only capable of analysing images in a limited manner because of the prevailing biases in real image datasets.

![image of Visual Genome and Visual7W](../../../../../../assets/images/Kafle_and_Kanan_2016_Figure5.png)

Source: [Kafle and Kanan, 2016](https://arxiv.org/abs/1610.01465) Figure 5. This graph shows the long-tailed nature of answer distributions in newer VQA datasets. For example, choosing the 500 most repeated answers in the training set would cover a 100% of all possible answers in COCO-QA but less than 50% in the Visual Genome dataset. For classification based frameworks, this translates to training a model with more output classes.


## VQA algorithm
Baseline models helps to determine the difficulty of a dataset and establish the minimal level of performance. To solve VQA challenges, generally three steps are undertaken in most VQA algorithm.

1. Extract image features: [Convolutional Neural Network (CNN)](http://yjpark.me/blog/jekyll/update/2018/11/09/notes-on-alexnet.html) that are pre-trained on ImageNet are mostly used. These include [VGG](http://yjpark.me/blog/jekyll/update/2018/12/09/notes-on-vgg.html), ResNet, and googLeNet. 
2. Extract question features: These models include Bag-Of-Words, Long-Short Term Memory, Gated Recurrent Units, and Skip-Thought Vectors.
3. Combine these features to produce an answer: Output from the first and second step were combined through simple concatenation, element-wise multiplication or addition, or bilinear pooling. To generate an answer, VQA was treated like a classification problem using linear classifier or neural networks.

### Bayesian and question-aware models
As VQA needs inferences and modelling relationships between images and questions, Bayesian frameworks help to draw semantic segmentation and train images and texts to model spatial relationships of the objects.
Though this is an interesting idea, performance is not superior potentially because the result of semantic segmentation is imperfect.

### Attention based models
Using global features alone may not provide sufficient question-relevant regions of the input space. Many recent models such as Stacked Attention Network (SAN) and Dynamic Memory Network (DMN) used spatial attention to create local CNN features from the images provided while some models also used attention mechanisms to extract text features. The assumption of these models is that certain visual regions in a image and certain words in a question are more informative than others for answering a given question. For example, the question of 'what colour is the umbrella?' would require more focus on 'colour' and 'umbrella'.

### Bilinear Pooling Methods
Early models tended to combine the image and the question with simple concatenation. [Multi-modal Compact Bilinear Pooling](https://arxiv.org/abs/1606.01847) showed promising advances in VQA performance by approximating the outer product between image and text features. Adding to this progress, [Multi-modal Low-Rank Bilinear Pooling](https://arxiv.org/abs/1610.04325) achieves similar performance with less computationally expensive (e.g. fewer parameters) methods.  

### Compositional VQA models
Compositional VQA models were motivated to address multiple steps of reasoning to the answers for each question. To illustrate, answering 'what is the left of the horse?' would require finding the horse and naming the object on the left side of it. [Neural Module Network (NMN)](https://arxiv.org/abs/1511.02799) takes external question parsers to find the sub-tasks in the question (i.e. break down into a sequence of sub-tasks). These sub-tasks are then carried out through separate neural sub-networks (e.g. network of 'find', 'describe', 'measure', 'transform').


In general, CNN based models tended to display better performance than Bayesian and compositional architectures despite their interesting theoretical motivations. 

## Discussion points
From the aforementioned datasets and models, five important discussion points need to be considered when developing a new VQA model:

1. Vision vs Language: Ablation studies revealed that the model trained with questions-only performs superior than image-only, indicating predictive power of language over image. 
This is potentially because questions are more prone to constrain the kinds of answers that could be provided and dataset tended to have strong bias in images and QA pairs.
This was further proved by the fact that the models were sensitive to the way questions are phrased and displayed varied accuracy where the same images were used.
It is implied that the models overly rely on language 'clues'. 

2. Essential attention?: The models with attention mechanisms tended to show better performance, but this was not always the case. To illustrate, when combining image and text features, the model with simple concatenation and attention mechanism did not perform better than the ones with element-wise multiplication and addition without attention mechanisms. This may be because attention mechanisms help to focus on discriminative regions rather than where the model actually should attend.

3. Effect of bias on method evaluation: Harder questions such as 'why' questions are rare and biases in existing datasets severely impairs the ability to evaluate VQA models. For example, if the accuracy for the questions beginning with 'Is' and 'Are' are improved by 15%, the whole accuracy would increase by 5% while the accuracy for 'why' and 'where' will only increase the total accuracy by 0.6%. 

4. Binary questions: There are many binary questions (answering with 'Yes/No') in the existing datasets. Binary questions are easy to evaluate and they can be comprehensive enough to contain variety of tasks (e.g. spatial reasoning, counting, drawing inferences). In practice, however, binary questions represent lack of complex questions and similarity between questions and answers built by human annotators.

5. Open-ended vs multiple choice: Multiple choice questions reduce VQA into determining which of the answer is correct rather than attempting to answer the question. Any VQA model theoretically should be able to provide answers without being given options of answers as inputs.


## Lessons learnt and future to-do-list
VQA models have shown its capacity to be better at human-computer interactions but there is still a large gap in performance between machine and human-generated answers.
The authors' discussions on downside (e.g. overly depending on language 'clues', dataset biases) and gaps (e.g. lack of comprehensive QA pairs) suggest that these aspects should be taken into consideration for improvements to newly-proposed models. In addition, given the paper documenting the development until 2015 or 2016, more recent developments in VQA will be investigated further.
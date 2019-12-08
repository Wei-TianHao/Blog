---
layout: post
comments: true
title: "元学习: 学习如何学习【译】"
date: 2019-9-17 00:00:00
tags: meta-learning long-read
---

> 学习如何学习的方法被称为元学习。元学习的目标是在接触到没见过的任务或者迁移到新环境中时，可以根据之前的经验和少量的样本快速学习如何应对。元学习有三种常见的实现方法：1）学习有效的距离度量方式（基于度量的方法）；2）使用带有显式或隐式记忆储存的（循环）神经网络（基于模型的方法）；3）训练以快速学习为目标的模型（基于优化的方法）

<!--more-->

**本文翻译自[Lilian](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)的英文博客，Lilian的博客质量非常高，向大家强烈安利~**

好的机器学习模型经常需要大量的数据来进行训练，但人却恰恰相反。小孩子看过一两次喵喵和小鸟后就能分辨出他们的区别。会骑自行车的人很快就能学会骑摩托车，有时候甚至不用人教。那么有没有可能让机器学习模型也具有相似的性质呢？如何才能让模型仅仅用少量的数据就学会新的概念和技能呢？这就是**元学习**要解决的问题。

我们期望好的元学习模型能够具备强大的适应能力和泛化能力。在测试时，模型会先经过一个自适应环节（adaptation process），即根据少量样本学习任务。经过自适应后，模型即可完成新的任务。自适应本质上来说就是一个短暂的学习过程，这就是为什么元学习也被称作[“学习”学习](https://www.cs.cmu.edu/~rsalakhu/papers/LakeEtAl2015Science.pdf). 

元学习可以解决的任务可以是任意一类定义好的机器学习任务，像是监督学习，强化学习等。具体的元学习任务例子有：
- 在没有猫的训练集上训练出来一个图片分类器，这个分类器需要在看过少数几张猫的照片后分辨出测试集的照片中有没有猫。
- 训练一个玩游戏的AI，这个AI需要快速学会如何玩一个从来没玩过的游戏。
- 一个仅在平地上训练过的机器人，需要在山坡上完成给定的任务。

{: class="table-of-content"}
* TOC
{:toc}


## 元学习问题定义

在本文中，我们主要关注监督学习中的元学习任务，比如图像分类。在之后的文章中我们会继续讲解更有意思的元强化学习。

### A Simple View

我们现在假设有一个任务的分布，我们从这个分布中采样了许多任务作为训练集。好的元学习模型在这个训练集上训练后，应当对这个空间里所有的任务都具有良好的表现，即使是从来没见过的任务。每个任务可以表示为一个数据集 $$ \mathcal{D} $$ ，数据集中包括特征向量 $$ x $$ 和标签 $$ y $$ ，分布表示为 $$ p(\mathcal{D}) $$ 。那么最佳的元学习模型参数可以表示为：

 $$ 
\theta^* = \arg\min_\theta \mathbb{E}_{\mathcal{D}\sim p(\mathcal{D})} [\mathcal{L}_\theta(\mathcal{D})]
 $$ 

上式的形式跟一般的学习任务非常像，只不过上式中的每个*数据集*是一个*数据样本*。

*少样本学习（Few-shot classification）* 是元学习的在监督学习中的一个实例。数据集 $$ \mathcal{D} $$ 经常被划分为两部分，一个用于学习的支持集（support set） $$ S $$ ，和一个用于训练和测试的预测集（prediction set） $$ B $$ ，即 $$ \mathcal{D}=\langle S, B\rangle $$ 。*K-shot N-class*分类任务，即支持集中有N类数据，每类数据有K个带有标注的样本。

![few-shot-classification]({{ '/assets/images/2019-09-19-meta-learning/few-shot-classification.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. 4-shot 2-class 图像分类的例子。 (图像来自[Pinterest](https://www.pinterest.com/))*


### 像测试一样训练


一个数据集 $$ \mathcal{D} $$ 包含许多对特征向量和标签，即 $$ \mathcal{D} = \{(\mathbf{x}_i, y_i)\} $$ 。每个标签属于一个标签类 $$ \mathcal{L} $$ 。假设我们的分类器 $$ f_\theta $$ 的输入是特征向量 $$ \mathbf{x} $$ ，输出是属于第 $$ y $$ 类的概率 $$ P_\theta(y\vert\mathbf{x}) $$ ， $$ \theta $$ 是分类器的参数。 

如果我们每次选一个 $$ B \subset \mathcal{D} $$ 作为训练的batch，则最佳的模型参数，应当能够最大化，多组batch的正确标签概率之和。

 $$ 
\begin{aligned}
\theta^* &= {\arg\max}_{\theta} \mathbb{E}_{(\mathbf{x}, y)\in \mathcal{D}}[P_\theta(y \vert \mathbf{x})] &\\
\theta^* &= {\arg\max}_{\theta} \mathbb{E}_{B\subset \mathcal{D}}[\sum_{(\mathbf{x}, y)\in B}P_\theta(y \vert \mathbf{x})] & \scriptstyle{\text{; trained with mini-batches.}}
\end{aligned}
 $$ 

few-shot classification的目标是，在小规模的support set上“快速学习”（类似fine-tuning）后，能够减少在prediction set上的预测误差。为了训练模型快速学习的能力，我们在训练的时候按照以下步骤：
1. 采样一个标签的子集, $$ L\subset\mathcal{L} $$ .
2. 根据采样的标签子集，采样一个support set $$ S^L \subset \mathcal{D} $$ 和一个training batch $$ B^L \subset \mathcal{D} $$ 。 $$ S^L $$ 和 $$ B^L $$ 中的数据的标签都属于 $$ L $$ ，即 $$ y \in L, \forall (x, y) \in S^L, B^L $$ .
3. 把support set作为模型的输入，进行“快速学习”。注意，不同的算法具有不同的学习策略，但总的来说，这一步不会永久性更新模型参数。 <!-- , $$ \hat{y}=f_\theta(\mathbf{x}, S^L) $$ -->
4. 把prediction set作为模型的输入，计算模型在 $$ B^L $$ 上的loss，根据这个loss进行反向传播更新模型参数。这一步与监督学习一致。

你可以把每一对 $$ (S^L, B^L) $$ 看做是一个数据点。模型被训练出了在其他数据集上扩展的能力。下式中的红色部分是元学习的目标相比于监督学习的目标多出来的部分。

 $$ 
\theta = \arg\max_\theta \color{red}{E_{L\subset\mathcal{L}}[} E_{\color{red}{S^L \subset\mathcal{D}, }B^L \subset\mathcal{D}} [\sum_{(x, y)\in B^L} P_\theta(x, y\color{red}{, S^L})] \color{red}{]}
 $$ 

这个想法有点像是我们面对某个只有少量数据的任务时，会使用在相关任务的大数据集上预训练的模型，然后进行fine-tuning。像是图形语义分割网络可以用在ImageNet上预训练的模型做初始化。相比于在一个特定任务上fine-tuning使得模型更好的拟合这个任务，元学习更进一步，它的目标是让模型优化以后能够在多个任务上表现的更好，类似于变得更容易被fine-tuning。

### 学习器和元学习器

还有一种常见的看待meta-learning的视角，把模型的更新划分为了两个阶段：
- 根据给定的任务，训练一个分类器 $$ f_\theta $$ 完成任务，作为“学习器”模型
- 同时，训练一个元学习器 $$ g_\phi $$ ，根据support set $$ S $$ 学习如何更新学习器模型的参数。 $$ \theta' = g_\phi(\theta, S) $$ 

则最后的优化目标中，我们需要更新 $$ \theta $$ 和 $$ \phi $$ 来最大化：

 $$ 
\mathbb{E}_{L\subset\mathcal{L}}[ \mathbb{E}_{S^L \subset\mathcal{D}, B^L \subset\mathcal{D}} [\sum_{(\mathbf{x}, y)\in B^L} P_{g_\phi(\theta, S^L)}(y \vert \mathbf{x})]]
 $$ 


### 常见方法

元学习主要有三类常见的方法：基于度量的方法（metric-based），基于模型的方法（model-based），基于优化的方法（optimization-based）。
Oriol Vinyals在NIPS 2018的meta-learning symposium上做了一个很好的[总结](http://metalearning-symposium.ml/files/vinyals.pdf)：

{: class="info"}
| ------------- | ------------- | ------------- | ------------- |
|  | Model-based | Metric-based | Optimization-based |
| ------------- | ------------- | ------------- | ------------- |
| **Key idea** | RNN; memory | Metric learning | Gradient descent |
| **How $$ P_\theta(y \vert \mathbf{x}) $$ is modeled?** | $$ f_\theta(\mathbf{x}, S) $$ | $$ \sum_{(\mathbf{x}_i, y_i) \in S} k_\theta(\mathbf{x}, \mathbf{x}_i)y_i $$ (*) | $$ P_{g_\phi(\theta, S^L)}(y \vert \mathbf{x}) $$ |

(*) $$ k_\theta $$ 是一个衡量 $$ \mathbf{x}_i $$ 和 $$ \mathbf{x} $$ 相似度的kernel function。

接下来我们会回顾各种方法的经典模型。


## 基于度量的方法

基于度量的元学习的核心思想类似于最近邻算法([k-NN分类](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)、[k-means聚类](https://en.wikipedia.org/wiki/K-means_clustering))和[核密度估计](https://en.wikipedia.org/wiki/Kernel_density_estimation)。该类方法在已知标签的集合上预测出来的概率，是support set中的样本标签的加权和。 权重由核函数（kernal function） $$ k_\theta $$ 算得，该权重代表着两个数据样本之间的相似性。

 $$ 
P_\theta(y \vert \mathbf{x}, S) = \sum_{(\mathbf{x}_i, y_i) \in S} k_\theta(\mathbf{x}, \mathbf{x}_i)y_i 
 $$ 

因此，学到一个好的核函数对于基于度量的元学习模型至关重要。[Metric learning](https://en.wikipedia.org/wiki/Similarity_learning#Metric_learning)正是针对该问题提出的方法，它的目标就是学到一个不同样本之间的metric或者说是距离函数。任务不同，好的metric的定义也不同。但它一定在任务空间上表示了输入之间的联系，并且能够帮助我们解决问题。

下面列出的所有方法都显式的学习了输入数据的嵌入向量（embedding vectors），并根据其设计合适的kernel function。

### Convolutional Siamese Neural Network

[Siamese Neural Network](https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf)最早被提出用来解决笔迹验证问题，siamese network由两个孪生网络组成，这两个网络的输出被联合起来训练一个函数，用于学习一对数据输入之间的关系。这两个网络结构相同，共享参数，实际上就是一个网络在学习如何有效地embedding才能显现出一对数据之间的关系。顺便一提，这是LeCun 1994年的论文。

[Koch, Zemel & Salakhutdinov (2015)](http://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf)提出了一种用siamese网络做one-shot image classification的方法。首先，训练一个用于图片验证的siamese网络，分辨两张图片是否属于同一类。然后在测试时，siamese网络把测试输入和support set里面的所有图片进行比较，选择相似度最高的那张图片所属的类作为输出。


![siamese]({{ '/assets/images/2019-09-19-meta-learning/siamese-conv-net.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. 卷积siamese网络用于few-shot image classification的例子。*

1. 首先，卷积siamese网络学习一个由多个卷积层组成的embedding函数 $$ f_\theta $$ ，把两张图片编码为特征向量。
2. 两个特征向量之间的L1距离可以表示为 $$ \vert f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j) \vert $$ 。
3. 通过一个linear feedforward layer和sigmoid把距离转换为概率。这就是两张图片属于同一类的概率。
4. loss函数就是cross entropy loss，因为label是二元的。

<!-- In this way, an efficient image embedding is trained so that the distance between two embeddings is proportional to the similarity between two images. -->

 $$ 
\begin{aligned}
p(\mathbf{x}_i, \mathbf{x}_j) &= \sigma(\mathbf{W}\vert f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j) \vert) \\
\mathcal{L}(B) &= \sum_{(\mathbf{x}_i, \mathbf{x}_j, y_i, y_j)\in B} \mathbf{1}_{y_i=y_j}\log p(\mathbf{x}_i, \mathbf{x}_j) + (1-\mathbf{1}_{y_i=y_j})\log (1-p(\mathbf{x}_i, \mathbf{x}_j))
\end{aligned}
 $$ 

Training batch $$ B $$ 可以通过对图片做一些变形增加数据量。你也可以把L1距离替换成其他距离，比如L2距离、cosine距离等等。只要距离是可导的就可以。

给定一个支持集 $$ S $$ 和一个测试图片 $$ \mathbf{x} $$ ，最终预测的分类为：

 $$ 
\hat{c}_S(\mathbf{x}) = c(\arg\max_{\mathbf{x}_i \in S} P(\mathbf{x}, \mathbf{x}_i))
 $$ 

 $$ c(\mathbf{x}) $$ 是图片 $$ \mathbf{x} $$ 的label， $$ \hat{c}(.) $$ 是预测的label。

这里我们有一个假设：学到的embedding在未见过的分类上依然能很好的衡量图片间的距离。这个假设跟迁移学习中使用预训练模型所隐含的假设是一样的。比如，在ImageNet上预训练的模型，其学到的卷积特征表达方式对于其他图像任务也有帮助。但实际上当新任务与旧任务有所差别的时候，预训练模型的效果就没有那么好了。

### Matching Networks


**Matching Networks** ([Vinyals et al., 2016](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf))的目标是：对于每一个给定的支持集 $$ S=\{x_i, y_i\}_{i=1}^k $$ (*k-shot* classification)，分别学一个分类器 $$ c_S $$ 。 这个分类器给出了给定测试样本 $$ \mathbf{x} $$ 时，输出 $$ y $$ 的概率分布。这个分类器的输出被定义为支持集中一系列label的加权和，权重由一个注意力核（attention kernel） $$ a(\mathbf{x}, \mathbf{x}_i) $$ 决定。权重应当与 $$ \mathbf{x} $$ 和 $$ \mathbf{x}_i $$ 间的相似度成正比。

![siamese]({{ '/assets/images/2019-09-19-meta-learning/matching-networks.png' | relative_url }})
{: style="width: 70%;" class="center"}
*Fig. 3. Matching Networks结构。（图像来源: [原论文](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)）*


 $$ 
c_S(\mathbf{x}) = P(y \vert \mathbf{x}, S) = \sum_{i=1}^k a(\mathbf{x}, \mathbf{x}_i) y_i
\text{, where }S=\{(\mathbf{x}_i, y_i)\}_{i=1}^k
 $$ 

Attention kernel由两个embedding function $$ f $$ 和 $$ g $$ 决定。分别用于encoding测试样例和支持集样本。两个样本之间的注意力权重是经过softmax归一化后的，他们embedding vectors的cosine距离 $$ \text{cosine}(.) $$ 。

 $$ 
a(\mathbf{x}, \mathbf{x}_i) = \frac{\exp(\text{cosine}(f(\mathbf{x}), g(\mathbf{x}_i))}{\sum_{j=1}^k\exp(\text{cosine}(f(\mathbf{x}), g(\mathbf{x}_j))}
 $$ 


#### Simple Embedding

在简化版本里，embedding function是一个使用单样本作为输入的神经网络。而且我们可以假设 $$ f=g $$ 。

#### Full Context Embeddings

Embeding vectors对于构建一个好的分类器至关重要。只把一个数据样本作为embedding function的输入，会导致很难高效的估计出整个特征空间。因此，Matching Network模型又进一步发展，通过把整个支持集 $$ S $$ 作为embedding function的额外输入来加强embedding的有效性，相当于给样本添加了语境，让embedding根据样本与支持集中样本的关系进行调整。


- $$ g_\theta(\mathbf{x}_i, S) $$ 在整个支持集 $$ S $$ 的语境下用一个双向LSTM来编码 $$ \mathbf{x}_i $$ .

- $$ f_\theta(\mathbf{x}, S) $$ 在支持集 $$ S $$ 上使用read attention机制编码测试样本 $$ \mathbf{x} $$ 。
    1. 首先测试样本经过一个简单的神经网络，比如CNN，以抽取基本特征 $$ f'(\mathbf{x}) $$ 。
    2. 然后，一个带有read attention vector的LSTM被训练用于生成部分hidden state：<br/>
    $$ 
    \begin{aligned}
    \hat{\mathbf{h}}_t, \mathbf{c}_t &= \text{LSTM}(f'(\mathbf{x}), [\mathbf{h}_{t-1}, \mathbf{r}_{t-1}], \mathbf{c}_{t-1}) \\
    \mathbf{h}_t &= \hat{\mathbf{h}}_t + f'(\mathbf{x}) \\
    \mathbf{r}_{t-1} &= \sum_{i=1}^k a(\mathbf{h}_{t-1}, g(\mathbf{x}_i)) g(\mathbf{x}_i) \\
    a(\mathbf{h}_{t-1}, g(\mathbf{x}_i)) &= \text{softmax}(\mathbf{h}_{t-1}^\top g(\mathbf{x}_i)) = \frac{\exp(\mathbf{h}_{t-1}^\top g(\mathbf{x}_i))}{\sum_{j=1}^k \exp(\mathbf{h}_{t-1}^\top g(\mathbf{x}_j))}
    \end{aligned}
    $$ 
    1. 最终，如果我们做k步的读取 $$ f(\mathbf{x}, S)=\mathbf{h}_K $$ 。

这类embedding方法被称作“全语境嵌入”（Full Contextual Embeddings）。有意思的是，这类方法对于困难的任务（few-shot classification on mini ImageNet）有所帮助，但对于简单的任务却没有提升（Omniglot）。

Matching Networks的训练过程与测试时的推理过程是一致的，详情请回顾之前的[章节](#像测试一样训练)。值得一提的是，Matching Networks的论文强调了训练和测试的条件应当一致的原则。

 $$ 
\theta^* = \arg\max_\theta \mathbb{E}_{L\subset\mathcal{L}}[ \mathbb{E}_{S^L \subset\mathcal{D}, B^L \subset\mathcal{D}} [\sum_{(\mathbf{x}, y)\in B^L} P_\theta(y\vert\mathbf{x}, S^L)]]
 $$ 



### Relation Network

**Relation Network (RN)** ([Sung et al., 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Sung_Learning_to_Compare_CVPR_2018_paper.pdf))与[siamese network](#convolutional-siamese-neural-network)比较像，但有以下几个不同点：
1. 两个样本间的相似系数不是由特征空间的L1距离决定的，而是由一个CNN分类器 $$ g_\phi $$ 预测的。两个样本 $$ \mathbf{x}_i $$ 和 $$ \mathbf{x}_j $$ 间的相似系数为 $$ r_{ij} = g_\phi([\mathbf{x}_i, \mathbf{x}_j]) $$ ，其中 $$ [.,.] $$ 代表着concatenation。
2. 目标优化函数是MSE损失，而不是cross-entropy，因为RN在预测时更倾向于把相似系数预测过程作为一个regression问题，而不是二分类问题， $$ \mathcal{L}(B) = \sum_{(\mathbf{x}_i, \mathbf{x}_j, y_i, y_j)\in B} (r_{ij} - \mathbf{1}_{y_i=y_j})^2 $$ 


![relation-network]({{ '/assets/images/2019-09-19-meta-learning/relation-network.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4. Relation Network的结构，图中是一个5分类1-shot的例子。(图片来源：[原论文](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Sung_Learning_to_Compare_CVPR_2018_paper.pdf))*

(注意：还有一个[Relation Network](https://deepmind.com/blog/neural-approach-relational-reasoning/)是DeepMind提出来用于关系推理的，不要搞混了。)


### Prototypical Networks

**Prototypical Networks** ([Snell, Swersky & Zemel, 2017](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf))用一个嵌入函数 $$ f_\theta $$ 把每个输入编码为一个M维特征向量。然后对每一类 $$ c \in \mathcal{C} $$ ，取所有支持集样本的特征向量的平均值作为这个类的prototype特征。

 $$ 
\mathbf{v}_c = \frac{1}{|S_c|} \sum_{(\mathbf{x}_i, y_i) \in S_c} f_\theta(\mathbf{x}_i)
 $$ 


![prototypical-networks]({{ '/assets/images/2019-09-19-meta-learning/prototypical-networks.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 5. 在少样本学习和无样本学习中的Prototypical networks。(图像来源：[原论文](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf))*

测试样本属于各类的概率分布由特征向量和prototype向量的距离取负后通过softmanx得到。

 $$ 
P(y=c\vert\mathbf{x})=\text{softmax}(-d_\varphi(f_\theta(\mathbf{x}), \mathbf{v}_c)) = \frac{\exp(-d_\varphi(f_\theta(\mathbf{x}), \mathbf{v}_c))}{\sum_{c' \in \mathcal{C}}\exp(-d_\varphi(f_\theta(\mathbf{x}), \mathbf{v}_{c'}))}
 $$ 

其中 $$ d_\varphi $$ 可以是任意距离函数，只要 $$ \varphi $$ 可导即可。这篇文章中，他们使用了平方欧氏距离。

损失函数用的是负对数似然: $$ \mathcal{L}(\theta) = -\log P_\theta(y=c\vert\mathbf{x}) $$ .



## 基于模型的方法

基于模型的元学习方法不对 $$ P_\theta(y\vert\mathbf{x}) $$ 作出任何假设。 $$ P_\theta(y\vert\mathbf{x}) $$ 是由一个专门用来快速学习的模型生成的，快速学习指的是这个模型可以根据少量的训练快速更新参数。有两种方式可以实现快速学习，1.设计好模型的内部架构使其能够快速学习，2.用另外一个模型来生成快速学习模型的参数。


### Memory-Augmented Neural Networks

许多模型架构使用了外部储存来帮助神经网络学习，像是[Neural Turing Machines](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#neural-turing-machines)和[Memory Networks](https://arxiv.org/abs/1410.3916)。使用外部存储，让神经网络能够更容易的学到新知识并提供给以后使用。这样的模型被称为**MANN**（"**Memory-Augmented Neural Network**"）。注意，只使用了*内部存储*的循环神经网络并不是MANN，比如RNN、LSTM。

MANN的目的是在仅给定几个训练样本的情况下，快速编码新的信息并适应新的任务，因此MANN非常适合用于元学习。[Santoro et al. (2016)](http://proceedings.mlr.press/v48/santoro16.pdf)以Neural Turing Machine（NTM）为基础，对训练和存储读写机制（也称为“寻址机制”，即如何对存储向量分配注意力权重）做了一系列的修改。如果你对NTM或寻址机制不太熟悉的话，可以参见原作者之前博文中的[NTM介绍](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#neural-turing-machines)。

简要回顾一下，NTM由一个控制器神经网络和存储器组成。控制器学习如何通过软注意力（soft attention）读写存储器，而存储器相当于是一个知识库。注意力权重是由寻址机制生成的，由询问的内容和位置共同决定。

![NTM]({{ '/assets/images/2019-09-19-meta-learning/NTM.png' | relative_url }})
{: style="width: 70%;" class="center"}
*Fig. 6. NTM的架构，t时刻的存储， $$ \mathbf{M}_t $$ 是一个大小为 $$ N \times M $$ 的矩阵，代表着N个M维的向量，每个向量是一条记录*

#### MANN for Meta-Learning

为了在元学习中使用MANN，我们需要训练这个网络使得它能够快速提炼新任务的信息，而且能够快速稳定的查询之前存储的特征。

[Santoro et al., 2016](http://proceedings.mlr.press/v48/santoro16.pdf)提出了一种有意思的训练方式，他们强迫存储器保留当前样本的信息直到对应的标签出现。在每个episode中，标签有 $$ 一步的延迟 $$ ，即每次给出的训练对为 $$ (\mathbf{x}_{t+1}, y_t) $$ 。

![NTM]({{ '/assets/images/2019-09-19-meta-learning/mann-meta-learning.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 7. MANN在元学习中的任务设置（图像来源：[原论文](http://proceedings.mlr.press/v48/santoro16.pdf)）。*

通过这种设定，MANN会学到要记住新数据集的信息，因为存储器需要保留着当前输入的信息，并且在对应的标签出现的时候取回之前存储的信息进行预测。

接下来，我们看看存储器是完成信息的存储和取回的。

#### Addressing Mechanism for Meta-Learning

为了让模型更加适应元学习，除了改变训练过程，作者还增加了一个完全基于内容的寻址机制。

**>> 如何从存储器中取回信息?**
<br/>

读取注意力（read attention）完全由内容相似度决定。

首先，根据t时刻的输入 $$ \mathbf{x} $$ ，控制器生成一个键值特征向量 $$ \mathbf{k}_t $$ 。然后用类似于NTM的方法，计算键值特征向量和存储器中每个向量的cosine距离，经过softmax归一化，得到一个读取权重向量 $$ \mathbf{w}_t^r $$ 。读取向量 $$ \mathbf{r}_t $$ 是对存储器中所有向量的加权和：

 $$ 
\mathbf{r}_i = \sum_{i=1}^N w_t^r(i)\mathbf{M}_t(i)
\text{, where } w_t^r(i) = \text{softmax}(\frac{\mathbf{k}_t \cdot \mathbf{M}_t(i)}{\|\mathbf{k}_t\| \cdot \|\mathbf{M}_t(i)\|})
 $$ 

 $$ M_t $$ 是t时刻的存储器矩阵， $$ M_t(i) $$ 是该矩阵中的第i行，即第i个向量。

**>> 如何往存储器中写入信息?**
<br/>

写入新信息的寻址机制跟[缓存置换机制](https://en.wikipedia.org/wiki/Cache_replacement_policies)很像。为了更好的适应元学习的任务，MANN使用的是**最近最少使用算法(Least Recently Used Access, LRUA)**。LRUA算法会优先覆盖*最少*使用的，或者*最近*刚用过的存储位置。

* 最少使用的位置：目的是保存经常使用的那些信息(参见[LFU](https://en.wikipedia.org/wiki/Least_frequently_used));
* 最近使用的位置：原因是刚用过的信息很有可能不会马上用到(参见[MRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Most_recently_used_(MRU))). 

除了LRUA，还有许多其他的缓存置换算法。根据场景的不同，其他的算法可能有更好的表现。另外相比于人为指定一种缓存替换机制，根据存储使用的规律，学出一套寻址策略可能效果更好。

这里的LRUA以一种所有运算都可微分的方式实现：
1. t时刻的使用权重 $$ \mathbf{w}^u_t $$ 是当前读写向量的和，再加上上一时刻的使用权重 $$ \gamma \mathbf{w}^u_{t-1} $$ ， $$ \gamma $$ 是一个衰减系数。
2. 写入向量由之前的读取权重（代表着最近使用的位置）和之前的最少使用权重（代表着最少使用的位置）插值得到。插值参数是超参数 $$ \alpha $$ 的sigmoid。
3. 最少使用权重 $$ \mathbf{w}_t^{lu} $$ 由使用权重 $$ \mathbf{w}_t^u $$ 二值化得到。首先选取 $$ \mathbf{w}_t^u $$ 第n小的元素，找出 $$ \mathbf{w}_t^u $$ 中所有比这个元素小的元素，对应的位置在 $$ \mathbf{w}_t^{lu} $$ 中设为1，反之则设为0。

 $$ 
\begin{aligned}
\mathbf{w}_t^u &= \gamma \mathbf{w}_{t-1}^u + \mathbf{w}_t^r + \mathbf{w}_t^w \\
\mathbf{w}_t^r &= \text{softmax}(\text{cosine}(\mathbf{k}_t, \mathbf{M}_t(i))) \\
\mathbf{w}_t^w &= \sigma(\alpha)\mathbf{w}_{t-1}^r + (1-\sigma(\alpha))\mathbf{w}^{lu}_{t-1}\\
\mathbf{w}_t^{lu} &= \mathbf{1}_{w_t^u(i) \leq m(\mathbf{w}_t^u, n)}
\text{, where }m(\mathbf{w}_t^u, n)\text{ is the }n\text{-th smallest element in vector }\mathbf{w}_t^u\text{.}
\end{aligned}
 $$ 

最后，在把 $$ \mathbf{w}_t^{lu} $$ 代表着的最近使用存储设为0以后，更新每个存储向量：
 $$ 
\mathbf{M}_t(i) = \mathbf{M}_{t-1}(i) + w_t^w(i)\mathbf{k}_t, \forall i
 $$ 


### Meta Networks

**Meta Networks** ([Munkhdalai & Yu, 2017](https://arxiv.org/abs/1703.00837))，简称**MetaNet**，是一个专门针对多任务间*快速*泛化设计的元学习模型，模型结构和训练过程都经过了调整。

#### Fast Weights

MetaNet的快速泛化能力依赖于“快参数（fast weights）”。有许多论文涉及这个话题，但我还没把他们全都仔细读过，没能找到一个具体的定义，只有一些模糊的共识。一般神经网络的权重是根据目标函数进行随机梯度下降更新的，但这个过程很慢。一种更快的学习方法是利用另外一个神经网络，预测当前神经网络的参数，预测出来的参数被称为*快参数*。而普通SGD生成的权重则被称为*慢参数*。

在MetaNet中，损失梯度作为*元信息*，被用于生产学习快参数的模型。慢参数和快参数在神经网络中被结合起来用于预测。快参数是针对任务进行优化产生的参数，使用快参数相当于针对当前的任务进行了优化。

![slow-fast-weights]({{ '/assets/images/2019-09-19-meta-learning/combine-slow-fast-weights.png' | relative_url }})
{: style="width: 50%;" class="center"}
*Fig. 8. 结合了慢参数和快参数的MLP。 $$ \bigoplus $$ is element-wise sum. (图像来源：[原论文](https://arxiv.org/abs/1703.00837)).*


#### Model Components


> 免责声明：下面的部分与原始论文有所不同。在我看来，这篇文章的想法很有趣，但写的不太好，所以我想用自己的语言来介绍这篇文章。

MetaNet的关键组件是：
- $$ f_\theta $$ ：一个由 $$ \theta $$ 决定的编码函数，发挥着元学习器的作用。负责把原始输入编码为特征向量。类似[Siamese Neural Network](#convolutional-siamese-neural-network)，我们希望训练这个编码函数使得能够根据其生成的特征向量判断两个输入是否属于同一类（验证任务）。
- $$ g_\phi $$ ：一个由 $$ \phi $$ 决定的基学习器，负责完成真正的学习任务。

如果我们就此打住的话，这基本就是[Relation Network](#relation-network)。
但MetaNet，给这两个模型额外增加了快参数（参加Fig. 8）。 

因此，我们需要额外两个模型，分别用于生成 $$ f $$ 和 $$ g $$ 的快参数。

- $$ F_w $$ ：一个由 $$ w $$ 决定的LSTM，用于学习嵌入函数 $$ f $$ 的快参数 $$ \theta^+ $$ 。它把 $$ f $$ 在验证任务上的loss梯度作为输入。
- $$ G_v $$ ：一个由 $$ v $$ 决定的神经网络，根据基学习器 $$ g $$ 的loss梯度学习其快参数 $$ \phi^+ $$ 。在MetaNet中，学习器的loss梯度被视作任务的*元信息*。

现在我们来看看MetaNet是怎么训练的。训练数据包含许多对数据集。一个支持集 $$ S=\{\mathbf{x}'_i, y'_i\}_{i=1}^K $$ 和一个测试集 $$ U=\{\mathbf{x}_i, y_i\}_{i=1}^L $$ 。我们有四个模型和四个模型参数集 $$ (\theta, \phi, w, v) $$ 要学。

![meta-net]({{ '/assets/images/2019-09-19-meta-learning/meta-network.png' | relative_url }})
{: style="width: 90%;" class="center"}
*Fig.9. MetaNet的结构。*


#### 训练过程

1. 从支持集 $$ S $$ 中随机采样一对输入， $$ (\mathbf{x}'_i, y'_i) $$ 和 $$ (\mathbf{x}'_j, y_j) $$ 。令 $$ \mathbf{x}_{(t,1)}=\mathbf{x}'_i $$ ， $$ \mathbf{x}_{(t,2)}=\mathbf{x}'_j $$ .<br/>
for $$ t = 1, \dots, K $$ :
    * a\. 计算编码函数的loss，即验证任务上的交叉熵<br/>
      $$ \mathcal{L}^\text{emb}_t = \mathbf{1}_{y'_i=y'_j} \log P_t + (1 - \mathbf{1}_{y'_i=y'_j})\log(1 - P_t)\text{, where }P_t = \sigma(\mathbf{W}\vert f_\theta(\mathbf{x}_{(t,1)}) - f_\theta(\mathbf{x}_{(t,2)})\vert) $$ 
2. 根据编码函数的loss计算任务级别的快参数:
 $$ \theta^+ = F_w(\nabla_\theta \mathcal{L}^\text{emb}_1, \dots, \mathcal{L}^\text{emb}_T) $$ 
3. 接下来遍历支持集 $$ S $$ 中的样本，计算样本级别的快参数<br/>
for $$ i=1, \dots, K $$ :
    * a\. 基学习器输出一个概率分布： $$ P(\hat{y}_i \vert \mathbf{x}_i) = g_\phi(\mathbf{x}_i) $$ ，loss可以用交叉熵或者MSE: $$ \mathcal{L}^\text{task}_i = y'_i \log g_\phi(\mathbf{x}'_i) + (1- y'_i) \log (1 - g_\phi(\mathbf{x}'_i)) $$ 
    * b\. 提取任务的元信息（loss的梯度）并计算样本级别的快参数:
      $$ \phi_i^+ = G_v(\nabla_\phi\mathcal{L}^\text{task}_i) $$ 
        * 把 $$ \phi^+_i $$ 放在“值”存储器 $$ \mathbf{M} $$ 的第 $$ i $$ 个位置。<br/>
    * d\. 编码器结合慢参数和快参数对支持集中的样本进行编码： $$ r'_i = f_{\theta, \theta^+}(\mathbf{x}'_i) $$ 
        * 把 $$ r'_i $$ 放在“键”存储器 $$ \mathbf{R} $$ 的第 $$ i $$ 个位置。 
4. 最后就是要来根据测试集 $$ U=\{\mathbf{x}_i, y_i\}_{i=1}^L $$ 来构建训练的loss了。注意，在这一步中，我们不使用 $$ F_w $$ 和 $$G_v$$ 计算快参数。编码器的快参数保持不变，而基学习器的快参数则根据查询存储器得到。<br/>
从 $$ \mathcal{L}_\text{train}=0 $$ 开始：<br/>
for $$ j=1, \dots, L $$ :
    * a\. 使用编码器结合慢参数和之前得到的快参数对测试样例进行编码：
      $$ r_j = f_{\theta, \theta^+}(\mathbf{x}_j) $$ 
    * b\. 运用注意力机制，查找编码得到的 $$ r_j $$ 在“键”存储器 $$ \mathbf{R} $$ 的位置，然后拿出对应位置“值”存储器 $$ \mathbf{M} $$ 对应位置的快参数，组合成当前测试样本对应的基学习器快参数。注意力函数可以随便选，MetaNet用的是cosine距离<br/>
  $$ 
    \begin{aligned}
    a_j &= \text{cosine}(\mathbf{R}, r_j) = [\frac{r'_1\cdot r_j}{\|r'_1\|\cdot\|r_j\|}, \dots, \frac{r'_N\cdot r_j}{\|r'_N\|\cdot\|r_j\|}]\\
    \phi^+_j &= \text{softmax}(a_j)^\top \mathbf{M}
    \end{aligned}
  $$ 
    * c\. 更新训练loss： $$ \mathcal{L}_\text{train} \leftarrow \mathcal{L}_\text{train} + \mathcal{L}^\text{task}(g_{\phi, \phi^+}(\mathbf{x}_i), y_i) $$ 
5. 根据 $$ \mathcal{L}_\text{train} $$ 更新所有参数 $$ (\theta, \phi, w, v) $$ .


## 基于优化的方法

深度学习模型通过反向传播梯度进行学习。然后基于梯度的优化方法并不适用于仅有少量训练样本的情况，也很难在短短几步之内达到收敛。那怎样才能调整现有的优化算法使得模型能够在仅有少量样本的情况下学好呢？这就是基于优化的元学习算法的目标。

### LSTM Meta-Learner

[Ravi & Larochelle (2017)](https://openreview.net/pdf?id=rJY0-Kcll) 把优化算法显式的建模出来，并命名为“元学习器”，原本处理任务的模型被称为“学习器”。元学习器的目标是使用少量支持集在仅仅几步之内快速更新学习器的参数，使得学习器能够快速适应新任务。

我们用 $$ M_\theta $$ 代表参数为 $$ \theta $$ 的学习器，用 $$ R_\Theta $$ 代表参数为 $$ \Theta $$ 的元学习器，loss函数为 $$ \mathcal{L} $$ .

#### 为什么使用 LSTM？

之所以使用 LSTM 作为元学习器的模型，有这样几点原因：
1. 反向传播中基于梯度的更新跟 LSTM 中 cell 状态的更新有相似之处。
2. 知道之前的梯度对当前的梯度更新有好处。可以参考 [momentum](http://ruder.io/optimizing-gradient-descent/index.html#momentum) 的原理。 

第t步时，设定学习率为$$ \alpha_t $$，更新学习器的参数：
 
 $$ 
\theta_t = \theta_{t-1} - \alpha_t \nabla_{\theta_{t-1}}\mathcal{L}_t
 $$ 

这个过程跟 LSTM 的 cell 状态更新具有相同的形式。如果我们令遗忘门 $$ f_t=1 $$，输入门 $$ i_t = \alpha_t $$， cell 状态 $$ c_t = \theta_t $$ , 新 cell 状态 $$ \tilde{c}_t = -\nabla_{\theta_{t-1}}\mathcal{L}_t $$，则：

 $$ 
\begin{aligned}
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\\
    &= \theta_{t-1} - \alpha_t\nabla_{\theta_{t-1}}\mathcal{L}_t
\end{aligned}
 $$ 


但是固定 $$ f_t=1 $$ 和 $$ i_t=\alpha_t $$ 可能并不是最好的，我们可以让他们随数据集变化而变化，由学习得到。

 $$ 
\begin{aligned}
f_t &= \sigma(\mathbf{W}_f \cdot [\nabla_{\theta_{t-1}}\mathcal{L}_t, \mathcal{L}_t, \theta_{t-1}, f_{t-1}] + \mathbf{b}_f) & \scriptstyle{\text{; how much to forget the old value of parameters.}}\\
i_t &= \sigma(\mathbf{W}_i \cdot [\nabla_{\theta_{t-1}}\mathcal{L}_t, \mathcal{L}_t, \theta_{t-1}, i_{t-1}] + \mathbf{b}_i) & \scriptstyle{\text{; corresponding to the learning rate at time step t.}}\\
\tilde{\theta}_t &= -\nabla_{\theta_{t-1}}\mathcal{L}_t &\\
\theta_t &= f_t \odot \theta_{t-1} + i_t \odot \tilde{\theta}_t &\\
\end{aligned}
 $$ 


#### Model Setup

![lstm-meta-learner]({{ '/assets/images/2019-09-19-meta-learning/lstm-meta-learner.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig.10. 如何训练学习器 $$ M_\theta $$ 和元学习器 $$ R_\Theta $$. (图像来源：[原论文](https://openreview.net/pdf?id=rJY0-Kcll)中的图片加了一些注释)*

在[Matching Networks](#matching-networks)中我们已经证明了用模仿测试过程的方式训练能够取得很好的效果，这里也用了类似的方法。在每个训练阶段，我们先采样一个数据集 $$ \mathcal{D} = (\mathcal{D}_\text{train}, \mathcal{D}_\text{test}) \in \hat{\mathcal{D}}_\text{meta-train} $$，再从 $$ \mathcal{D}_\text{train} $$ 中采样 $$T$$ 轮 mini-batches 用于更新 $$\theta$$。学习器参数的最终状态$$ \theta_T $$被用来在测试数据 $$ \mathcal{D}_\text{test} $$ 上训练元学习器。

有两个实现的细节需要注意一下：
1. 如何压缩 LSTM 元学习的参数空间？元学习器是在建模一个神经网络的参数，所以有上百万个变量要学。为了减小元学习器的参数空间，这篇文章借鉴了[共享参数](https://arxiv.org/abs/1606.04474)的方法。元学习器本质上学习的是一种更新原则，即如何根据一个参数的值和其梯度生成这个参数的新值（比如一阶方法，牛顿法等），与参数在学习器中的位置无关。所以我们可以假设所有参数的更新原则都是一样的，即元学习只需要输出一维变量即可。
2. 为了简化训练过程，元学习器假设 loss $$ \mathcal{L}_t $$ 和梯度 $$ \nabla_{\theta_{t-1}} \mathcal{L}_t $$ 是独立的。


![train-meta-learner]({{ '/assets/images/2019-09-19-meta-learning/train-meta-learner.png' | relative_url }})
{: style="width: 100%;" class="center"}


### MAML

**Model-Agnostic Meta-Learning** 简称 **MAML** ([Finn, et al. 2017](https://arxiv.org/abs/1703.03400))， 是一种非常通用的优化算法，可以被用于任何基于梯度下降学习的模型。

假设我们的模型是 $$ f_\theta $$，参数为 $$ \theta $$。给定一个任务 $$ \tau_i $$ 

假设我们的模型是 $$ f_\theta $$，参数为 $$ \theta $$。给定一个任务 $$ \tau_i $$ 和其相应的数据集 $$ (\mathcal{D}^{(i)}_\text{train}, \mathcal{D}^{(i)}_\text{test}) $$，我们可以对模型参数进行一次或多次梯度下降。（下式中只进行了一次迭代）：

 $$ 
\theta'_i = \theta - \alpha \nabla_\theta\mathcal{L}^{(0)}_{\tau_i}(f_\theta)
 $$ 

其中 $$ \mathcal{L}^{(0)} $$ 是由编号为0的小数据batch算得的loss。


![MAML]({{ '/assets/images/2019-09-19-meta-learning/maml.png' | relative_url }})
{: style="width: 45%;" class="center"}
*Fig. 11. MAML图示。 (图像来源：[原论文](https://arxiv.org/abs/1703.03400))*

当然，上面这个式子只针对一个特定的任务进行了优化。而MAML为了能够更好地扩展到一系列任务上，我们想要寻找一个在给定任意任务后**微调过程最高效**的 $$ \theta^* $$。现在，假设我们采样了一个编号为1的数据batch用于更新元目标。对应的loss记为 $$ \mathcal{L}^{(1)} $$。$$ \mathcal{L}^{(0)} $$ 和 $$ \mathcal{L}^{(1)} $$的上标只代表着数据batch不同，他们都是同一个目标方程计算得到的。那么

 $$ 
\begin{aligned}
\theta^* 
&= \arg\min_\theta \sum_{\tau_i \sim p(\tau)} \mathcal{L}_{\tau_i}^{(1)} (f_{\theta'_i}) = \arg\min_\theta \sum_{\tau_i \sim p(\tau)} \mathcal{L}_{\tau_i}^{(1)} (f_{\theta - \alpha\nabla_\theta \mathcal{L}_{\tau_i}^{(0)}(f_\theta)}) & \\
\theta &\leftarrow \theta - \beta \nabla_{\theta} \sum_{\tau_i \sim p(\tau)} \mathcal{L}_{\tau_i}^{(1)} (f_{\theta - \alpha\nabla_\theta \mathcal{L}_{\tau_i}^{(0)}(f_\theta)}) & \scriptstyle{\text{; updating rule}}
\end{aligned}
 $$ 


![MAML Algorithm]({{ '/assets/images/2019-09-19-meta-learning/maml-algo.png' | relative_url }})
{: style="width: 60%;" class="center"}
*Fig. 12. MAML算法的一般形式。 (图像来源：[原论文](https://arxiv.org/abs/1703.03400))*


#### First-Order MAML

上面的元优化过程依赖于二阶导数（多次迭代）。而为了加快计算，简化实现过程，一个忽略了二阶项的简化版MAML被提出了，称为 **First-Order MAML (FOMAML)**。

让我们来考虑一下执行 $$ k $$ 次内循环（微调过程）梯度下降的过程（$$ k\geq1 $$）。假设一开始的模型参数为 $$ \theta_\text{meta} $$ ：

 $$ 
\begin{aligned}
\theta_0 &= \theta_\text{meta}\\
\theta_1 &= \theta_0 - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_0)\\
\theta_2 &= \theta_1 - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_1)\\
&\dots\\
\theta_k &= \theta_{k-1} - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_{k-1})
\end{aligned}
 $$ 

 而在外循环中，我们采样一个新的数据batch用于更新元目标。

 $$ 
\begin{aligned}
\theta_\text{meta} &\leftarrow \theta_\text{meta} - \beta g_\text{MAML} & \scriptstyle{\text{; update for meta-objective}} \\[2mm]
\text{其中 } g_\text{MAML}
&= \nabla_{\theta} \mathcal{L}^{(1)}(\theta_k) &\\[2mm]
&= \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot (\nabla_{\theta_{k-1}} \theta_k) \dots (\nabla_{\theta_0} \theta_1) \cdot (\nabla_{\theta} \theta_0) & \scriptstyle{\text{; following the chain rule}} \\
&= \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot \prod_{i=1}^k \nabla_{\theta_{i-1}} \theta_i &  \\
&= \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot \prod_{i=1}^k \nabla_{\theta_{i-1}} (\theta_{i-1} - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_{i-1})) &  \\
&= \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot \prod_{i=1}^k (I - \alpha\nabla_{\theta_{i-1}}(\nabla_\theta\mathcal{L}^{(0)}(\theta_{i-1}))) &
\end{aligned}
 $$ 

MAML的梯度是：

 $$ 
g_\text{MAML} = \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot \prod_{i=1}^k (I - \alpha \color{red}{\nabla_{\theta_{i-1}}(\nabla_\theta\mathcal{L}^{(0)}(\theta_{i-1}))})
 $$ 

一阶 MAML 忽略了用红色标记的二阶导数部分。它被简化为了下式，等价于最后一次内循环梯度更新的结果。

 $$ 
g_\text{FOMAML} = \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k)
 $$ 


### Reptile

**Reptile** ([Nichol, Achiam & Schulman, 2018](https://arxiv.org/abs/1803.02999)) 是一个超级简单的元学习优化算法。它跟 MAML 类似，它们都靠梯度下降进行元优化，而且都是模型无关的算法。

Reptiled 的执行流程如下:
* 1) 采样一个任务, 
* 2) 在这个任务上进行多次梯度下降 
* 3) 把模型参数向新参数靠近

如下图中算法所示：
 给定初始参数 $$ \theta $$，$$ \text{SGD}(\mathcal{L}_{\tau_i}, \theta, k) $$ 根据 loss $$ \mathcal{L}_{\tau_i} $$ 进行 $k$ 次随机梯度下降，之后返回参数向量。带batch的版本则每次采样多个任务。Reptile的梯度定义为 $$ (\theta - W)/\alpha $$，其中 $$ \alpha $$ 是 SGD 所使用的步长。

![Reptile Algorithm]({{ '/assets/images/2019-09-19-meta-learning/reptile-algo.png' | relative_url }})
{: style="width: 52%;" class="center"}
*Fig. 13. Batch 版本的 Reptile 算法. (图像来源：[原论文](https://arxiv.org/abs/1803.02999))*

一眼看上去，这个算法跟普通的 SGD 很像。但是，因为内循环里的梯度下降可以发生好多次，使得 $$ \text{SGD}(\mathbb{E}
_\tau[\mathcal{L}_{\tau}], \theta, k) $$ 与 $$ \mathbb{E}_\tau [\text{SGD}(\mathcal{L}_{\tau}, \theta, k)] $$ 在 k > 1 时产生了区别。


#### The Optimization Assumption

假设任务 $$ \tau \sim p(\tau) $$ 有一个最优的模型参数空间构成的流形 $$ \mathcal{W}_{\tau}^* $$ 。 当参数 $$ \theta $$ 处于这个流形 $$ \mathcal{W}_{\tau}^* $$ 上面的时候，模型 $$ f_\theta $$ 能够在任务 $$ \tau $$ 上达到最好的效果。为了找到一个对于所有任务都足够好的模型，我们想要找一个靠近所有任务的最优流形的参数，即：

 $$ 
\theta^* = \arg\min_\theta \mathbb{E}_{\tau \sim p(\tau)} [\frac{1}{2} \text{dist}(\theta, \mathcal{W}_\tau^*)^2]
 $$ 


![Reptile Algorithm]({{ '/assets/images/2019-09-19-meta-learning/reptile-optim.png' | relative_url }})
{: style="width: 50%;" class="center"}
*Fig. 14. 为了靠近不同任务的最优流形，Reptile 算法在交替更新参数。 (图像来源：[原论文](https://arxiv.org/abs/1803.02999))*

我们设 $$ \text{dist}(.) $$ 为 L2 距离，并定义一个点 $$ \theta $$ 和一个集合 $$ \mathcal{W}_\tau^* $$ 之间的距离等价于 $$ \theta $$ 和一个该流形上最接近 $$ \theta $$ 的点 $$ W_{\tau}^*(\theta) $$ 的距离。

 $$ 
\text{dist}(\theta, \mathcal{W}_{\tau}^*) = \text{dist}(\theta, W_{\tau}^*(\theta)) \text{, where }W_{\tau}^*(\theta) = \arg\min_{W\in\mathcal{W}_{\tau}^*} \text{dist}(\theta, W)
 $$ 


欧拉距离平方的梯度为：

 $$ 
\begin{aligned}
\nabla_\theta[\frac{1}{2}\text{dist}(\theta, \mathcal{W}_{\tau_i}^*)^2]
&= \nabla_\theta[\frac{1}{2}\text{dist}(\theta, W_{\tau_i}^*(\theta))^2] & \\
&= \nabla_\theta[\frac{1}{2}(\theta - W_{\tau_i}^*(\theta))^2] & \\
&= \theta - W_{\tau_i}^*(\theta) & \scriptstyle{\text{; See notes.}}
\end{aligned}
 $$ 

注意：根据原论文，“*一个点 Θ 和一个集合 S 的欧拉距离平方的梯度为 2(Θ − p)，其中 p 是 S 中离 Θ 最近的点*"。理论上来说，S 中最靠近 Θ 的点应该也是一个 Θ 的函数，但我不确定为什么计算梯度的时候不需要担心 p 的导数。(如果有什么想法欢迎讨论。)

因此，一个随机梯度下降的更新步骤为：

 $$ 
\theta = \theta - \alpha \nabla_\theta[\frac{1}{2} \text{dist}(\theta, \mathcal{W}_{\tau_i}^*)^2] = \theta - \alpha(\theta - W_{\tau_i}^*(\theta)) = (1-\alpha)\theta + \alpha W_{\tau_i}^*(\theta)
 $$ 

虽然很难精确计算出任务最优流形上的最近点 $$ W_{\tau_i}^*(\theta) $$ ，但Reptile 可以根据 $$ \text{SGD}(\mathcal{L}_\tau, \theta, k) $$ 拟合出来。


#### Reptile vs FOMAML

为了说明 Reptile 和 MAML 的深层联系，我们用一个做了两步梯度下降的更新公式做例子，即 $$ \text{SGD}(.) $$ 中 k=2。 和[上面](#maml)定义的一样，$$ \mathcal{L}^{(0)} $$ 和 $$ \mathcal{L}^{(1)} $$ 只是不太batch对应的loss。为了方便阅读，我们使用两个记号：$$ g^{(i)}_j = \nabla_{\theta} \mathcal{L}^{(i)}(\theta_j) $$ 和 $$ H^{(i)}_j = \nabla^2_{\theta} \mathcal{L}^{(i)}(\theta_j) $$。

 $$ 
\begin{aligned}
\theta_0 &= \theta_\text{meta}\\
\theta_1 &= \theta_0 - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_0)= \theta_0 - \alpha g^{(0)}_0 \\
\theta_2 &= \theta_1 - \alpha\nabla_\theta\mathcal{L}^{(1)}(\theta_1) = \theta_0 - \alpha g^{(0)}_0 - \alpha g^{(1)}_1
\end{aligned}
 $$ 

如[前面章节](#first-order-maml)所述，FOMAML 的梯度是最后一次内循环梯度更新的结果。因此，当 k=1 时：

 $$ 
\begin{aligned}
g_\text{FOMAML} &= \nabla_{\theta_1} \mathcal{L}^{(1)}(\theta_1) = g^{(1)}_1 \\
g_\text{MAML} &= \nabla_{\theta_1} \mathcal{L}^{(1)}(\theta_1) \cdot (I - \alpha\nabla^2_{\theta} \mathcal{L}^{(0)}(\theta_0)) = g^{(1)}_1 - \alpha H^{(0)}_0 g^{(1)}_1
\end{aligned}
 $$ 

Reptile的梯度则定义为：

 $$ 
g_\text{Reptile} = (\theta_0 - \theta_2) / \alpha = g^{(0)}_0 + g^{(1)}_1
 $$ 


现在，我们可以得到：

![Reptile vs FOMAML]({{ '/assets/images/2019-09-19-meta-learning/reptile_vs_FOMAML.png' | relative_url }})
{: style="width: 50%;" class="center"}
*Fig. 15. Reptile 和 FOMAML 在一次外循环里的元优化方式对比. (图像来源：[slides](https://www.slideshare.net/YoonhoLee4/on-firstorder-metalearning-algorithms) on Reptile by Yoonho Lee.)*

 $$ 
\begin{aligned}
g_\text{FOMAML} &= g^{(1)}_1 \\
g_\text{MAML} &= g^{(1)}_1 - \alpha H^{(0)}_0 g^{(1)}_1 \\
g_\text{Reptile} &= g^{(0)}_0 + g^{(1)}_1
\end{aligned}
 $$ 


接下来，我们对 $$ g^{(1)}_1 $$ 使用 [泰勒展开](https://en.wikipedia.org/wiki/Taylor_series)。对一个可微函数 $$ f(x) $$ 进行 $$ a $$ 阶展开的式子为：

 $$ 
f(x) = f(a) + \frac{f'(a)}{1!}(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \dots = \sum_{i=0}^\infty \frac{f^{(i)}(a)}{i!}(x-a)^i
 $$ 

我们把 $$ \nabla_{\theta}\mathcal{L}^{(1)}(.) $$ 看做一个函数，把 $$ \theta_0 $$ 看做自变量。 $$ g_1^{(1)} $$ 在 $$ \theta_0 $$ 处的泰勒展开为：

 $$ 
\begin{aligned}
g_1^{(1)} &= \nabla_{\theta}\mathcal{L}^{(1)}(\theta_1) \\
&= \nabla_{\theta}\mathcal{L}^{(1)}(\theta_0) + \nabla^2_\theta\mathcal{L}^{(1)}(\theta_0)(\theta_1 - \theta_0) + \frac{1}{2}\nabla^3_\theta\mathcal{L}^{(1)}(\theta_0)(\theta_1 - \theta_0)^2 + \dots & \\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + \frac{\alpha^2}{2}\nabla^3_\theta\mathcal{L}^{(1)}(\theta_0) (g_0^{(0)})^2 + \dots & \scriptstyle{\text{; because }\theta_1-\theta_0=-\alpha g_0^{(0)}} \\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)
\end{aligned}
 $$ 

把 MAML 的一步内循环梯度更新用 $$ g_1^{(1)} $$ 的展开式重写：

 $$ 
\begin{aligned}
g_\text{FOMAML} &= g^{(1)}_1 = g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)\\
g_\text{MAML} &= g^{(1)}_1 - \alpha H^{(0)}_0 g^{(1)}_1 \\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2) - \alpha H^{(0)}_0 (g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2))\\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} - \alpha H^{(0)}_0 g_0^{(1)} + \alpha^2 \alpha H^{(0)}_0 H^{(1)}_0 g_0^{(0)} + O(\alpha^2)\\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} - \alpha H^{(0)}_0 g_0^{(1)} + O(\alpha^2)
\end{aligned}
 $$ 


Reptile的梯度变为：

 $$ 
\begin{aligned}
g_\text{Reptile} 
&= g^{(0)}_0 + g^{(1)}_1 \\
&= g^{(0)}_0 + g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)
\end{aligned}
 $$ 


现在，我们有了三种不同的梯度更新法则：

 $$ 
\begin{aligned}
g_\text{FOMAML} &= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)\\
g_\text{MAML} &= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} - \alpha H^{(0)}_0 g_0^{(1)} + O(\alpha^2)\\
g_\text{Reptile}  &= g^{(0)}_0 + g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)
\end{aligned}
 $$ 

在训练过程中，我们经常会对多个数据batch做平均。在我们的例子中，小batch (0) 和 (1)是可交换的， 因为他们都是随机选取的。期望 $$ \mathbb{E}_{\tau,0,1} $$ 表示在任务$$ \tau $$上，两个标号为 (0) 和 (1) 的数据batch的平均值。

令,

- $$ A = \mathbb{E}_{\tau,0,1} [g_0^{(0)}] = \mathbb{E}_{\tau,0,1} [g_0^{(1)}] $$ ; 代表着任务loss的平均梯度。沿着 $$ A $$ 代表的方向更新模型，能够使模型在当前任务上的表现更好。
- $$ B = \mathbb{E}_{\tau,0,1} [H^{(1)}_0 g_0^{(0)}] = \frac{1}{2}\mathbb{E}_{\tau,0,1} [H^{(1)}_0 g_0^{(0)} + H^{(0)}_0 g_0^{(1)}] = \frac{1}{2}\mathbb{E}_{\tau,0,1} [\nabla_\theta(g^{(0)}_0 g_0^{(1)})] $$ ; 代表着增加同个任务上两个小batch梯度的内积的方向。沿着 $$ B $$ 代表的方向更新模型，能够使得模型在当前任务上具有更好的泛化性。

综上所述， MAML 和 Reptile 的优化目标相同，都是更好的任务表现（由 A 主导）和更好的泛化能力（由 B 主导）。当梯度更新由泰勒展开的前三项近似时：

 $$ 
\begin{aligned}
\mathbb{E}_{\tau,1,2}[g_\text{FOMAML}] &= A - \alpha B + O(\alpha^2)\\
\mathbb{E}_{\tau,1,2}[g_\text{MAML}] &= A - 2\alpha B + O(\alpha^2)\\
\mathbb{E}_{\tau,1,2}[g_\text{Reptile}]  &= 2A - \alpha B + O(\alpha^2)
\end{aligned}
 $$ 

虽然我们没法确定忽略的 $$ O(\alpha^2) $$ 项是否对于参数更新有着重要作用。但根据 FOMAML 与 MAML 的表现相近来看，说高阶导数对于梯度更新不太重要还是比较靠谱的。

---

引用本文:

```
@article{weng2018metalearning,
  title   = "Meta-Learning: Learning to Learn Fast",
  author  = "Weng, Lilian",
  journal = "lilianweng.github.io/lil-log",
  year    = "2018",
  url     = "http://lilianweng.github.io/lil-log/2018/11/29/meta-learning.html"
}
```

*如果你发现本文章有错误，欢迎联系作者（lilian dot wengweng at gmail dot com）或译者（phi.wth at gmail dot com）。非常感谢！*

敬请期待更多文章！


## 引用

[1] Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum. ["Human-level concept learning through probabilistic program induction."](https://www.cs.cmu.edu/~rsalakhu/papers/LakeEtAl2015Science.pdf) Science 350.6266 (2015): 1332-1338.

[2] Oriol Vinyals' talk on ["Model vs Optimization Meta Learning"](http://metalearning-symposium.ml/files/vinyals.pdf)

[3] Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. ["Siamese neural networks for one-shot image recognition."](http://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf) ICML Deep Learning Workshop. 2015.

[4] Oriol Vinyals, et al. ["Matching networks for one shot learning."](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) NIPS. 2016.

[5] Flood Sung, et al. ["Learning to compare: Relation network for few-shot learning."](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Sung_Learning_to_Compare_CVPR_2018_paper.pdf) CVPR. 2018.

[6] Jake Snell, Kevin Swersky, and Richard Zemel. ["Prototypical Networks for Few-shot Learning."](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf) CVPR. 2018.

[7] Adam Santoro, et al. ["Meta-learning with memory-augmented neural networks."](http://proceedings.mlr.press/v48/santoro16.pdf) ICML. 2016.

[8] Alex Graves, Greg Wayne, and Ivo Danihelka. ["Neural turing machines."](https://arxiv.org/abs/1410.5401) arXiv preprint arXiv:1410.5401 (2014).

[9] Tsendsuren Munkhdalai and Hong Yu. ["Meta Networks."](https://arxiv.org/abs/1703.00837) ICML. 2017.

[10] Sachin Ravi and Hugo Larochelle. ["Optimization as a Model for Few-Shot Learning."](https://openreview.net/pdf?id=rJY0-Kcll) ICLR. 2017.

[11] Chelsea Finn's BAIR blog on ["Learning to Learn"](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/).

[12] Chelsea Finn, Pieter Abbeel, and Sergey Levine. ["Model-agnostic meta-learning for fast adaptation of deep networks."](https://arxiv.org/abs/1703.03400) ICML 2017.

[13] Alex Nichol, Joshua Achiam, John Schulman. ["On First-Order Meta-Learning Algorithms."](https://arxiv.org/abs/1803.02999) arXiv preprint arXiv:1803.02999 (2018).

[14] [Slides on Reptile](https://www.slideshare.net/YoonhoLee4/on-firstorder-metalearning-algorithms) by Yoonho Lee.


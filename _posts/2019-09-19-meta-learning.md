---
layout: post
comments: true
title: "元学习: 学习如何学习"
date: 2019-9-17 00:00:00
tags: meta-learning long-read
---

> 学习如何学习的方法被称为元学习。元学习的目标是在接触到没见过的任务或者迁移到新环境中时，可以根据之前的经验和少量的样本快速学习如何应对。元学习有三种常见的实现方法：1）学习有效的距离度量方式（基于度量的方法）；2）使用带有显式或隐式记忆储存的（循环）神经网络（基于模型的方法）；3）训练以快速学习为目标的模型（基于优化的方法）

<!--more-->

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

我们现在假设有一个任务的分布，我们从这个分布中采样了许多任务作为训练集。好的元学习模型在这个训练集上训练后，应当对这个空间里所有的任务都具有良好的表现，即使是从来没见过的任务。每个任务可以表示为一个数据集$$\mathcal{D}$$，数据集中包括特征向量$$x$$和标签$$y$$，分布表示为$$p(\mathcal{D})$$。那么最佳的元学习模型参数可以表示为：

$$
\theta^* = \arg\min_\theta \mathbb{E}_{\mathcal{D}\sim p(\mathcal{D})} [\mathcal{L}_\theta(\mathcal{D})]
$$

上式的形式跟一般的学习任务非常像，只不过上式中的每个*数据集*是一个*数据样本*。

*少样本学习（Few-shot classification）* 是元学习的在监督学习中的一个实例。数据集$$\mathcal{D}$$经常被划分为两部分，一个用于学习的支持集（support set）$$S$$，和一个用于训练和测试的预测集（prediction set）$$B$$，即$$\mathcal{D}=\langle S, B\rangle$$。*K-shot N-class*分类任务，即支持集中有N类数据，每类数据有K个带有标注的样本。

![few-shot-classification]({{ '/assets/images/2019-09-19-meta-learning/few-shot-classification.png' | relative_url }}) {: style="width: 100%;" class="center"} Fig. 1. An example of 4-shot 2-class image classification. (Image thumbnails are from Pinterest)

![few-shot-classification](../assets/images/2019-09-19-meta-learning/few-shot-classification.png)
*Fig. 1. 4-shot 2-class 图像分类的例子。 (图像来自[Pinterest](https://www.pinterest.com/))*


### 像测试一样训练


一个数据集$$\mathcal{D}$$包含许多对特征向量和标签，即$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}$$。每个标签属于一个标签类$$\mathcal{L}$$。假设我们的分类器$$f_\theta$$的输入是特征向量$$\mathbf{x}$$，输出是属于第$$y$$类的概率$$P_\theta(y\vert\mathbf{x})$$，$$\theta$$是分类器的参数。 

如果我们每次选一个$$B \subset \mathcal{D}$$作为训练的batch，则最佳的模型参数，应当能够最大化，多组batch的正确标签概率之和。

$$
\begin{aligned}
\theta^* &= {\arg\max}_{\theta} \mathbb{E}_{(\mathbf{x}, y)\in \mathcal{D}}[P_\theta(y \vert \mathbf{x})] &\\
\theta^* &= {\arg\max}_{\theta} \mathbb{E}_{B\subset \mathcal{D}}[\sum_{(\mathbf{x}, y)\in B}P_\theta(y \vert \mathbf{x})] & \scriptstyle{\text{; trained with mini-batches.}}
\end{aligned}
$$

few-shot classification的目标是，在小规模的support set上“快速学习”（类似fine-tuning）后，能够减少在prediction set上的预测误差。为了训练模型快速学习的能力，我们在训练的时候按照以下步骤：
1. 采样一个标签的子集, $$L\subset\mathcal{L}$$.
2. 根据采样的标签子集，采样一个support set $$S^L \subset \mathcal{D}$$ 和一个training batch $$B^L \subset \mathcal{D}$$。$$S^L$$和$$B^L$$中的数据的标签都属于$$L$$，即$$y \in L, \forall (x, y) \in S^L, B^L$$.
3. 把support set作为模型的输入，进行“快速学习”。注意，不同的算法具有不同的学习策略，但总的来说，这一步不会永久性更新模型参数。 <!-- , $$\hat{y}=f_\theta(\mathbf{x}, S^L)$$ -->
4. 把prediction set作为模型的输入，计算模型在$$B^L$$上的loss，根据这个loss进行反向传播更新模型参数。这一步与监督学习一致。

你可以把每一对$$(S^L, B^L)$$看做是一个数据点。模型被训练出了在其他数据集上扩展的能力。下式中的红色部分是元学习的目标相比于监督学习的目标多出来的部分。

$$
\theta = \arg\max_\theta \color{red}{E_{L\subset\mathcal{L}}[} E_{\color{red}{S^L \subset\mathcal{D}, }B^L \subset\mathcal{D}} [\sum_{(x, y)\in B^L} P_\theta(x, y\color{red}{, S^L})] \color{red}{]}
$$

这个想法有点像是我们面对某个只有少量数据的任务时，会使用在相关任务的大数据集上预训练的模型，然后进行fine-tuning。像是图形语义分割网络可以用在ImageNet上预训练的模型做初始化。相比于在一个特定任务上fine-tuning使得模型更好的拟合这个任务，元学习更进一步，它的目标是让模型优化以后能够在多个任务上表现的更好，类似于变得更容易被fine-tuning。

### 学习器和元学习器

还有一种常见的看待meta-learning的视角，把模型的更新划分为了两个阶段：
- 根据给定的任务，训练一个分类器$$f_\theta$$完成任务，作为“学习器”模型
- 同时，训练一个元学习器$$g_\phi$$，根据support set $$S$$学习如何更新学习器模型的参数。$$\theta' = g_\phi(\theta, S)$$

则最后的优化目标中，我们需要更新$$\theta$$和$$\phi$$来最大化：

$$
\mathbb{E}_{L\subset\mathcal{L}}[ \mathbb{E}_{S^L \subset\mathcal{D}, B^L \subset\mathcal{D}} [\sum_{(\mathbf{x}, y)\in B^L} P_{g_\phi(\theta, S^L)}(y \vert \mathbf{x})]]
$$


### 常见方法

元学习主要有三类常见的方法：metric-based，model-based，optimization-based。
Oriol Vinyals在meta-learning symposium @ NIPS 2018上做了一个很好的[总结](http://metalearning-symposium.ml/files/vinyals.pdf)：



|  | Model-based | Metric-based | Optimization-based |
| ------------- | ------------- | ------------- | ------------- |
| **Key idea** | RNN; memory | Metric learning | Gradient descent |
| **How $$P_\theta(y \vert \mathbf{x})$$ is modeled?** | $$f_\theta(\mathbf{x}, S)$$ | $$\sum_{(\mathbf{x}_i, y_i) \in S} k_\theta(\mathbf{x}, \mathbf{x}_i)y_i$$ (*) | $$P_{g_\phi(\theta, S^L)}(y \vert \mathbf{x})$$ |

(*) $$k_\theta$$ 是一个衡量$$\mathbf{x}_i$$和$$\mathbf{x}$$相似度的kernel function。

接下来我们会回顾各种方法的经典模型。


## Metric-Based

Metric-based meta-learning的核心思想类似于最近邻算法([k-NN分类](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)、[k-means聚类](https://en.wikipedia.org/wiki/K-means_clustering))和[核密度估计](https://en.wikipedia.org/wiki/Kernel_density_estimation)。该类方法在已知标签的集合上预测出来的概率，是support set中的样本标签的加权和。 权重由核函数（kernal function）$$k_\theta$$算得，该权重代表着两个数据样本之间的相似性。

$$
P_\theta(y \vert \mathbf{x}, S) = \sum_{(\mathbf{x}_i, y_i) \in S} k_\theta(\mathbf{x}, \mathbf{x}_i)y_i 
$$

因此，学到一个好的核函数对于metric-based meta-learning模型至关重要。[Metric learning](https://en.wikipedia.org/wiki/Similarity_learning#Metric_learning)正是针对该问题提出的方法，它的目标就是学到一个不同样本之间的metric或者说是距离函数。任务不同，好的metric的定义也不同。但它一定在任务空间上表示了输入之间的联系，并且能够帮助我们解决问题。

下面列出的所有方法都显式的学习了输入数据的嵌入向量（embedding vectors），并根据其设计合适的kernel function。

### Convolutional Siamese Neural Network

[Siamese Neural Network](https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf)最早被提出用来解决笔迹验证问题，siamese network由两个孪生网络组成，这两个网络的输出被联合起来训练一个函数，用于学习一对数据输入之间的关系。这两个网络结构相同，共享参数，实际上就是一个网络在学习如何有效地embedding才能显现出一对数据之间的关系。顺便一提，这是LeCun 1994年的论文。

[Koch, Zemel & Salakhutdinov (2015)](http://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf)提出了一种用siamese网络做one-shot image classification的方法。首先，训练一个用于图片验证的siamese网络，分辨两张图片是否属于同一类。然后在测试时，siamese网络把测试输入和support set里面的所有图片进行比较，选择相似度最高的那张图片所属的类作为输出。

![siamese](/assets/images/2019-09-19-meta-learning/siamese-conv-net.png)
*Fig. 2. 卷积siamese网络用于few-shot image classification的例子。*

1. 首先，卷积siamese网络学习一个由多个卷积层组成的embedding函数$$f_\theta$$，把两张图片编码为特征向量。
2. 两个特征向量之间的L1距离可以表示为$$\vert f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j) \vert$$。
3. 通过一个linear feedforward layer和sigmoid把距离转换为概率。这就是两张图片属于同一类的概率。
4. loss函数就是cross entropy loss，因为label是二元的。

<!-- In this way, an efficient image embedding is trained so that the distance between two embeddings is proportional to the similarity between two images. -->

$$
\begin{aligned}
p(\mathbf{x}_i, \mathbf{x}_j) &= \sigma(\mathbf{W}\vert f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j) \vert) \\
\mathcal{L}(B) &= \sum_{(\mathbf{x}_i, \mathbf{x}_j, y_i, y_j)\in B} \mathbf{1}_{y_i=y_j}\log p(\mathbf{x}_i, \mathbf{x}_j) + (1-\mathbf{1}_{y_i=y_j})\log (1-p(\mathbf{x}_i, \mathbf{x}_j))
\end{aligned}
$$

Training batch $$B$$可以通过对图片做一些变形增加数据量。你也可以把L1距离替换成其他距离，比如L2距离、cosine距离等等。只要距离是可导的就可以。

给定一个支持集$$S$$和一个测试图片$$\mathbf{x}$$，最终预测的分类为：

$$
\hat{c}_S(\mathbf{x}) = c(\arg\max_{\mathbf{x}_i \in S} P(\mathbf{x}, \mathbf{x}_i))
$$

$$c(\mathbf{x})$$是图片$$\mathbf{x}$$的label，$$\hat{c}(.)$$是预测的label。

这里我们有一个假设：学到的embedding在未见过的分类上依然能很好的衡量图片间的距离。这个假设跟迁移学习中使用预训练模型所隐含的假设是一样的。比如，在ImageNet上预训练的模型，其学到的卷积特征表达方式对于其他图像任务也有帮助。但实际上当新任务与旧任务有所差别的时候，预训练模型的效果就没有那么好了。

### Matching Networks


**Matching Networks** ([Vinyals et al., 2016](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf))的目标是：对于每一个给定的支持集$$S=\{x_i, y_i\}_{i=1}^k$$ (*k-shot* classification)，分别学一个分类器$$c_S$$。 这个分类器给出了给定测试样本$$\mathbf{x}$$时，输出$$y$$的概率分布。这个分类器的输出被定义为支持集中一系列label的加权和，权重由一个注意力核（attention kernel）$$a(\mathbf{x}, \mathbf{x}_i)$$决定。权重应当与$$\mathbf{x}$$和$$\mathbf{x}_i$$间的相似度成正比。

<img src="../assets/images/2019-09-19-meta-learning/matching-networks.png" width="70%">

*Fig. 3. Matching Networks结构。（图像来源: [original paper](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)）*


$$
c_S(\mathbf{x}) = P(y \vert \mathbf{x}, S) = \sum_{i=1}^k a(\mathbf{x}, \mathbf{x}_i) y_i
\text{, where }S=\{(\mathbf{x}_i, y_i)\}_{i=1}^k
$$

Attention kernel由两个embedding function $$f$$和$$g$$决定。分别用于encoding测试样例和支持集样本。两个样本之间的注意力权重是经过softmax归一化后的，他们embedding vectors的cosine距离$$\text{cosine}(.)$$。

$$
a(\mathbf{x}, \mathbf{x}_i) = \frac{\exp(\text{cosine}(f(\mathbf{x}), g(\mathbf{x}_i))}{\sum_{j=1}^k\exp(\text{cosine}(f(\mathbf{x}), g(\mathbf{x}_j))}
$$


#### Simple Embedding

在简化版本里，embedding function是一个使用单样本作为输入的神经网络。而且我们可以假设$$f=g$$。

#### Full Context Embeddings

Embeding vectors对于构建一个好的分类器至关重要。只把一个数据样本作为embedding function的输入，会导致很难高效的估计出整个特征空间。因此，Matching Network模型又进一步发展，通过把整个支持集$$S$$作为embedding function的额外输入来加强embedding的有效性，相当于给样本添加了语境，让embedding根据样本与支持集中样本的关系进行调整。


- $$g_\theta(\mathbf{x}_i, S)$$在整个支持集$$S$$的语境下用一个双向LSTM来编码$$\mathbf{x}_i$$.

- $$f_\theta(\mathbf{x}, S)$$在支持集$$S$$上使用read attention机制编码测试样本$$\mathbf{x}$$。
    1. 首先测试样本经过一个简单的神经网络，比如CNN，以抽取基本特征$$f'(\mathbf{x})$$。
    2. 然后，一个带有read attention vector的LSTM被训练用于生成部分hidden state：<br/>
    $$
    \begin{aligned}
    \hat{\mathbf{h}}_t, \mathbf{c}_t &= \text{LSTM}(f'(\mathbf{x}), [\mathbf{h}_{t-1}, \mathbf{r}_{t-1}], \mathbf{c}_{t-1}) \\
    \mathbf{h}_t &= \hat{\mathbf{h}}_t + f'(\mathbf{x}) \\
    \mathbf{r}_{t-1} &= \sum_{i=1}^k a(\mathbf{h}_{t-1}, g(\mathbf{x}_i)) g(\mathbf{x}_i) \\
    a(\mathbf{h}_{t-1}, g(\mathbf{x}_i)) &= \text{softmax}(\mathbf{h}_{t-1}^\top g(\mathbf{x}_i)) = \frac{\exp(\mathbf{h}_{t-1}^\top g(\mathbf{x}_i))}{\sum_{j=1}^k \exp(\mathbf{h}_{t-1}^\top g(\mathbf{x}_j))}
    \end{aligned}
    $$
    1. 最终，如果我们做k步的读取$$f(\mathbf{x}, S)=\mathbf{h}_K$$。

这类embedding方法被称作“全语境嵌入”（Full Contextual Embeddings）。有意思的是，这类方法对于困难的任务（few-shot classification on mini ImageNet）有所帮助，但对于简单的任务却没有提升（Omniglot）。

Matching Networks的训练过程与测试时的推理过程是一致的，详情请回顾之前的[章节](#像测试一样训练)。值得一提的是，Matching Networks的论文强调了训练和测试的条件应当一致的原则。

$$
\theta^* = \arg\max_\theta \mathbb{E}_{L\subset\mathcal{L}}[ \mathbb{E}_{S^L \subset\mathcal{D}, B^L \subset\mathcal{D}} [\sum_{(\mathbf{x}, y)\in B^L} P_\theta(y\vert\mathbf{x}, S^L)]]
$$



### Relation Network

**Relation Network (RN)** ([Sung et al., 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Sung_Learning_to_Compare_CVPR_2018_paper.pdf))与[siamese network](#convolutional-siamese-neural-network)有所相似，但有以下几个不同点：
1. 两个样本间的相似系数不是由特征空间的L1距离决定的，而是由一个CNN分类器$$g_\phi$$预测的。两个样本$$\mathbf{x}_i$$和$$\mathbf{x}_j$$间的相似系数为$$r_{ij} = g_\phi([\mathbf{x}_i, \mathbf{x}_j])$$，其中$$[.,.]$$代表着concatenation。
2. 目标优化函数是MSE损失，而不是cross-entropy，因为RN在预测时更倾向于把相似系数预测过程作为一个regression问题，而不是二分类问题，$$\mathcal{L}(B) = \sum_{(\mathbf{x}_i, \mathbf{x}_j, y_i, y_j)\in B} (r_{ij} - \mathbf{1}_{y_i=y_j})^2$$


![relation-network]({{ '/assets/images/relation-network.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4. Relation Network architecture for a 5-way 1-shot problem with one query example. (Image source: [original paper](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Sung_Learning_to_Compare_CVPR_2018_paper.pdf))*

(Note: There is another [Relation Network](https://deepmind.com/blog/neural-approach-relational-reasoning/) for relational reasoning, proposed by DeepMind. Don't get confused.)


### Prototypical Networks

**Prototypical Networks** ([Snell, Swersky & Zemel, 2017](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)) use an embedding function $$f_\theta$$ to encode each input into a $$M$$-dimensional feature vector. A *prototype* feature vector is defined for every class $$c \in \mathcal{C}$$, as the mean vector of the embedded support data samples in this class.

$$
\mathbf{v}_c = \frac{1}{|S_c|} \sum_{(\mathbf{x}_i, y_i) \in S_c} f_\theta(\mathbf{x}_i)
$$


![prototypical-networks]({{ '/assets/images/prototypical-networks.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 5. Prototypical networks in the few-shot and zero-shot scenarios. (Image source: [original paper](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf))*

The distribution over classes for a given test input $$\mathbf{x}$$ is a softmax over the inverse of distances between the test data embedding and prototype vectors.

$$
P(y=c\vert\mathbf{x})=\text{softmax}(-d_\varphi(f_\theta(\mathbf{x}), \mathbf{v}_c)) = \frac{\exp(-d_\varphi(f_\theta(\mathbf{x}), \mathbf{v}_c))}{\sum_{c' \in \mathcal{C}}\exp(-d_\varphi(f_\theta(\mathbf{x}), \mathbf{v}_{c'}))}
$$

where $$d_\varphi$$ can be any distance function as long as $$\varphi$$ is differentiable. In the paper, they used the squared euclidean distance.

The loss function is the negative log-likelihood: $$\mathcal{L}(\theta) = -\log P_\theta(y=c\vert\mathbf{x})$$.



## Model-Based

Model-based meta-learning models make no assumption on the form of $$P_\theta(y\vert\mathbf{x})$$. Rather it depends on a model designed specifically for fast learning --- a model that updates its parameters rapidly with a few training steps. This rapid parameter update can be achieved by its internal architecture or controlled by another meta-learner model. 


### Memory-Augmented Neural Networks

A family of model architectures use external memory storage to facilitate the learning process of neural networks, including [Neural Turing Machines](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#neural-turing-machines) and [Memory Networks](https://arxiv.org/abs/1410.3916). With an explicit storage buffer, it is easier for the network to rapidly incorporate new information and not to forget in the future. Such a model is known as **MANN**, short for "**Memory-Augmented Neural Network**".  Note that recurrent neural networks with only *internal memory* such as vanilla RNN or LSTM are not MANNs.

Because MANN is expected to encode new information fast and thus to adapt to new tasks after only a few samples, it fits well for meta-learning. Taking the Neural Turing Machine (NTM) as the base model, [Santoro et al. (2016)](http://proceedings.mlr.press/v48/santoro16.pdf) proposed a set of modifications on the training setup and the memory retrieval mechanisms (or "addressing mechanisms", deciding how to assign attention weights to memory vectors). Please go through [the NTM section](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#neural-turing-machines) in my other post first if you are not familiar with this matter before reading forward.

As a quick recap, NTM couples a controller neural network with external memory storage. The controller learns to read and write memory rows by soft attention, while the memory serves as a knowledge repository. The attention weights are generated by its addressing mechanism: content-based + location based.


![NTM]({{ '/assets/images/NTM.png' | relative_url }})
{: style="width: 70%;" class="center"}
*Fig. 6. The architecture of Neural Turing Machine (NTM). The memory at time t, $$\mathbf{M}_t$$ is a matrix of size $$N \times M$$, containing N vector rows and each has M dimensions.*


#### MANN for Meta-Learning

To use MANN for meta-learning tasks, we need to train it in a way that the memory can encode and capture information of new tasks fast and, in the meantime, any stored representation is easily and stably accessible.

The training described in [Santoro et al., 2016](http://proceedings.mlr.press/v48/santoro16.pdf) happens in an interesting way so that the memory is forced to hold information for longer until the appropriate labels are presented later. In each training episode, the truth label $$y_t$$ is presented with **one step offset**, $$(\mathbf{x}_{t+1}, y_t)$$: it is the true label for the input at the previous time step t, but presented as part of the input at time step t+1. 


![NTM]({{ '/assets/images/mann-meta-learning.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 7. Task setup in MANN for meta-learning (Image source: [original paper](http://proceedings.mlr.press/v48/santoro16.pdf)).*

In this way, MANN is motivated to memorize the information of a new dataset, because the memory has to hold the current input until the label is present later and then retrieve the old information to make a prediction accordingly.

Next let us see how the memory is updated for efficient information retrieval and storage.


#### Addressing Mechanism for Meta-Learning

Aside from the training process, a new pure content-based addressing mechanism is utilized to make the model better suitable for meta-learning.


**>> How to read from memory?**
<br/>
The read attention is constructed purely based on the content similarity.

First, a key feature vector $$\mathbf{k}_t$$ is produced at the time step t by the controller as a function of the input $$\mathbf{x}$$. Similar to NTM, a read weighting vector $$\mathbf{w}_t^r$$ of N elements is computed as the cosine similarity between the key vector and every memory vector row, normalized by softmax. The read vector $$\mathbf{r}_t$$ is a sum of memory records weighted by such weightings:

$$
\mathbf{r}_i = \sum_{i=1}^N w_t^r(i)\mathbf{M}_t(i)
\text{, where } w_t^r(i) = \text{softmax}(\frac{\mathbf{k}_t \cdot \mathbf{M}_t(i)}{\|\mathbf{k}_t\| \cdot \|\mathbf{M}_t(i)\|})
$$

where $$M_t$$ is the memory matrix at time t and $$M_t(i)$$ is the i-th row in this matrix.


**>> How to write into memory?**
<br/>
The addressing mechanism for writing newly received information into memory operates a lot like the [cache replacement](https://en.wikipedia.org/wiki/Cache_replacement_policies) policy. The **Least Recently Used Access (LRUA)** writer is designed for MANN to better work in the scenario of meta-learning. A LRUA write head prefers to write new content to either the *least used* memory location or the *most recently used* memory location.
* Rarely used locations: so that we can preserve frequently used information (see [LFU](https://en.wikipedia.org/wiki/Least_frequently_used));
* The last used location: the motivation is that once a piece of information is retrieved once, it probably won't be called again for a while (see [MRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Most_recently_used_(MRU))). 

There are many cache replacement algorithms and each of them could potentially replace the design here with better performance in different use cases. Furthermore, it would be a good idea to learn the memory usage pattern and addressing strategies rather than arbitrarily set it.

The preference of LRUA is carried out in a way that everything is differentiable:
1. The usage weight $$\mathbf{w}^u_t$$ at time t is a sum of current read and write vectors, in addition to the decayed last usage weight, $$\gamma \mathbf{w}^u_{t-1}$$, where $$\gamma$$ is a decay factor. 
2. The write vector is an interpolation between the previous read weight (prefer "the last used location") and the previous least-used weight (prefer "rarely used location"). The interpolation parameter is the sigmoid of a hyperparameter $$\alpha$$.
3. The least-used weight $$\mathbf{w}^{lu}$$ is scaled according to usage weights $$\mathbf{w}_t^u$$, in which any dimension remains at 1 if smaller than the n-th smallest element in the vector and 0 otherwise.


$$
\begin{aligned}
\mathbf{w}_t^u &= \gamma \mathbf{w}_{t-1}^u + \mathbf{w}_t^r + \mathbf{w}_t^w \\
\mathbf{w}_t^r &= \text{softmax}(\text{cosine}(\mathbf{k}_t, \mathbf{M}_t(i))) \\
\mathbf{w}_t^w &= \sigma(\alpha)\mathbf{w}_{t-1}^r + (1-\sigma(\alpha))\mathbf{w}^{lu}_{t-1}\\
\mathbf{w}_t^{lu} &= \mathbf{1}_{w_t^u(i) \leq m(\mathbf{w}_t^u, n)}
\text{, where }m(\mathbf{w}_t^u, n)\text{ is the }n\text{-th smallest element in vector }\mathbf{w}_t^u\text{.}
\end{aligned}
$$


Finally, after the least used memory location, indicated by $$\mathbf{w}_t^{lu}$$, is set to zero, every memory row is updated:

$$
\mathbf{M}_t(i) = \mathbf{M}_{t-1}(i) + w_t^w(i)\mathbf{k}_t, \forall i
$$



### Meta Networks

**Meta Networks** ([Munkhdalai & Yu, 2017](https://arxiv.org/abs/1703.00837)), short for **MetaNet**, is a meta-learning model with architecture and training process designed for *rapid* generalization across tasks. 


#### Fast Weights

The rapid generalization of MetaNet relies on "fast weights". There are a handful of papers on this topic, but I haven't read all of them in detail and I failed to find a very concrete definition, only a vague agreement on the concept. Normally weights in the neural networks are updated by stochastic gradient descent in an objective function and this process is known to be slow. One faster way to learn is to utilize one neural network to predict the parameters of another neural network and the generated weights are called *fast weights*. In comparison, the ordinary SGD-based weights are named *slow weights*.  

In MetaNet, loss gradients are used as *meta information* to populate models that learn fast weights. Slow and fast weights are combined to make predictions in neural networks.


![slow-fast-weights]({{ '/assets/images/combine-slow-fast-weights.png' | relative_url }})
{: style="width: 50%;" class="center"}
*Fig. 8. Combining slow and fast weights in a MLP. $$\bigoplus$$ is element-wise sum. (Image source: [original paper](https://arxiv.org/abs/1703.00837)).*


#### Model Components

> Disclaimer: Below you will find my annotations are different from those in the paper. imo, the paper is poorly written, but the idea is still interesting. So I'm presenting the idea in my own language.


Key components of MetaNet are:
- An embedding function $$f_\theta$$, parameterized by $$\theta$$, encodes raw inputs into feature vectors. Similar to [Siamese Neural Network](#convolutional-siamese-neural-network), these embeddings are trained to be useful for telling whether two inputs are of the same class (verification task).
- A base learner model $$g_\phi$$, parameterized by weights $$\phi$$, completes the actual learning task.

If we stop here, it looks just like [Relation Network](#relation-network). MetaNet, in addition, explicitly models the fast weights of both functions and then aggregates them back into the model (See Fig. 8). 

Therefore we need additional two functions to output fast weights for $$f$$ and $$g$$ respectively.
- $$F_w$$: a LSTM parameterized by $$w$$ for learning fast weights $$\theta^+$$ of the embedding function $$f$$. It takes as input gradients of $$f$$'s embedding loss for verification task.
- $$G_v$$: a neural network parameterized by $$v$$ learning fast weights $$\phi^+$$ for the base learner $$g$$ from its loss gradients. In MetaNet, the learner's loss gradients are viewed as the *meta information* of the task.

Ok, now let's see how meta networks are trained. The training data contains multiple pairs of datasets: a support set $$S=\{\mathbf{x}'_i, y'_i\}_{i=1}^K$$ and a test set  $$U=\{\mathbf{x}_i, y_i\}_{i=1}^L$$. Recall that we have four networks and four sets of model parameters to learn, $$(\theta, \phi, w, v)$$.


![meta-net]({{ '/assets/images/meta-network.png' | relative_url }})
{: style="width: 90%;" class="center"}
*Fig.9. The MetaNet architecture.*


#### Training Process

1. Sample a random pair of inputs at each time step t from the support set $$S$$, $$(\mathbf{x}'_i, y'_i)$$ and $$(\mathbf{x}'_j, y_j)$$. Let $$\mathbf{x}_{(t,1)}=\mathbf{x}'_i$$ and $$\mathbf{x}_{(t,2)}=\mathbf{x}'_j$$.<br/>
for $$t = 1, \dots, K$$:
    * a\. Compute a loss for representation learning; i.e., cross entropy for the verification task:<br/>
      $$\mathcal{L}^\text{emb}_t = \mathbf{1}_{y'_i=y'_j} \log P_t + (1 - \mathbf{1}_{y'_i=y'_j})\log(1 - P_t)\text{, where }P_t = \sigma(\mathbf{W}\vert f_\theta(\mathbf{x}_{(t,1)}) - f_\theta(\mathbf{x}_{(t,2)})\vert)$$
2. Compute the task-level fast weights:
$$\theta^+ = F_w(\nabla_\theta \mathcal{L}^\text{emb}_1, \dots, \mathcal{L}^\text{emb}_T)$$
3. Next go through examples in the support set $$S$$ and compute the example-level fast weights. Meanwhile, update the memory with learned representations.<br/>
for $$i=1, \dots, K$$:
    * a\. The base learner outputs a probability distribution: $$P(\hat{y}_i \vert \mathbf{x}_i) = g_\phi(\mathbf{x}_i)$$ and the loss can be cross-entropy or MSE: $$\mathcal{L}^\text{task}_i = y'_i \log g_\phi(\mathbf{x}'_i) + (1- y'_i) \log (1 - g_\phi(\mathbf{x}'_i))$$
    * b\. Extract meta information (loss gradients) of the task and compute the example-level fast weights:
      $$\phi_i^+ = G_v(\nabla_\phi\mathcal{L}^\text{task}_i)$$
        * Then store $$\phi^+_i$$ into $$i$$-th location of the "value" memory $$\mathbf{M}$$.<br/>
    * d\. Encode the support sample into a task-specific input representation using both slow and fast weights: $$r'_i = f_{\theta, \theta^+}(\mathbf{x}'_i)$$
        * Then store $$r'_i$$ into $$i$$-th location of the "key" memory $$\mathbf{R}$$. 
4. Finally it is the time to construct the training loss using the test set $$U=\{\mathbf{x}_i, y_i\}_{i=1}^L$$.<br/>
Starts with $$\mathcal{L}_\text{train}=0$$:<br/>
for $$j=1, \dots, L$$:
    * a\. Encode the test sample into a task-specific input representation:
      $$r_j = f_{\theta, \theta^+}(\mathbf{x}_j)$$
    * b\. The fast weights are computed by attending to representations of support set samples in memory $$\mathbf{R}$$. The attention function is of your choice. Here MetaNet uses cosine similarity:<br/>
  $$
    \begin{aligned}
    a_j &= \text{cosine}(\mathbf{R}, r_j) = [\frac{r'_1\cdot r_j}{\|r'_1\|\cdot\|r_j\|}, \dots, \frac{r'_N\cdot r_j}{\|r'_N\|\cdot\|r_j\|}]\\
    \phi^+_j &= \text{softmax}(a_j)^\top \mathbf{M}
    \end{aligned}
  $$
    * c\. Update the training loss: $$\mathcal{L}_\text{train} \leftarrow \mathcal{L}_\text{train} + \mathcal{L}^\text{task}(g_{\phi, \phi^+}(\mathbf{x}_i), y_i) $$
5. Update all the parameters $$(\theta, \phi, w, v)$$ using $$\mathcal{L}_\text{train}$$.



## Optimization-Based

Deep learning models learn through backpropagation of gradients. However, the gradient-based optimization is neither designed to cope with a small number of training samples, nor to converge within a small number of optimization steps. Is there a way to adjust the optimization algorithm so that the model can be good at learning with a few examples? This is what optimization-based approach meta-learning algorithms intend for.


### LSTM Meta-Learner

The optimization algorithm can be explicitly modeled. [Ravi & Larochelle (2017)](https://openreview.net/pdf?id=rJY0-Kcll) did so and named it "meta-learner", while the original model for handling the task is called "learner". The goal of the meta-learner is to efficiently update the learner's parameters using a small support set so that the learner can adapt to the new task quickly.

Let's denote the learner model as $$M_\theta$$ parameterized by $$\theta$$, the meta-learner as $$R_\Theta$$ with parameters $$\Theta$$, and the loss function $$\mathcal{L}$$.


#### Why LSTM?

The meta-learner is modeled as a LSTM, because:
1. There is similarity between the gradient-based update in backpropagation and the cell-state update in LSTM.
2. Knowing a history of gradients benefits the gradient update; think about how [momentum](http://ruder.io/optimizing-gradient-descent/index.html#momentum) works. 


The update for the learner's parameters at time step t with a learning rate $$\alpha_t$$ is:

$$
\theta_t = \theta_{t-1} - \alpha_t \nabla_{\theta_{t-1}}\mathcal{L}_t
$$


It has the same form as the cell state update in LSTM, if we set forget gate $$f_t=1$$, input gate $$i_t = \alpha_t$$, cell state $$c_t = \theta_t$$, and new cell state $$\tilde{c}_t = -\nabla_{\theta_{t-1}}\mathcal{L}_t$$:

$$
\begin{aligned}
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\\
    &= \theta_{t-1} - \alpha_t\nabla_{\theta_{t-1}}\mathcal{L}_t
\end{aligned}
$$


While fixing $$f_t=1$$ and $$i_t=\alpha_t$$ might not be the optimal, both of them can be learnable and adaptable to different datasets.

$$
\begin{aligned}
f_t &= \sigma(\mathbf{W}_f \cdot [\nabla_{\theta_{t-1}}\mathcal{L}_t, \mathcal{L}_t, \theta_{t-1}, f_{t-1}] + \mathbf{b}_f) & \scriptstyle{\text{; how much to forget the old value of parameters.}}\\
i_t &= \sigma(\mathbf{W}_i \cdot [\nabla_{\theta_{t-1}}\mathcal{L}_t, \mathcal{L}_t, \theta_{t-1}, i_{t-1}] + \mathbf{b}_i) & \scriptstyle{\text{; corresponding to the learning rate at time step t.}}\\
\tilde{\theta}_t &= -\nabla_{\theta_{t-1}}\mathcal{L}_t &\\
\theta_t &= f_t \odot \theta_{t-1} + i_t \odot \tilde{\theta}_t &\\
\end{aligned}
$$


#### Model Setup

![lstm-meta-learner]({{ '/assets/images/lstm-meta-learner.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig.10. How the learner $$M_\theta$$ and the meta-learner $$R_\Theta$$ are trained. (Image source: [original paper](https://openreview.net/pdf?id=rJY0-Kcll) with more annotations)*


The training process mimics what happens during test, since it has been proved to be beneficial in [Matching Networks](#matching-networks). During each training epoch, we first sample a dataset $$\mathcal{D} = (\mathcal{D}_\text{train}, \mathcal{D}_\text{test}) \in \hat{\mathcal{D}}_\text{meta-train}$$ and then sample mini-batches out of $$\mathcal{D}_\text{train}$$ to update $$\theta$$ for $$T$$ rounds. The final state of the learner parameter $$\theta_T$$ is used to train the meta-learner on the test data $$\mathcal{D}_\text{test}$$.


Two implementation details to pay extra attention to:
1. How to compress the parameter space in LSTM meta-learner? As the meta-learner is modeling parameters of another neural network, it would have hundreds of thousands of variables to learn. Following the [idea](https://arxiv.org/abs/1606.04474) of sharing parameters across coordinates, 
2. To simplify the training process, the meta-learner assumes that the loss $$\mathcal{L}_t$$ and the gradient $$\nabla_{\theta_{t-1}} \mathcal{L}_t$$ are independent.


![train-meta-learner]({{ '/assets/images/train-meta-learner.png' | relative_url }})
{: style="width: 100%;" class="center"}


### MAML

**MAML**, short for **Model-Agnostic Meta-Learning** ([Finn, et al. 2017](https://arxiv.org/abs/1703.03400)) is a fairly general optimization algorithm, compatible with any model that learns through gradient descent.

Let's say our model is $$f_\theta$$ with parameters $$\theta$$. Given a task $$\tau_i$$ and its associated dataset $$(\mathcal{D}^{(i)}_\text{train}, \mathcal{D}^{(i)}_\text{test})$$, we can update the model parameters by one or more gradient descent steps (the following example only contains one step):

$$
\theta'_i = \theta - \alpha \nabla_\theta\mathcal{L}^{(0)}_{\tau_i}(f_\theta)
$$

where $$\mathcal{L}^{(0)}$$ is the loss computed using the mini data batch with id (0).


![MAML]({{ '/assets/images/maml.png' | relative_url }})
{: style="width: 45%;" class="center"}
*Fig. 11. Diagram of MAML. (Image source: [original paper](https://arxiv.org/abs/1703.03400))*


Well, the above formula only optimizes for one task. To achieve a good generalization across a variety of tasks, we would like to find the optimal $$\theta^*$$ so that the task-specific fine-tuning is more efficient. Now, we sample a new data batch with id (1) for updating the meta-objective. The loss, denoted as $$\mathcal{L}^{(1)}$$, depends on the mini batch (1). The superscripts in $$\mathcal{L}^{(0)}$$ and $$\mathcal{L}^{(1)}$$ only indicate different data batches, and they refer to the same loss objective for the same task.

$$
\begin{aligned}
\theta^* 
&= \arg\min_\theta \sum_{\tau_i \sim p(\tau)} \mathcal{L}_{\tau_i}^{(1)} (f_{\theta'_i}) = \arg\min_\theta \sum_{\tau_i \sim p(\tau)} \mathcal{L}_{\tau_i}^{(1)} (f_{\theta - \alpha\nabla_\theta \mathcal{L}_{\tau_i}^{(0)}(f_\theta)}) & \\
\theta &\leftarrow \theta - \beta \nabla_{\theta} \sum_{\tau_i \sim p(\tau)} \mathcal{L}_{\tau_i}^{(1)} (f_{\theta - \alpha\nabla_\theta \mathcal{L}_{\tau_i}^{(0)}(f_\theta)}) & \scriptstyle{\text{; updating rule}}
\end{aligned}
$$


![MAML Algorithm]({{ '/assets/images/maml-algo.png' | relative_url }})
{: style="width: 60%;" class="center"}
*Fig. 12. The general form of MAML algorithm. (Image source: [original paper](https://arxiv.org/abs/1703.03400))*


#### First-Order MAML

The meta-optimization step above relies on second derivatives. To make the computation less expensive, a modified version of MAML omits second derivatives, resulting in a simplified and cheaper implementation, known as **First-Order MAML (FOMAML)**.

Let's consider the case of performing $$k$$ inner gradient steps, $$k\geq1$$. Starting with the initial model parameter $$\theta_\text{meta}$$:

$$
\begin{aligned}
\theta_0 &= \theta_\text{meta}\\
\theta_1 &= \theta_0 - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_0)\\
\theta_2 &= \theta_1 - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_1)\\
&\dots\\
\theta_k &= \theta_{k-1} - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_{k-1})
\end{aligned}
$$

Then in the outer loop, we sample a new data batch for updating the meta-objective.

$$
\begin{aligned}
\theta_\text{meta} &\leftarrow \theta_\text{meta} - \beta g_\text{MAML} & \scriptstyle{\text{; update for meta-objective}} \\[2mm]
\text{where } g_\text{MAML}
&= \nabla_{\theta} \mathcal{L}^{(1)}(\theta_k) &\\[2mm]
&= \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot (\nabla_{\theta_{k-1}} \theta_k) \dots (\nabla_{\theta_0} \theta_1) \cdot (\nabla_{\theta} \theta_0) & \scriptstyle{\text{; following the chain rule}} \\
&= \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot \prod_{i=1}^k \nabla_{\theta_{i-1}} \theta_i &  \\
&= \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot \prod_{i=1}^k \nabla_{\theta_{i-1}} (\theta_{i-1} - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_{i-1})) &  \\
&= \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot \prod_{i=1}^k (I - \alpha\nabla_{\theta_{i-1}}(\nabla_\theta\mathcal{L}^{(0)}(\theta_{i-1}))) &
\end{aligned}
$$

The MAML gradient is:

$$
g_\text{MAML} = \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k) \cdot \prod_{i=1}^k (I - \alpha \color{red}{\nabla_{\theta_{i-1}}(\nabla_\theta\mathcal{L}^{(0)}(\theta_{i-1}))})
$$

The First-Order MAML ignores the second derivative part in red. It is simplified as follows, equivalent to the derivative of the last inner gradient update result.

$$
g_\text{FOMAML} = \nabla_{\theta_k} \mathcal{L}^{(1)}(\theta_k)
$$


### Reptile

**Reptile** ([Nichol, Achiam & Schulman, 2018](https://arxiv.org/abs/1803.02999)) is a remarkably simple meta-learning optimization algorithm. It is similar to MAML in many ways, given that both rely on meta-optimization through gradient descent and both are model-agnostic.

The Reptile works by repeatedly:
* 1) sampling a task, 
* 2) training on it by multiple gradient descent steps, 
* 3) and then moving the model weights towards the new parameters. 

See the algorithm below:
$$\text{SGD}(\mathcal{L}_{\tau_i}, \theta, k)$$ performs stochastic gradient update for k steps on the loss $$\mathcal{L}_{\tau_i}$$ starting with initial parameter $$\theta$$ and returns the final parameter vector. The batch version samples multiple tasks instead of one within each iteration. The reptile gradient is defined as $$(\theta - W)/\alpha$$, where $$\alpha$$ is the stepsize used by the SGD operation.


![Reptile Algorithm]({{ '/assets/images/reptile-algo.png' | relative_url }})
{: style="width: 52%;" class="center"}
*Fig. 13. The batched version of Reptile algorithm. (Image source: [original paper](https://arxiv.org/abs/1803.02999))*


At a glance, the algorithm looks a lot like an ordinary SGD. However, because the task-specific optimization can take more than one step. it eventually makes $$\text{SGD}(\mathbb{E}
_\tau[\mathcal{L}_{\tau}], \theta, k)$$ diverge from $$\mathbb{E}_\tau [\text{SGD}(\mathcal{L}_{\tau}, \theta, k)]$$ when k > 1.


#### The Optimization Assumption

Assuming that a task $$\tau \sim p(\tau)$$ has a manifold of optimal network configuration, $$\mathcal{W}_{\tau}^*$$. The model $$f_\theta$$ achieves the best performance for task $$\tau$$ when $$\theta$$ lays on the surface of $$\mathcal{W}_{\tau}^*$$. To find a solution that is good across tasks, we would like to find a parameter close to all the optimal manifolds of all tasks:

$$
\theta^* = \arg\min_\theta \mathbb{E}_{\tau \sim p(\tau)} [\frac{1}{2} \text{dist}(\theta, \mathcal{W}_\tau^*)^2]
$$


![Reptile Algorithm]({{ '/assets/images/reptile-optim.png' | relative_url }})
{: style="width: 50%;" class="center"}
*Fig. 14. The Reptile algorithm updates the parameter alternatively to be closer to the optimal manifolds of different tasks. (Image source: [original paper](https://arxiv.org/abs/1803.02999))*


Let's use the L2 distance as $$\text{dist}(.)$$ and the distance between a point $$\theta$$ and a set $$\mathcal{W}_\tau^*$$ equals to the distance between $$\theta$$ and a point $$W_{\tau}^*(\theta)$$ on the manifold that is closest to $$\theta$$:

$$
\text{dist}(\theta, \mathcal{W}_{\tau}^*) = \text{dist}(\theta, W_{\tau}^*(\theta)) \text{, where }W_{\tau}^*(\theta) = \arg\min_{W\in\mathcal{W}_{\tau}^*} \text{dist}(\theta, W)
$$


The gradient of the squared euclidean distance is:

$$
\begin{aligned}
\nabla_\theta[\frac{1}{2}\text{dist}(\theta, \mathcal{W}_{\tau_i}^*)^2]
&= \nabla_\theta[\frac{1}{2}\text{dist}(\theta, W_{\tau_i}^*(\theta))^2] & \\
&= \nabla_\theta[\frac{1}{2}(\theta - W_{\tau_i}^*(\theta))^2] & \\
&= \theta - W_{\tau_i}^*(\theta) & \scriptstyle{\text{; See notes.}}
\end{aligned}
$$

Notes: According to the Reptile paper, "*the gradient of the squared euclidean distance between a point Θ and a set S is the vector 2(Θ − p), where p is the closest point in S to Θ*". Technically the closest point in S is also a function of Θ, but I'm not sure why the gradient does not need to worry about the derivative of p. (Please feel free to leave me a comment or send me an email about this if you have ideas.)

Thus the update rule for one stochastic gradient step is:

$$
\theta = \theta - \alpha \nabla_\theta[\frac{1}{2} \text{dist}(\theta, \mathcal{W}_{\tau_i}^*)^2] = \theta - \alpha(\theta - W_{\tau_i}^*(\theta)) = (1-\alpha)\theta + \alpha W_{\tau_i}^*(\theta)
$$

The closest point on the optimal task manifold $$W_{\tau_i}^*(\theta)$$ cannot be computed exactly, but Reptile approximates it using $$\text{SGD}(\mathcal{L}_\tau, \theta, k)$$.


#### Reptile vs FOMAML

To demonstrate the deeper connection between Reptile and MAML, let's expand the update formula with an example performing two gradient steps, k=2 in $$\text{SGD}(.)$$. Same as defined [above](#maml), $$\mathcal{L}^{(0)}$$ and $$\mathcal{L}^{(1)}$$ are losses using different mini-batches of data. For ease of reading, we adopt two simplified annotations: $$g^{(i)}_j = \nabla_{\theta} \mathcal{L}^{(i)}(\theta_j)$$ and $$H^{(i)}_j = \nabla^2_{\theta} \mathcal{L}^{(i)}(\theta_j)$$.

$$
\begin{aligned}
\theta_0 &= \theta_\text{meta}\\
\theta_1 &= \theta_0 - \alpha\nabla_\theta\mathcal{L}^{(0)}(\theta_0)= \theta_0 - \alpha g^{(0)}_0 \\
\theta_2 &= \theta_1 - \alpha\nabla_\theta\mathcal{L}^{(1)}(\theta_1) = \theta_0 - \alpha g^{(0)}_0 - \alpha g^{(1)}_1
\end{aligned}
$$

According to the [early section](#first-order-maml), the gradient of FOMAML is the last inner gradient update result. Therefore, when k=1:

$$
\begin{aligned}
g_\text{FOMAML} &= \nabla_{\theta_1} \mathcal{L}^{(1)}(\theta_1) = g^{(1)}_1 \\
g_\text{MAML} &= \nabla_{\theta_1} \mathcal{L}^{(1)}(\theta_1) \cdot (I - \alpha\nabla^2_{\theta} \mathcal{L}^{(0)}(\theta_0)) = g^{(1)}_1 - \alpha H^{(0)}_0 g^{(1)}_1
\end{aligned}
$$

The Reptile gradient is defined as:

$$
g_\text{Reptile} = (\theta_0 - \theta_2) / \alpha = g^{(0)}_0 + g^{(1)}_1
$$


Up to now we have:

![Reptile vs FOMAML]({{ '/assets/images/reptile_vs_FOMAML.png' | relative_url }})
{: style="width: 50%;" class="center"}
*Fig. 15. Reptile versus FOMAML in one loop of meta-optimization. (Image source: [slides](https://www.slideshare.net/YoonhoLee4/on-firstorder-metalearning-algorithms) on Reptile by Yoonho Lee.)*

$$
\begin{aligned}
g_\text{FOMAML} &= g^{(1)}_1 \\
g_\text{MAML} &= g^{(1)}_1 - \alpha H^{(0)}_0 g^{(1)}_1 \\
g_\text{Reptile} &= g^{(0)}_0 + g^{(1)}_1
\end{aligned}
$$


Next let's try further expand $$g^{(1)}_1$$ using [Taylor expansion](https://en.wikipedia.org/wiki/Taylor_series). Recall that Taylor expansion of a function $$f(x)$$ that is differentiable at a number $$a$$ is:

$$
f(x) = f(a) + \frac{f'(a)}{1!}(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \dots = \sum_{i=0}^\infty \frac{f^{(i)}(a)}{i!}(x-a)^i
$$

We can consider $$\nabla_{\theta}\mathcal{L}^{(1)}(.)$$ as a function and $$\theta_0$$ as a value point. The Taylor expansion of $$g_1^{(1)}$$ at the value point $$\theta_0$$ is:

$$
\begin{aligned}
g_1^{(1)} &= \nabla_{\theta}\mathcal{L}^{(1)}(\theta_1) \\
&= \nabla_{\theta}\mathcal{L}^{(1)}(\theta_0) + \nabla^2_\theta\mathcal{L}^{(1)}(\theta_0)(\theta_1 - \theta_0) + \frac{1}{2}\nabla^3_\theta\mathcal{L}^{(1)}(\theta_0)(\theta_1 - \theta_0)^2 + \dots & \\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + \frac{\alpha^2}{2}\nabla^3_\theta\mathcal{L}^{(1)}(\theta_0) (g_0^{(0)})^2 + \dots & \scriptstyle{\text{; because }\theta_1-\theta_0=-\alpha g_0^{(0)}} \\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)
\end{aligned}
$$


Plug in the expanded form of $$g_1^{(1)}$$ into the MAML gradients with one step inner gradient update:

$$
\begin{aligned}
g_\text{FOMAML} &= g^{(1)}_1 = g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)\\
g_\text{MAML} &= g^{(1)}_1 - \alpha H^{(0)}_0 g^{(1)}_1 \\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2) - \alpha H^{(0)}_0 (g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2))\\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} - \alpha H^{(0)}_0 g_0^{(1)} + \alpha^2 \alpha H^{(0)}_0 H^{(1)}_0 g_0^{(0)} + O(\alpha^2)\\
&= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} - \alpha H^{(0)}_0 g_0^{(1)} + O(\alpha^2)
\end{aligned}
$$


The Reptile gradient becomes:

$$
\begin{aligned}
g_\text{Reptile} 
&= g^{(0)}_0 + g^{(1)}_1 \\
&= g^{(0)}_0 + g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)
\end{aligned}
$$


So far we have the formula of three types of gradients:

$$
\begin{aligned}
g_\text{FOMAML} &= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)\\
g_\text{MAML} &= g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} - \alpha H^{(0)}_0 g_0^{(1)} + O(\alpha^2)\\
g_\text{Reptile}  &= g^{(0)}_0 + g_0^{(1)} - \alpha H^{(1)}_0 g_0^{(0)} + O(\alpha^2)
\end{aligned}
$$


During training, we often average over multiple data batches. In our example, the mini batches (0) and (1) are interchangeable since both are drawn at random. The expectation $$\mathbb{E}_{\tau,0,1}$$ is averaged over two data batches, ids (0) and (1), for task $$\tau$$.

Let,
- $$A = \mathbb{E}_{\tau,0,1} [g_0^{(0)}] = \mathbb{E}_{\tau,0,1} [g_0^{(1)}]$$; it is the average gradient of task loss. We expect to improve the model parameter to achieve better task performance by following this direction pointed by $$A$$.
- $$B = \mathbb{E}_{\tau,0,1} [H^{(1)}_0 g_0^{(0)}] = \frac{1}{2}\mathbb{E}_{\tau,0,1} [H^{(1)}_0 g_0^{(0)} + H^{(0)}_0 g_0^{(1)}] = \frac{1}{2}\mathbb{E}_{\tau,0,1} [\nabla_\theta(g^{(0)}_0 g_0^{(1)})]$$; it is the direction (gradient) that increases the inner product of gradients of two different mini batches for the same task. We expect to improve the model parameter to achieve better generalization over different data by following this direction pointed by $$B$$.


To conclude, both MAML and Reptile aim to optimize for the same goal, better task performance (guided by A) and better generalization (guided by B), when the gradient update is approximated by first three leading terms. 

$$
\begin{aligned}
\mathbb{E}_{\tau,1,2}[g_\text{FOMAML}] &= A - \alpha B + O(\alpha^2)\\
\mathbb{E}_{\tau,1,2}[g_\text{MAML}] &= A - 2\alpha B + O(\alpha^2)\\
\mathbb{E}_{\tau,1,2}[g_\text{Reptile}]  &= 2A - \alpha B + O(\alpha^2)
\end{aligned}
$$

It is not clear to me whether the ignored term $$O(\alpha^2)$$ might play a big impact on the parameter learning. But given that FOMAML is able to obtain a similar performance as the full version of MAML, it might be safe to say higher-level derivatives would not be critical during gradient descent update.

---

Cited as:

```
@article{weng2018metalearning,
  title   = "Meta-Learning: Learning to Learn Fast",
  author  = "Weng, Lilian",
  journal = "lilianweng.github.io/lil-log",
  year    = "2018",
  url     = "http://lilianweng.github.io/lil-log/2018/11/29/meta-learning.html"
}
```

*If you notice mistakes and errors in this post, don't hesitate to leave a comment or contact me at [lilian dot wengweng at gmail dot com] and I would be very happy to correct them asap.*

See you in the next post!


## Reference

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


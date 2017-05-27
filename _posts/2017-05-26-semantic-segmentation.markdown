---
layout: post
title: 语义分割笔记
date:   2017-05-26 10:00:42 +0800
categories: 论文


---



## Image Caption为什么需要Semantic Segmentation

一开始的网络只是把CNN的FC层直接输入RNN，但这个层里面的东西是难以解释的，但是RNN这么稀里糊涂的弄一弄就能描述出来图片了。这让人非常没有掌控感，于是后来Google有一篇论文就是讨论输入RNN的东西的可解释性是否对于Image caption有作用，一个很自然的想法是不仅要输入图像中的各项特征，而最好能把图像中的各个物体标注出来，将语义信息输入RNN。结果发现，输入可解释的信息大大提高了神经网络的表现。并不是稀里糊涂的一通训练就可以得到好的效果的。这篇论文非常具有启发性。一个创新之后，对这个创新中的局部进行优化，对局部之间的协作方式进行优化，对创新中说得不清晰或者不合理的部分敢于反思并探索，往往大的提升就在这些模糊的区域中了。接下来是几篇经典论文串讲，从最基础的AlexNet开始。

## ImageNet Classification with Deep Convolutional Neural Networks

这篇论文开创了利用深度卷积神经网络进行图像识别的方法。也就是著名的AlexNet，结构如下图，5个卷积层，3个全连接层：

![044A5076-2F92-4F13-BC6C-B16399096979](/assets/2017-05-26-semantic-segmentation/044A5076-2F92-4F13-BC6C-B16399096979.png)

虽然AlexNet不是CNN的开创者，但他使用了许多技术使得CNN的识别能力大幅提高并已成为现在的标准配置，有

1. ReLU：没有饱和的问题，更快
2. Overlapping Pooling，轻微改善，防止过拟合
3. 多GPU并行, 更快
4. LRN，ReLU后的局部归一化，虽然ReLU对很大的X依然有效，但这样还是能改善一些
5. 减少过拟合：1）数据扩增，各种形态学变化之类的。 2）Dropout，方便好用，记得test的时乘上
6. Weight Decay，感觉实际上就是正则项$$ \lambda$$ 

接下来介绍如何用CNN作语义分割



## Fully Convolutional Networks for Semantic Segmentation

这篇论文最早提出了全卷积网络的概念，想法其实很简单，CNN的输出是一维的向量，如果我们把最后面的FC层全都换成卷积层，就可以输出二维向量了，下图就是AlexNet卷积化后形成的全卷积网络：

![AD9967C8-2C82-411F-9322-FCF3743CF1C4](/assets/2017-05-26-semantic-segmentation/AD9967C8-2C82-411F-9322-FCF3743CF1C4.png)

而且因为FCN与CNN结构非常相似，任务也比较接近，可以利用CNN训练好的网络进行Fine tuning，节省训练时间。而且在计算卷积的时候因为receptive fields重叠的非常多，所以训练很高效（这里不是很懂。。）

但从图中可以看出，这样最终生成的图像是比原来小的，而语义分割需要得到与原图同样大小的图像，那怎么办呢？接着论文提出了upsampling，deconvolution（CS231n里讲这个名字被吐槽的很多，叫conv transpose之类的比较好）的技巧（本质就是插值）。Deconvolution实际上就是将卷积的正向传播和反向传播反过来。反向卷积能否学习对于表现没有明显提升，所以学习率被置零了。但deconvolution又带来了一个问题，就是分辨率的问题，很容易想象出来，好比一张小照片被放大了一样，非常模糊。为了解决这个问题，作者又提出了skip layer的方法，即将前面的卷积层与后面同样大小的反卷积层结合起来。

![1705275F-792B-4BA4-8A47-72B219882490](/assets/2017-05-26-semantic-segmentation/1705275F-792B-4BA4-8A47-72B219882490.png)



## DeepLab: Semantic Image Segmentation withDeep Convolutional Nets, Atrous Convolution,and Fully Connected CRFs

这篇论文提出了使用DCNN实现语义分割的3个主要挑战

1. DCNN降低了特征的分辨率，而且为了保证图片不太小加入了100的padding，引入了噪声
2. 图片上存在着大小不一的物体
3. 图片特征在DCNN中的空间变化不变性导致的细节丢失（局部精确性与分类准确性的矛盾，上一篇论文使用了skip layer来处理这个问题）

#### 第一个问题

作者首先更改了最后两层池化层，把pooling的stride改为1，同时加上1个padding，这样池化后像素的个数就不再改变了。

![766fc04b86b72f7e09d8f8ff6cb648e2_r](/assets/2017-05-26-semantic-segmentation/766fc04b86b72f7e09d8f8ff6cb648e2_r-1.png)

上图的a是原来的池化，b是更改后的池化，c是为了增加感受野带洞的卷积atrous conv。（**==这里池化和卷积分的不太清楚，之后看下代码==**）为什么要带洞呢，是因为b图的感受野是比a要小的，可以看出b图中池化后的连续三个像素对应着池化前的5个，而a图则对应着7个。这会导致全局性的削弱。因此作者收到atrous算法的启发，加上了洞。在扩大分辨率的同时保持了感受野。更改后输出的预测图的大小是原来的4倍，下图直观展示了效果，下图是先将一张图片downsample为1/2，然后分别使用竖向高斯导数卷积核和atrous核，最后再upsampling，高斯核只能得到原图的1/4坐标的预测，而atrous核能得到全部像素的预测![屏幕快照 2017-05-26 上午11.05.02](/assets/2017-05-26-semantic-segmentation/屏幕快照 2017-05-26 上午11.05.02.png)

atrous具体的实现方法有两种，一种是往卷积核里插0，一种是把图片subsample，然后再标准卷积

有一点要说一下，为什么不把池化层直接去掉呢？主要是因为去掉以后网络结构改变，没法使用训练好的网络fine tuning。因为图像识别的数据量比较大，网络训练的比较成熟，所以一般都希望能够借助其训练好的模型。

#### 第二个问题

不同尺寸目标的问题。一个好的解决方案是对于不同尺寸分别做DCNN，但这样太慢了，所以作者用了ASPP，并行的使用多个rate不同的atrous conv，这些卷积核共享参数，所以训练快了很多。如下图所示。

![431DA9CC-7296-4224-B087-6FD7482B8733](/assets/2017-05-26-semantic-segmentation/431DA9CC-7296-4224-B087-6FD7482B8733.png)

#### 第三个问题

局部精确性与分类表现的矛盾问题。作者说有两种解决方法，一种就是利用多层网络中的信息来增强细节，如skip layers；另一种就是使用一些super-pixel（把像素划分成区域）表示，直接去底层获取信息，比如CRF

CRF的一个101http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/（下面写了个3分钟版本的CRF感想，这个以后还有再系统学一下）

简单概括来说，CRF就是对于一个给定的全局观测，许多设定的特征函数，计算一个标签序列在这些特征函数下的得分，然后加权求和求得这个标签序列的得分。再将所有标签序列的得分Softmax归一化，作为该序列的概率。

#### $$  score(l|s) = \Sigma_{j=1}^{m}\Sigma_{i=1}^n\lambda_jf_j(s, i, l_i, l_{i-1}) $$

#### $$  p(l|s) = \frac{exp[score(l|s)]}{\Sigma_{l'} exp[score(l'|s)]} $$

$$ f_j$$ 是特征函数，具体定义由问题决定（比如在词义分析中，可以定义为形容词后面是名词则$$ f$$ 为1，否则为0），$$ l$$ 是一个标签序列，这里的公式针对的是一维的情况，在图像标注中应该改成二维的，$$ l_{i-1}$$ 在二维中对应着$$ i$$ 的邻居节点的标签

要做的事情就是学习$$ \lambda_j$$ 的值，这跟Logistics回归非常像，实际上这就是个时间序列版的logistics回归。一般目标是用最大似然估计来衡量学习。

每一个HMM（隐马尔科夫模型）都等价于一个CRF，就是说CRF比HMM更强。对HMM模型取对数之后吧概率对数看做权值，即化为CRF。这是因为CRF的特征函数具有更强的自由性，可以根据全局来定义特征函数，而HMM自身带有局部性，限制了其相应的特征函数。而且CRF可以使用任意权重，而HMM只能使用对数概率作为权重。

在这篇论文中，优化的目标是使下面这个函数最小

#### $$ E(x) = \Sigma_i\theta_i(x_i)+\Sigma_{ij}\theta_{ij}(x_i, x_j)$$

$$ x_i$$ 是第$$ i$$ 个像素的标签，$$ \theta_i(x_i) = -log P(x_i)$$ ，$$ P(x_i)$$ 是第i个像素贴上$$ x_i$$ 这个标签的概率（由DCNN算出来的），$$ \theta_{ij}(x_i, x_j)$$ 是像素$$ i$$ 像素$$ j$$ 之间关系的度量

#### $$ \theta_{ij}(x_i, x_j) = \mu(x_i, x_j)[w_1\ exp(-\frac{||p_i-p_j||^2}{2\sigma_{\alpha}^2}-\frac{||I_i-I_j||^2}{2\sigma_{\beta}^2}) \\\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ + w_2\ exp(-\frac{||p_i-p_j||^2}{2\sigma_{\gamma}^2})]$$

$$ \mu$$ 在$$ x_i, x_j$$ 相等的时候是0，不相等时是1（只会惩罚相同标签的像素），这就是个双边滤波……

#### 实验中有启发的几个点

1. learning rate使用poly策略比较好

2. batch size小一点（最后取了10），迭代次数多一点更有利于训练

3. 在PASCAL-Person-Part上训练的时候LargeFOV和ASPP对于训练效果都没有提升，但CRF的提升效果非常明显。技术有适用性吧，No free lunch theory.

4. 从结果上来看CRF好像做了一些平滑和去噪的工作。（**==对于CRF理解还不太到位，之前感觉像是起到精细化的作用，这里主要是双边滤波在起作用？==**）
   ![D93EE4D0-6893-45C5-B6D6-65E608C01E7B](/assets/2017-05-26-semantic-segmentation/D93EE4D0-6893-45C5-B6D6-65E608C01E7B.png)

   ![270A5970-D3CE-4C4D-86C0-86BC9E526AA3](/assets/2017-05-26-semantic-segmentation/270A5970-D3CE-4C4D-86C0-86BC9E526AA3.png)
   ![E19801B2-2B12-443A-BB57-C9786F9AFF16](/assets/2017-05-26-semantic-segmentation/E19801B2-2B12-443A-BB57-C9786F9AFF16.png)

5. Cityscapes的图片非常大，作者一开始先缩小了一半再训练的，但后来发现用原始大小的图片训练能提高1.9%，效果很明显（但我感觉缩小一半对于细节的损失并不是很大因为原始图片有2048*1024，可能是因为训练量上升了？）。作者的处理方法是把原始图片分割成几张有重叠区域的图片再训练，训练好了拼起来。

6. Failure Modes，作者发现他们的模型难以抓住复杂的边界，如下图，甚至可能会被CRF完全抹掉，因为因为DCNN算出来的东西不够自信（零星、稀疏）。作者看好encoder-decoder结构能够解决这个问题
   ![E7E2AC8F-283B-4374-B92F-2928E8E0C857](/assets/2017-05-26-semantic-segmentation/E7E2AC8F-283B-4374-B92F-2928E8E0C857.png)

## Faster R-CNN:Towards Real-Time Object Detection with Region Proposal Networks

目标检测近来的发展得益于region proposal methods和region-based convolutional nueral networks的成功。RPM负责给出粗略的语义分割，而R-CNN负责精细化的检测。Fast R-CNN已经得到了几乎实时的运行时间，而现在瓶颈就在于计算RPM，本文的目标就是使用RPN来突破该瓶颈，达到实时目标检测。这篇论文提出了RPN代替了常用的Region proposal methods,负责给出粗略的语义分割。

主要的原理是共享卷积层。作者们发现region-based detectors（比如Fast R-CNN）使用的卷积层产生的特征，也可以用来生成region proposals。

#### RPN的构建

为了共享卷积，作者考察了ZF model（5层共享卷积）和SZ model（VGG，13层共享卷积层）

为了生成region proposals，作者在最后一个共享卷积层输出的特征层上做slide window。把一个window里的通过一个全连接层，生成一个低维向量。这个向量接着再被喂进两个平行的全连接层，分别用于矩形定位和矩形分类打分。

![E31B36A8-B635-46F8-A51D-F7154C75DDC8](/assets/2017-05-26-semantic-segmentation/E31B36A8-B635-46F8-A51D-F7154C75DDC8.png)

实际上这个slide window就是个卷积，后面的两层也是卷积层。对于每个window会提出k个region proposal。作者说这个方法有个很重要的属性是translation invariant（平移不变性，平移后仍能预测出相同大小的anchor boxes）。

#### RPN的学习过程

有着最大IOU或与所有goud-truth box 的IOU都大于70%的anchor会被赋予正标签；

与所有ground-truth box的IOU都小于30%的anchor会被赋予负例；

其他的anchor不会对训练有贡献。Loss function如下

#### $$ L(p_i, t_i) = \frac1{N_{cls}}\Sigma_iL_{cls}(p_i, p_i^*) + \lambda \frac1{N_{reg}} \Sigma_i p_i^* L_{reg}(t_i, t_i^*)$$

i是anchor的index，$$ p_i$$ 是anchor i被预测为是一个物体的概率，$$ p^*_i$$ 是ground-truth（如果anchor是positive则为1，否则为0），$$ t_i$$ 是表示box四个坐标的参数向量，$$ t^*_I$$ 是ground-truth。$$ L_{cls}$$ 是log loss（Softmax分类器)，$$ L_{reg}(t_i, t_i^*)=R(t_i - t_i^*)$$ ，$$ R$$ 是robust loss function(smooth L1) (==**这是啥**==)。因为$$ p^*_i$$ ，第二项只有正例的时候才会起作用。$$ \lambda$$ 是一个平衡系数。

#### 优化

每个mini-batch都来自于同一张图片，随机取128个正例和128个负例，如果不够128个正例，就用负例填上

#### 共享卷积特征

共享卷积特征存在这一个困难，Fast R-CNN的训练是基于固定的region proposals的，所以没法直接训练联合模型。而且不知道联合训练是否能让共享卷积层收敛。所以作者提出了按如下步骤训练的方法。

1. 训练RPN，用ImageNet预训练的模型fine-tune。
2. 训练Fast R-CNN，使用第1步中RPN生成的proposals，到现在为止没有共享卷积层
3. 用Fast R-CNN初始化RPN的训练，但是只fine-tune RPN自己的层，不更改共享的卷积层
4. fine-tune Fast R-CNN的fc层，也不更改共享的卷积层

也就是说卷积层没被联合训练过。

#### 实现细节

许多RPN proposals高度重叠，作者使用了名为non-maximum suppression（NMS）的技术，NMS大大降低了proposal的数量而没有损害检测精度。

#### 实验

下面是几种技术对于结果的影响的比较

![屏幕快照 2017-05-26 上午11.14.58](/assets/2017-05-26-semantic-segmentation/屏幕快照 2017-05-26 上午11.14.58.png)



### Deformable Convolutional Networks

视觉识别中一个很大的问题在于图像的变形（角度、大小、姿势等）。以往的训练都是通过增加数据来使网络熟悉各种变形或使用一些形变时不变的特征（像是SIFT, scale invariant feature transform）。这篇论文提出CNN需要专门针对变形的结构才能较好的解决这个问题，因此提出了deformable convolution。

#### Deformable Convolution

基本思想是改变卷积层的核，原来核是一个方形，现在对于核中每个元素加上一个offset，卷积后的特征不再来源于一个方形，而可能来源于各种形状。

![屏幕快照 2017-05-26 下午1.20.54](/assets/2017-05-26-semantic-segmentation/屏幕快照 2017-05-26 下午1.20.54.png)

原来的卷积公式是这样子：

#### $$ y(p_0) = \Sigma_{p_n} w(p_n) \cdot x(p_0+p_n)$$

加上偏移量$$ \Delta p_n$$ 后变成这个样子：

#### $$ y(p_0) = \Sigma_{p_n} w(p_n) \cdot x(p_0+p_n+ \Delta p_n)$$

但因为偏移量常常是小数，所以要用双线性插值找到偏移后的坐标最接近的整数位置，公式略。

#### Deformable RoI Pooling

RoI pooling是将一个任意大小的图片转化为固定大小输出的池化。池化函数是bin内的平均值。原始公式是

#### $$ y(i,j) = \Sigma_{p} x(p_0 + p) / n_{ij}$$

$$ p_0$$ 是bin的左上角，p是枚举位置，$$ n_{ij}$$ 是bin内的元素总数， 加上偏移量后

#### $$ y(i,j) = \Sigma_{p} x(p_0 + p +\Delta p_{ij}) / n_{ij}$$

偏移量的学习学习的是相对系数（图片大小的百分比），这样能够适用于不同大小的图片。

还可以扩展到position-sensitive RoI pooling，**==（这里不太清楚，以后再看）==**



#### 理解Deformable ConvNets

下图是使用了的deformable conv后感受野的变化

![屏幕快照 2017-05-26 下午8.27.21](/assets/2017-05-26-semantic-segmentation/屏幕快照 2017-05-26 下午8.27.21.png)

而且因为核具有自己调整的特性，可以轻松识别出不同scale的物体，下图展示了这一特性，每张图片中的红点是三层卷积对应的感受野，绿点是最高层的中心

![屏幕快照 2017-05-26 下午8.30.40](/assets/2017-05-26-semantic-segmentation/屏幕快照 2017-05-26 下午8.30.40.png)

对于RoI也是类似的效果，黄框的分数是由红框的平均值计算来的

![屏幕快照 2017-05-26 下午8.30.49](/assets/2017-05-26-semantic-segmentation/屏幕快照 2017-05-26 下午8.30.49.png)

#### 与相关工作的对比

有几个有趣的点

1. Effective Receptive Field这里提到，感受野虽然理论上随着层数线性增长，但实际上是成根号增长的，比预期的慢很多，因此即使是顶层的单元感受野也很小。因此Atrous Conv由于其有效增加感受野得到了广泛的应用
2. 之前也有动态filter的研究，但都只是值的变化而不是位置的变化
3. 当多层卷积结合起来以后，可能会有着跟deformable conv类似的效果，但存在着本质上的不同。经过复杂学习后得到的东西如果换一种思考方式就变得意外简单。

#### 实验

几种网络应用了deformable conv后的效果比较：

![屏幕快照 2017-05-26 下午8.59.34](/assets/2017-05-26-semantic-segmentation/屏幕快照 2017-05-26 下午8.59.34.png)

不知道为什么Faster R-CNN的提升效果最差（可能是RPN），而DeepLab应用6层deformable conv后效果反而变差了（猜测是感受野过大，容易分散，或在某些特征点收敛，过于集中，太关注于局部信息）

#### Aligned-Inception-ResNet

**==这个网络还需要学习一下==**

#### 感想

感觉这篇论文的想法非常秒，很优雅。在知乎上看到一句话，ALAN Huang说的，感觉非常有启发性

> conv，pooling这种操作，其实可以分成三阶段： indexing（im2col） ，reduce(sum), reindexing（col2im). 在每一阶段都可以做一些事情。 用data driven的方式去学每一阶段的参数，也是近些年的主流方向。

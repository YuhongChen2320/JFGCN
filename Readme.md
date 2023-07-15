Joint Learning of Feature and Topology for Multi-view Graph Convolutional Network 

(JFGCN in Pytorch)

##  Usage

```python train.py```

## Method framework

![image-20230715194946257](C:\Users\11833\AppData\Roaming\Typora\typora-user-images\image-20230715194946257.png)

We design a multi-view autoencoder to approximate matrix decomposition, which integrates the consistency of multi-view data. Simultaneously, the k-Nearest Neighbor (kNN) and k-Farthest Neighbor (kFN) strategies are utilized to calculate a more accurate set of topology matrices from two perspectives. Then JFGCN dynamically adjust it by using a flexible graph convolution to learn a robust connective pattern.

## Requirements

* Python 3.9
* Pytorch 1.12.1


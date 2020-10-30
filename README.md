# Revisiting Light Field Rendering with Deep Anti-Aliasing Neural Network
### [Project Page](https://gaochangwu.github.io/) | [Paper](https://gaochangwu.github.io/)

[Gaochang Wu](https://gaochangwu.github.io/)\*<sup>1</sup>,
[Yebin Liu](http://www.liuyebin.com/)\*<sup>2</sup>,
[Lu Fang](http://luvision.net/)\*<sup>3</sup>,
[Tianyou Chai](http://www.sapi.neu.edu.cn/)<sup>1</sup><br>

<sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego <br>
<sup>*</sup>denotes equal contribution


## Abstract
![Teaser Image](https://user-images.githubusercontent.com/3310961/84946597-cdf59800-b09d-11ea-8f0a-e8aaeee77829.png)

We show that passing input points through a simple Fourier feature mapping enables a multilayer perceptron (MLP) to learn high-frequency functions in low-dimensional problem domains. These results shed light on recent advances in computer vision and graphics that achieve state-of-the-art results by using MLPs to represent complex 3D objects and scenes. Using tools from the neural tangent kernel (NTK) literature, we show that a standard MLP fails to learn high frequencies both in theory and in practice. To overcome this spectral bias, we use a Fourier feature mapping to transform the effective NTK into a stationary kernel with a tunable bandwidth. We suggest an approach for selecting problem-specific Fourier features that greatly improves the performance of MLPs for low-dimensional regression tasks relevant to the computer vision and graphics communities.

## Code
We provide a [demo IPython notebook](https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb) as a simple reference for the core idea. The scripts used to generate the paper plots and tables are located in the [Experiments](https://github.com/tancik/fourier-feature-networks/tree/master/Experiments) directory.

# Revisiting Light Field Rendering with Deep Anti-Aliasing Neural Network
### [Project Page](https://gaochangwu.github.io/) | [Paper](https://gaochangwu.github.io/)

[Gaochang Wu](https://gaochangwu.github.io/)<sup>1</sup>,
[Yebin Liu](http://www.liuyebin.com/)\*<sup>2</sup>,
[Lu Fang](http://luvision.net/)\*<sup>3</sup>,
[Tianyou Chai](http://www.sapi.neu.edu.cn/)<sup>1</sup><br>

<sup>1</sup>State Key Laboratory of Synthetical Automation for Process Industries, Northeastern University <br> 
<sup>2</sup>Department of Automation, Tsinghua University <br>
<sup>3</sup>Tsinghua-Berkeley Shenzhen Institute <br>
<sup>*</sup>denotes corresponding author


## Abstract
![Teaser Image](https://github.com/GaochangWu/DA2N/blob/main/imgs/FA.png)

The light field (LF) reconstruction is mainly confronted with two challenges, large disparity and non-Lambertian effect. Typical approaches either address the large disparity challenge using depth estimation followed by view synthesis or eschew explicit depth information to enable non-Lambertian rendering, but rarely solve both challenges in a unified framework. In this paper, we revisit the classic LF rendering framework to address both challenges by incorporating it with advanced deep learning techniques. First, we analytically show that the essential issue behind the large disparity and non-Lambertian challenges is the aliasing problem. Classic LF rendering approaches typically mitigate the aliasing with a reconstruction filter in the Fourier domain, which is, however, intractable to implement within a deep learning pipeline. Instead, we introduce an alternative framework to perform anti-aliasing reconstruction in the image domain and prove in theory the comparable and even superior efficacy on the aliasing issue. To explore the full potential, we then embed the anti-aliasing framework into a deep neural network through the design of an integrated architecture and trainable parameters. The network is trained through end-to-end optimization using a peculiar training set, including regular LFs and unstructured LFs. The proposed deep learning pipeline shows a substantial superiority on solving both the large disparity and the non-Lambertian challenges compared with other state-of-the-art approaches. In addition to the view interpolation for a LF, we also show that the proposed pipeline also benefits light field view extrapolation.

## Results
![Teaser Image](https://github.com/GaochangWu/DA2N/blob/main/imgs/Result1.png) <br>
Comparison of the results (x16 upsampling) on the LFs from the ICME DSLF dataset [34] <br>
![Teaser Image](https://github.com/GaochangWu/DA2N/blob/main/imgs/Result2.png) <br>
Compar ison of the results (x16 upsampling) on the LFs from the MPI Light Field Archive [38] <br>

## Note for Code
1. The code for 3D light field (1D angular and 2D spatial) reconstruction is "main3d.py". Recommend using the model with upsampling scale \alpha_s=3 for x8 or x9 reconstruction, and the model with upsampling scale \alpha_s=4 for x16 reconstruction. <br>

2. The code for 4D light field reconstruction is "batch4d.py". <br>

3. Please cite our paper if it helps, thank you! <br>


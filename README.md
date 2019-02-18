# FCSS (Fully Convolutional Self-Similarity) Descriptor

> Version 1.0 (14 Aug. 2017)
>
> Contributed by Seungryong Kim (srkim89@yonsei.ac.kr).

This code is written in MATLAB, and implements the FCSS descriptor [[website](http://diml.yonsei.ac.kr/~srkim/FCSS/)]. 

## Dependencies
  - Download [VLFeat] (http://www.vlfeat.org/) and [MatConvNet] (http://www.vlfeat.org/matconvnet/).
  - Download the datasets:
    - [Taniai Benchmark] (http://taniai.space/projects/cvpr16_dccs/);
    - [Proposal Flow Benchmark] (http://www.di.ens.fr/willow/research/proposalflow/);
    - [Pascal-VOC Part Dataset] (https://people.eecs.berkeley.edu/~tinghuiz/projects/flowWeb/).

### Getting started ###
  - `main_FCSS_test.m` shows how to compute dense flow fields using the pretrained FCSS descriptor (`data/fcss/net-epoch.mat`) with SIFT Flow [1] and Proposal Flow [2] optimization.
  - `main_FCSS_train_Tatsunori.m`  shows how to train a new model.
  - `get_train_Tatsunori.m`: prepares the filenames of training samples.

# Main functions 
  - `getBatch_Tatsunori.m`: prepares the images of training samples.
  - `init_FCSS.m`: builds an initial model of FCSS descriptor.
  - `CSSlayer.m`: builds convolutional self-similarity (CSS) layers using a bilinear sampler similar to spatial transformer networks (STNs) [3].
  - `CSSlayer_shift.m`: builds convolutional self-similarity (CSS) using Taylor expansion.
  - `CorrespondenceLoss.m`: builds a weakly-supervised correspondence loss for FCSS descriptor.
  
# Notes

  - The code is provided for academic use only. Use of the code in any commercial or industrial related activities is prohibited. 
  - If you use our code, please cite the paper. 

```
@InProceedings{kim2017,
author = {Seungryong Kim and Dongbo Min and Bumsub Ham and Sangryul Jeon and Stephen Lin and Kwanghoon Sohn},
title = {FCSS: Fully Convolutional Self-Similarity for Dense Semantic Correspondence},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE},
year = {2017}
}
```

# References

[1] C. Liu, J. Yuen, and A. Torralba, "Sift flow: Dense correspondence across scenes and its applications", IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 33(5), pp. 815-830, 2011.

[2] B. Ham, M. Cho, C. Schmid, and J. Ponce, "Proposal flow: Semantic correspondences from object proposals", IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 2017.

[3] M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu, "Spatial transformer networks", Neural Information Processing Systems (NIPS), 2015.

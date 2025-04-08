# Weed Growth Estimation
Group 13
Maosheng Jiang, Moeghiez Bhatti, Artin Sanaye

## Introduction

With the population growing faster then ever [[1](#1)], weed management is becoming a critical issue in modern agriculture. Unregulated weed growth can severely reduce crop yields, which leads to food shortages and rising food prices. Small weeds growing alonside crops are generally less harmful that fully grown weeds, while fully grown weeds significantly impact the crop growth by competing with crops for essential soil nutrients. Monitoring weed growth is thus essential for timely removal in order to protect the growth of the desired crops.

Conventional methods require frequent manual inspection of soils and lands in order to early identitfy when weeds will impact crop growth. Such manual inspection is both time-intensive and expensive. Another common approach for weed prevention is to utilize pesticides, such as herbicides. This is an effective tool for the prevention of weed growth, however it has a negative impact on the environment and health of people. The major downside of utilizing herbicides for weed prevention is the adverse effect on the environment and people [[2](#2)].

The rise of emerging Deep Learning techniques has opened a wide range of possibilities in many areas of interest. One of these areas is in the agriculture [[3](#3), [6](#6)]. However, the area of estimating the growth of weed species is not widely explored yet using modern deep learning techniques. The estimation of the growth of weed species is an interesting topic, as the ability to see how the weed evolves enables us to get insights for effective weed management such as expected weed volume, grow direction and weed size estimation.

### Problem statement
In this blog, we aim to explore the effectiveness of a simple transfomer based model to estimate the growth of weed species through segmentation masks, based on sequences of temporal past weed mask images, where the goal is to estimate the growth of weeds by computing a mask that represents the weed in the future. The main motivation for the exploration of utilizing the transformer architecture is to introduce a novel method for growth estimation to bridge the gap between labor- and time-intensive manual inspection and environment-damaging pesticides. 

### Research questions
To tackle this problem, we will answer the following research questions: 
- How does the accuracy of transformer based archictures compare to the UNet model for weed growth estimation on multiple temporal past images?
- What is the effect on accuracy of models that incorporate sequences of prior images against models that perform single-image estimation?
- What is the effect of trying to predict masks of weed plants furhter into the future?

### Contributions
Our work has three major contributions in the field of weed growth prediction. First, we show the accuracy impact for weed growth estimation by taking into account past images of the weed. Second, we show the impact on accuracy for different temporal estimation settings, i.e. what is the effect of estimating growth further into the future. Lastly, we lay a foundation for intersted researchers for further research into weed growth estimation for the agricultural industry.

Furthermore, our github repository [[13](#13)] is made open-source in order to provide reproducibility and encourage further research into this topic.

## Related work
Previous work in weed analysis through Deep Learning has focussed on the weed detection in both early stages and mature stages of growth [[3](#3), [5](#5)]. The goal of weed detection is to detect and identify weed based on some form of input, typically a RGB image which contains the weed growing on soil. Weed detection using deep learning techniques has shown accurate precision in the ability to detect weeds [[5](#5)]. However, the problem with weed detection is the inability to see how the plants progresses visually in futures stages of growth.

Another past research on weed growth prediction has investigated weed growth estimation using segmentation masks [[7](#7)], however, the authors' approach involves an end-to-end segmentation with an EfficientNet backbone. They estimate the growth of weed species using single images, but do not leverage temporal priors to perform the growth estimation for future weed images. Temperoal priors could encode specific hints for the plant growth, such past growth direction and volume increase. Our work tries to bridge this gap by taking into account past images to explore whether this method estimates future growth of weed more accurately.

To the best of our knowledge, previous literature did not attempted to explore the problem of growth estimation by taking into account multiple previous temporal images in order to estimate the segmentation mask of the weed in future timestamps. 

## Background
In this section, we explain briefly the underlying background information for this work, we will briefly cover some background information.

### Transformers
After the introduction of the Transformer architecture by Vaswani et al. [[8](#8)], the deep learning community quickly discovered that the transformer architecture could be applied in other fields of Deep Learning. It was originally introduced for neural machine translation, in fields such as computer vision, the transformer encoder-only models performed suprisingly well for computer vision tasks [[9]#9, [10]#10].

We have chosen to use the transformers-encoder model in our architecture due to the versatile attention mechanism. Due to the self attention mechanism, it was very promising to utilize the attention to reason about previous weed images in the past to estimate the growth of future weed images.

### Segmentation
Another important work that we built upon is the UNet model introduced by Ronneberger et al [[11](#11)]. Originally made for biomedical image segmentation, it also shows great performances for other fields, such as depth estimation [[12](#12)]. Since we want to estimate the growth of weed in the future by providing a segmentation mask of the possible future weed, we can leverage the simplicity of the UNet model.


## Methodology

In this section, we will outline the method used in our project for estimating the growth of weed.

### Defining growth
The problem of estimating growth prediction could be tackled in different ways. One way of defining growth is for example measuring the bounding boxes of the plants. However, due to the random nature of growths of plants in general, we opted for utilizing masks of the plants instead. Using this method, we are able to capture finer details about the plant itself, such as more precise plant area and shape.

### Dataset
We have utilized the Moving Fields Weed Dataset (MFWD) by Genze et al. [[14](#14)] in our experiments. This dataset contains 28 weed species commonly found in sorghum and maize fields in Germany. It contains 94,321 images with high spatial and temporal resolution. What stands out is that the dataset contains multiple images of a specific weed species captured through time, which is a requirement for our model since it needs past images to estimate the future image. For the experiments of estimating weed growth, we have focussed on estimating growth of only one weed species: ARTVU. 

The MFWD dataset contains 2847 high-resolution images, where all images are sequentially taken and timestamped. The interval of the taken images are 12 hours on average, which means roughly two images per day. In this work, we will try to estimate the future growth of images based on this timestamp, i.e. if we want to predict 4 timestamps in the future, then this means that we predict roughly 2 days into the future.

### Obtaining masks
The majority part of the MFWD dataset contains high-resolution RGB images, and only a few hand-crafted segmentation masks. Due to the tedious process of manually creating segmentation masks, we leveraged a key property of the RGB images to automatically extract binary masks from the RGB images. Because the images are all taken in trays of brown soil with green weed on top, we can distinguish the weed by thresholding the images in the HSV color space.

After manually tuning the threshold, we find that the following threshold values work the best:

Table 1: Our fine tuned HSV thresholds for automatic binary mask generation.
| |Hue| Saturation | Value |
|-------- | -------- | -------- | -------- |
|Min| 30 | 87 | 86 |
|Max| 50 | 255 | 255 |

### Models
In this work, we propose a transformer based model which takes into account past binary masks of weed to estimate future binary masks of weed. To compare our model against other simpler baselines: a simplified UNet model that performs segmentation based on a single image, and a simplified UNet model that performs segmentation based on past images.

Our first model is our proposed model for the problem of weed growth estimation, the transformer based model that takes 4 past masks of weed images as input, which we call UNet_ViT. This model as a skip connection from the last input image to the output, since the last image most likely contains the most useful information for the estimation of the growth.

The architecture of the UNet_ViT is shown in the image below:
![UNet_ViT Modle](https://i.imgur.com/0FiYkOr.png =500x)


Secondly we have implemented a simplified UNet model that also has 4 past masks of weed images as input, which we call UNet_4.

The architecture of the UNet_4 is shown in the image below:
![UNet_4](https://i.imgur.com/t7cvKOM.png =350x)

Lastly, we have the exact same model as UNet_4, with the only difference in that the input consists of a single image, i.e. it estimates the growth based on a single image. We have named this model UNet_1.

The architecture of the UNet_1 is shown in the image below:
![UNet_1](https://i.imgur.com/QN3Whiu.png =350x)

In the table below, we have listed a table of the characteristics of the models:

Table 2: Parameter count and weight size of each model architecture.
|Model| #Params | Weight size |
|-------- | -------- | -------- |
|UNet_ViT| 471,874,816 | 1800.06 MB |
|UNet_4| 483,008 | 1.84 MB |
|UNet_1| 482,576 | 1.84 MB |

### Training
The training process is straightforward and consists of a few important key aspects. First, we create a training and validation dataset which consists of the binary mask images from the MFWD dataset. The training dataset consists of 4 binary masks of the weed ARTVU, and the label is a single binary masks, representing the mask of the plant in the future.

However, since we also have the UNet_1 model which performs single image prediction, we have created corresponding train- and validation-datasets for the UNet model that only takes in a single binary mask, rather than 4.

Because the models produce binary masks only, we utilize the Jaccard Index or Intersection over Union (IoU) as our loss function for training.

<!-- Below we have an overview of the datasets:
|Dataset| 1_4 |  |
|-------- | -------- | -------- |
|Train_instances| 0.5947   | TODO   |
|Validation_instances| 0.5690   | 0.4197   |
 -->


### Evaluation
Since our outputs are binary masks representing the estimated weed growth, we utilize the accuracy given by the IoU metric to evaluate our prediction against the ground truth binary masks.

## Experimental Setup
In this section, we will provide a brief summary of the implementation details of the models.
### Implementation details
Our UNet_ViT variant contains a simple UNet-encoder, which converts the binary masks into a 768 dimensional embedding. This will be fed into a pre-trained vision transformer,  `vit_base_patch16_224` from HuggingFace. The output of the classifier embedding will then be processing through a UNet decoder to construct the estimated binary mask.

The UNet_1 and UNet_4 variants are implemented as simplified versions of a classical UNet-architecture introduced by the original paper. The simplifications consists of reducing the parameters count by reducing the feature map channels and reducing the amount of convolutions operations.
### Training details

The training settings of all three models are the same. Our optimizer is AdamW, with a learning rate of 0.001. Each model is trained for 30 epochs.

The models are all trained on a NVIDIA RTX3060 GPU with 6Gb VRAM. Each epoch for the UNet_ViT took 25 minutes to complete and each epoch for the UNet_1 and UNet_2 took 15 seconds to complete.


## Results
Our experiment results are shown in the table below:

Table 3: Results of the experiments.
|Model| acc. dt = 1 | acc. dt = 4 |
|-------- | -------- | -------- |
|UNet_ViT| 0.5947   | 0.4078   |
|UNet_4| 0.5690   | 0.4197   |
|UNet_1| 0.6029   | 0.4586   |

Here, `acc.` stands for the accuracy on the unseen validation dataset of the corresponding model using the IoU metric. The higher the accuracy, the better. Furthermore, `dt = k` signifies the delta time into the future that we want to predict. For example, `dt = 4` means that we are trying to estimate 4 timesteps into the future, which in this case means 2 days (4 x 12hrs).

We visualized a sample input from the validation dataset, where the UNet_ViT model predicts 4 future timesteps (approximately 2 days) based on images from 4 previous timesteps (equivalent to 2 days of past data):
![Results visualized](https://i.imgur.com/SZkbNzA.png)
In the top row, the first image is the binary mask of the 4th input image. To recap, the UNet_ViT expects 4 images as input, where this one is the last one. The middle binary mask is the estimated mask. The right image is the estimated mask with the 4th input binary image subtracted, essentially displaying the growth.

In the bottom row, the left image is the RGB input corresponding to the 4th input binary mask. The middle image is showing the 4th RGB input image with the Growth Mask overlayed on top, showcasing the growth in red. The right image is the ground truth RGB image of the actual growth.

## Discussion
According to the results, the UNet_1 model surprisingly performed the best among the three models with 0.6029 for the prediction of 1 timestamp ahead and 0.4586 for predicting 4 timestamps ahead. We expected that the UNet_1 model would perform the worst since it only has the information of a single binary mask, while the other two models have 3 additional images for reasoning. A possible cause for the lower performance of UNet_4 and UNet_ViT could lie in the natural growth of weed and also plants in general. New weed may emerge at random and this could confuse the models in the prediction of the future growth, where as the UNet_1 only has to reason about a single image. The emergence of new weed can be seen clearly in UNet_ViT architecture figure.

If we compare the models with each other which takes 4 past images into account, i.e. the UNet_4 and UNet_ViT models, then we can see that UNet_ViT performs better on the prediction of 1 timestamp in the future with an accuracy of 0.5947. However UNet_4 has a marginally better accuracy for the prediction of 4 timestamps ahead with an accuracy of 0.4197. A likely cause for this discrepancy is the difference in the architecture size. As can be seen in table 2, the UNet_ViT contains much more parameters compared to UNet_4. This suggests that UNet_ViT requires much more training data and more training rounds for better accuracy in tasks to predict weed growth further into the future. Due to time and resource constraints, we have trained each model for 30 epochs, which may be insufficient for UNet_ViT.


## Limitations

The biggest limitation is the amount of training data. Our training dataset for the ARTVU weed species has roughly 2000 training samples, which may be enough for simple UNet models since they do not contain a lot of trainable parameters and thus do not require many training samples. As for the transformer based model, 2000 training samples are likely to be insufficient, given that the transformer based model in our research had 3 orders of magnitude more trainable parameters compared to the UNet models.

Furthermore, the MWFD dataset could also be enhanced. Currently, the MWFD dataset contains images which are not taken at precise regular intervals, instead they are taken roughly on the same interval with differences in the range of hours. This inregular imaging introduces imprecise predictions as the models will fail to learn the temporal correlations of the dataset.

Lastly, the images of the MWFD dataset contains random weed emergences which are not removed from the growing trays. This causes an additional variable not related to image segmentation and could decrease model accuracy. For the task of our project, it is best to focus only on the growth of weed, instead of where and when new weeds will emerge in the images.

## Conclusion and Future Work
In this blog, we have investigated how the transformer architecture performed for the task of weed growth estimation by taking into account past images. Our results show promising potential of transformer-based models, especially in estimating weed growth close to the present. However, for prediction of further into the future, our results show that transformer-based models may not be the best choise. However, we also found that a simple UNet architecture which performs single image prediction outperforms both models which had the advantages of 3 extra past images.

This project is a preliminary investigation of utilizing the versatile transformer architecture for weed growth estimation. Future work could build upon this work for more extensive research in this area, such as creating more training data, finetuning the architecture, and more training rounds for better convergence and accuracy. We hope that we have set a starting point for intersted researchers to enter this exciting area of weed growth estimation.

## References
<a id="1">[1]</a>  Hannah Ritchie, Lucas Rodés-Guirao, Edouard Mathieu, Marcel Gerber, Esteban Ortiz-Ospina, Joe Hasell and Max Roser (2023) - “Population Growth” Published online at OurWorldinData.org. Retrieved from: 'https://ourworldindata.org/population-growth' [Online Resource]

<a id="2">[2]</a> Mohd Ghazi R, Nik Yusoff NR, Abdul Halim NS, Wahab IRA, Ab Latif N, Hasmoni SH, Ahmad Zaini MA, Zakaria ZA. Health effects of herbicides and its current removal strategies. Bioengineered. 2023 Dec;14(1):2259526. doi: 10.1080/21655979.2023.2259526. Epub 2023 Sep 25. PMID: 37747278; PMCID: PMC10761135.

<a id="3">[3]</a> Yangkai Zhang, Mengke Wang, Danlei Zhao, Chunye Liu, Zhengguang Liu,Early weed identification based on deep learning: A review,Smart Agricultural Technology, Volume 3, 2023, 100123, ISSN 2772-3755, https://doi.org/10.1016/j.atech.2022.100123.

<a id="4">[4]</a> Genze, N., Vahl, W.K., Groth, J. et al. Manually annotated and curated Dataset of diverse Weed Species in Maize and Sorghum for Computer Vision. Sci Data 11, 109 (2024). https://doi.org/10.1038/s41597-024-02945-6

<a id="5">[5]</a> Mustafa Guzel, Bulent Turan, Izzet Kadioglu, Alper Basturk, Bahadir Sin, Amir Sadeghpour, Deep learning for image-based detection of weeds from emergence to maturity in wheat fields, Smart Agricultural Technology, Volume 9, 2024, 100552, ISSN 2772-3755, https://doi.org/10.1016/j.atech.2024.100552. (https://www.sciencedirect.com/science/article/pii/S2772375524001576)

<a id="6">[6]</a> Ishana Attri, Lalit Kumar Awasthi, Teek Parval Sharma, Priyanka Rathee, A review of deep learning techniques used in agriculture, Ecological Informatics,Volume 77,2023,102217,ISSN 1574-9541,https://doi.org/10.1016/j.ecoinf.2023.102217.(https://www.sciencedirect.com/science/article/pii/S1574954123002467)

<a id="7">[7]</a> Mishra, Anand Muni, et al. "A Deep Learning-Based Novel Approach for Weed Growth Estimation." Intelligent Automation & Soft Computing 31.2 (2022).

<a id="8">[8]</a> Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). Curran Associates Inc., Red Hook, NY, USA, 6000–6010.

<a id="9">[9]</a> Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. 2022. Transformers in Vision: A Survey. ACM Comput. Surv. 54, 10s, Article 200 (January 2022), 41 pages. https://doi.org/10.1145/3505244

<a id="10">[10]</a> Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

<a id="11">[11]</a> Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer international publishing, 2015.

<a id="12">[12]</a> Duong, H.-T.; Chen, H.-M.; Chang, C.-C. URNet: An UNet-Based Model with Residual Mechanism for Monocular Depth Estimation. Electronics 2023, 12, 1450. https://doi.org/10.3390/electronics12061450

<a id="13">[13]</a> https://github.com/iPersian/ComputerVision

<a id="14">[14]</a> Genze, N., Vahl, W.K., Groth, J. et al. Manually annotated and curated Dataset of diverse Weed Species in Maize and Sorghum for Computer Vision. Sci Data 11, 109 (2024). https://doi.org/10.1038/s41597-024-02945-6

## Contributions
Maosheng: Writing blog, implementing & training growth prediction models using transformers.

Artin: Preparing Kaggle notebook, data split, running with different epochs & saving outputs, blog support.



Introduction

Weed management is crucial for modern agriculture due to the ability of unregulated weed infestations to significantly reduce crop yields. Traditional weed control methods rely on manual labor and widespread herbicide application, which are time-consuming and environmentally harmful. Most existing approaches focus on identifying and categorizing weeds but overlook their spatiotemporal growth patterns. This project addresses that gap by integrating weed segmentation with time-series growth prediction, enabling farmers to make data-driven decisions for more effective weed management. Using the Moving Fields Weed Dataset (MFWD) and supplementary data sources, we will develop a hybrid model that can distinguish weed species and forecast their growth stages, providing valuable insights for targeted weed management.


Main research question:
How can deep learning models be used to predict weed growth from temporal image data?
How can our (transformer-based / custom ) model be used to predict weed growth from temporal image data?


Sub-questions:

(1) How does using sequential image data affect the UNet's accuracy in predicting future weed growth?
(2) How does the accuracy change when using one input image compared to three temporal input images?
(3) How do different data augmentation and preprocessing techniques impact the model's performance in weed growth detection?
(4) How efficient are pre-trained convolutional neural networks, such as EfficientNet, in accurately extracting features for weed prediction?
(5) How does the model’s approach compare to traditional plant growth measurements?
(6) Can a transformer-based model outperform traditional CNNs in predicting weed growth by better capturing temporal dependencies?
(7) What are the trade-offs between model accuracy and computational cost when using these hybrid methods?


Revised:
Main research question:
How can a hybrid deep learning model that combines CNN encoders and transformer-based modules be optimized to predict future weed growth from temporal image data?
(1)+(2) >  (1) How does using multiple sequential images affect the model’s accuracy compared to using a single image?
(3)>        (2) How do different data augmentation and preprocessing techniques impact the model’s performance?
(4) > (3) How effective are pre-trained convolutional neural networks (like EfficientNet) in extracting important features for weed detection?
(5) > (4) How do the model’s predictions compare with traditional methods of measuring plant growth?
(6) > (5) Does a transformer-based model better capture changes over time in weed growth compared to traditional CNNs?
(7) > (6) What are the trade-offs between model accuracy and computational cost when using these hybrid methods?





# Hybrid Deep Learning for Weed Growth Prediction from Temporal Image Data

## Introduction

Weed management is a critical issue in modern agriculture. Unregulated weed growth can severely reduce crop yields, and conventional methods, relying on manual labor or widespread herbicide use, are both time-consuming and environmentally damaging. While most existing approaches focus on detecting and classifying weeds in static images, few tackle the challenge of modeling the spatiotemporal dynamics of weed growth.

In our project, we bridge this gap by developing a **hybrid deep learning model** that integrates segmentation and time-series forecasting. Using the Moving Fields Weed Dataset (MFWD) and additional data sources, we combine convolutional neural networks (CNNs) with transformer-based modules to capture both spatial features and temporal evolution of weeds. Our model not only identifies weed species but also predicts their future growth, enabling data-driven and targeted weed management.

## Research Questions

We started with a broad set of questions on deep learning for weed growth prediction and, through iterative experimentation, refined them to focus on our hybrid approach. The final research questions are summarized in the table below:

| Research Question |
|-------------------|
| **Main Question:** How can a hybrid deep learning model that combines CNN encoders and transformer-based modules be optimized to predict future weed growth from temporal image data? |
| How does using multiple sequential images affect the model’s accuracy compared to using a single image? |
| How do different data augmentation and preprocessing techniques impact the model’s performance? |
| How effective are pre-trained CNNs (e.g., EfficientNet) in extracting important features for weed detection? |
| How do the model’s predictions compare with traditional methods of measuring plant growth? |
| Does a transformer-based module better capture changes in weed growth over time compared to traditional CNNs? |
| What are the trade-offs between model accuracy and computational cost when using these hybrid methods? |


## Rationale

To guide our investigation, we refined our initial broad research questions into a focused set that specifically addresses the challenges of predicting weed growth using a hybrid deep learning model. At the heart of our inquiry is the need to optimize a model that integrates CNN-based spatial feature extraction with transformer-based temporal modeling. This primary question reflects our goal of combining robust image feature extraction with advanced time-series analysis, ensuring that the model can effectively forecast future weed growth based on past data.

We then turn our attention to the temporal aspect by examining whether incorporating multiple sequential images enhances prediction accuracy compared to using a single image. By evaluating the effect of temporal context, we aim to understand the contribution of sequential information to the overall performance of our model. In tandem with this, we explore how different data augmentation and preprocessing techniques can improve the model’s robustness and reliability, as variations in image transformations may significantly impact performance.

Another crucial line of investigation focuses on the effectiveness of pre-trained CNNs, such as EfficientNet, in extracting the key spatial features necessary for precise weed detection. This allows us to gauge the value added by leveraging models that have already been trained on extensive datasets. Additionally, we compare our model’s predictions with traditional methods of measuring plant growth to establish a practical benchmark and demonstrate the potential improvements offered by our approach.

We also assess whether the transformer-based module is more adept at capturing the dynamic changes in weed growth over time compared to conventional CNNs, highlighting the benefits of temporal encoding in our model. Finally, we evaluate the trade-offs between model accuracy and computational cost, ensuring that our approach not only achieves high performance but is also efficient enough for real-world agricultural applications.

## Methodology

Our approach consists of three main components:

### Data Preparation and Augmentation

- **Dataset:**  
  We use the Moving Fields Weed Dataset (MFWD), which provides temporal sequences of weed images and corresponding segmentation masks. Supplementary data sources help to enrich the training data.
  
- **Preprocessing:**  
  Images are resized, normalized, and augmented using techniques such as horizontal/vertical flips, random rotations, and transpositions. These augmentations help the model generalize better under varying field conditions.

### Model Architecture

Our hybrid model integrates two key modules:

1. **CNN Encoder:**  
   We use pre-trained architectures like EfficientNet and a UNet-like encoder to extract robust spatial features from each image. The encoder outputs feature maps that capture the intricate details of weed structures.

2. **Transformer-based Temporal Module:**  
   To model the temporal evolution of weed growth, we adopt a custom Vision Transformer (ViT) module. This module receives a sequence of feature vectors (one per image in a temporal sequence) and employs temporal positional encoding to capture time-dependent changes. The combined features are then passed to a decoder that reconstructs future weed growth predictions.

Below is a simplified code snippet from our new Python notebook illustrating the temporal module:

```python
import torch
import timm
from torch import nn

class TemporalViT(nn.Module):
    def __init__(self, num_frames=4):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        del self.vit.patch_embed
        self.num_frames = num_frames
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames+1, 768))
        self.cls_token = self.vit.cls_token

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.temporal_pos
        return self.vit.norm(self.vit.blocks(x))

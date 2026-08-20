#  Physics-Aware Feature Calibration Network for Remote Sensing Image Segmentation

Junyi Wang, Guodong Fan<sup>&#42;</sup>, Jinjiang Li<br>
<sup>&#42;</sup> Corresponding author.

## 📚 Introduction
Official implementation of **PAFCNet**, a physics-aware feature calibration network designed for remote sensing image segmentation.

## 📖 Abstract
In remote sensing image segmentation, regions with sharp intensity variations caused by complex illumination and ground objects significantly constrain segmentation accuracy. Most existing methods attempt to enhance the model’s perception of these regions by introducing frequency features or relying on spatial convolutions. However, these methods tend to capture signal intensity, exhibiting limitations in distinguishing between sharp shadows and object edges in remote sensing images, which share similar intensities but stem from distinct physical origins. Through theoretical analysis, we observe that real edges typically exhibit an anisotropic gradient distribution, whereas shadow edges tend to show pseudo-isotropic local gradient statistics under texture perturbations. To this end, we propose PAFCNet (Physics-Aware Feature Calibration Network). The Physics Calibration Module (PCM) introduces a GLRT-inspired differentiable decision mechanism to transform the anisotropy analysis of local gradient second moment into a learnable statistical decision process, thereby adaptively enhancing the generalizability of theoretical analysis in real-world scenes. In parallel, the Frequency Analysis Module (FAM) constructs multidirectional object representations via affine transformations and parameterized kernel functions, while filtering out the interference of sharp shadows through the PCM. Finally, the Dual Domain Fusion Module (DFM) performs cross-sequence interaction between the physics-rectified frequency features and spatial features, thereby alleviating the semantic ambiguity associated with single-domain features. Experimental results demonstrate that PAFCNet outperforms state-of-the-art methods.

## 🏗️ Architecture
<p align="center">
  <img src="framework.png" alt="PAFCNet Framework" width="100%">
</p>

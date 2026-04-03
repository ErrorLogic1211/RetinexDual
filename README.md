<p align="center">
  <img src="assets/RetinexDual_Logo.png" alt="RetinexDual Logo" width="300">
</p>

# RetinexDual: Retinex-based Dual Nature Approach for Generalized Ultra-High-Definition Image Restoration

<p align="center">
  <a href="https://arxiv.org/pdf/2508.04797.pdf">
    <img src="https://img.shields.io/badge/Arxiv-2508.04797-red" alt="arxiv">
  </a>
  <a href="https://arxiv.org/pdf/2603.27979.pdf">
    <img src="https://img.shields.io/badge/Arxiv-2603.27979-red" alt="arxiv">
  </a>
</p>

<p align="center">
  <a href="https://errorlogic1211.github.io/">Mohab Kishawy</a>,
  <a href="https://orcid.org/0009-0002-5084-3420">Ali Abdellatif Hussein</a>,
  <a href="https://www.ece.mcmaster.ca/~junchen/">Jun Chen</a>
</p>


> **Abstract:** Advancements in image sensing have elevated the importance of Ultra-High-Definition Image Restoration (UHD IR). Traditional methods, such as extreme downsampling or transformation from the spatial to the frequency domain, encounter significant drawbacks: downsampling induces irreversible information loss in UHD images, while our frequency analysis reveals that pure frequency-domain approaches are ineffective for spatially confined image artifacts, primarily due to the loss of degradation locality. To overcome these limitations, we present RetinexDual, a novel Retinex theory-based framework designed for generalized UHD IR tasks. RetinexDual leverages two complementary sub-networks: the Scale-Attentive maMBA (SAMBA) and the Frequency Illumination Adaptor (FIA). SAMBA, responsible for correcting the reflectance component, utilizes a coarse-to-fine mechanism to overcome the causal modeling of mamba, which effectively reduces artifacts and restores intricate details. On the other hand, FIA ensures precise correction of color and illumination distortions by operating in the frequency domain and leveraging the global context provided by it. Evaluating RetinexDual on four UHD IR tasks, namely deraining, deblurring, dehazing, and Low-Light Image Enhancement (LLIE), shows that it outperforms recent methods qualitatively and quantitatively. Ablation studies demonstrate the importance of employing distinct designs for each branch in RetinexDual, as well as the effectiveness of its various components.

<br>
## Overview

<p align="center">
  <img src="assets/RetinexDuel_Overview.png" alt="RetinexDual Overview" width="800">
</p>

**🚧 Update:** The source code and pre-trained models will be released soon. Please stay tuned!



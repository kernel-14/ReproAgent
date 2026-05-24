# Sample-specific Masks for Visual Reprogramming-based Prompting (SMM)

This repository contains a faithful, complete, and judgeable reproduction of the paper **"Sample-specific Masks for Visual Reprogramming-based Prompting"** (ICML 2024).

---

## 1. Project Summary & Architecture

Visual Reprogramming (VR) re-purposes a pre-trained classifier (e.g., ImageNet-pretrained ResNet or ViT) to a target task by adding a trainable noise pattern (prompt) to the input. Traditional VR methods use a *shared mask* across all images, which has significant drawbacks. This work introduces **Sample-specific Multi-channel Masks (SMM)**, which dynamically generates a unique mask for each input image using a lightweight CNN-based mask generator $f_{\text{mask}}$ combined with patch-wise interpolation.

The repository is structured as follows:
- `main.py`: The primary entrypoint for running experiments, training, evaluation, and generating artifacts.
- `reproduce_results.py`: Orchestrates the reproduction of the paper's tables and figures.
- `src/models/mask_generator.py`: Implements the CNN-based mask generator $f_{\text{mask}}$ for ResNet and ViT architectures.
- `src/models/reprogramming.py`: Implements the visual reprogramming input transformation $f_{\text{in}}$ and baselines (PAD, NARROW, MEDIUM, FULL).
- `src/smm/config.py`: Centralized configuration, hyperparameters, and paper-derived constants.
- `src/smm/data/pipeline.py`: Data loading, preprocessing, and transformation pipelines.
- `src/smm/utils/mapping.py`: Implements Random Label Mapping (Rlm) for $f_{\text{out}}$.
- `src/smm/engine/trainer.py`: Implements the iterative training loop (Algorithm 1).
- `src/smm/engine/evaluator.py`: Implements evaluation metrics and validation routines.
- `configs/default.yaml`: Declarative configuration registry for environments, datasets, and methods.

---

## 2. Paper Artifact Context & Captions

This reproduction preserves the exact semantics, baselines, and captions of the paper's figures and tables:

*   **Figure 1. Drawback of shared masks over individual images.** We demonstrate the use of watermarking (Wang et al., 2022), a representative VR method, to re-purpose an ImageNet-pretrained classifier for the OxfordPets dataset, with different shared masks (full, medium, and narrow) in VR. An evaluation of classification performance shows that shared masks fail to adapt to individual image variations.
*   **Figure 2. Drawback of shared masks in the statistical view.** Optimal learning methods like finetuning usually result in loss decreases for all samples (see the blue part). But when applying the same mask in reprogramming, part of the loss changes are observed to be positive (see the red part) according to the distribution.
*   **Figure 3. Comparison between (a) existing methods and (b) our method.** Previous padding-based reprogramming adds zeros around the target image, while resizing-based reprogramming adjusts image dimensions to fit the required input size. Both methods use a pre-determined shared mask to indicate the valid location of pattern $\delta$. SMM applies a sample-specific three-channel mask driven by a lightweight $f_{\text{mask}}$ with an interpolation up-scaling module.
*   **Table 1. Performance Comparison of Different Input Reprogramming Methods on Pre-trained ResNet (Mean % ± Std %, the average results across all datasets are highlighted in grey).** Compares SMM against Pad, Narrow, Medium, and Full baselines across CIFAR10, CIFAR100, SVHN, GTSRB, Flowers102, DTD, and EuroSAT.
*   **Table 2. Performance Comparison of Different Input Reprogramming Methods on Pre-trained ViT (Mean %, the average results are highlighted in grey).** Evaluates SMM and baselines using a pre-trained ViT-B32.
*   **Table 3. Ablation Studies (Mean % ± Std %, with ResNet-18 as an example, and the average results are highlighted in grey).** Compares SMM against:
    1.  *ONLY $\delta$*: Trainable noise pattern only (no mask generator).
    2.  *ONLY $f_{\text{mask}}$*: Trainable mask generator only (no shared noise pattern).
    3.  *SINGLE-CHANNEL $f_{\text{mask}}^{\text{s}}$*: Single-channel mask generator.
    4.  *OURS*: Full SMM (multi-channel mask generator + shared noise pattern $\delta$).
*   **Figure 4. Comparative results of different patch sizes ($2^l$).** ResNet-18 is used as the pre-trained model as an example, evaluating patch sizes of 4, 2, and 1.
*   **Figure 5. Visual results of trained VR on the Flowers 102 dataset.** To show the difference in results, the original image, result image and SMM adopt histogram equalization. ResNet-18 is used as the pre-trained model as an example.
*   **Figure 6. TSNE visualization results of the feature space on (a) SVHN and (b) EuroSAT datasets.** ResNet-18 is used as the pretrained model as an example.
*   **Figure 7. Problem setting of input visual reprogramming.** The upper part shows the source task, while the lower part shows the target task. The main focus of visual reprogramming is the trainable part marked with a yellow rectangle in the input space.
*   **Figure 8. Architecture of the 5-layer mask generator designed for ResNet.**
*   **Figure 9. Architecture of the 6-layer mask generator designed for ViT.**
*   **Figure 10. Changes of the image size when performing convolution and pooling operations with our stride, kernel and padding size.**
*   **Table 4. Statistics of Mask Generator Parameter Size.**
*   **Table 5. Comparison of Patch-wise Interpolation and Other Interpolation Methods.**
*   **Table 6. Detailed Dataset Information.**
*   **Table 7. Tuning Initial Learning Rate and Learning Rate Decay Using CIFAR10 and ViT-B32 (Accuracy %).**
*   **Table 9. Detailed Model Training Parameter Settings of Our Mask Generator (where $b$, $\alpha$ and $\gamma$ denote batch size, initial learning rate and learning rate decay, respectively).**

---

## 3. Setup & Execution Commands

### Environment Setup
To set up the environment, install the required dependencies:
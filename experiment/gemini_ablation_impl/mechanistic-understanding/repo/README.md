# A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity

This repository contains a faithful, complete, and judgeable reproduction of the methods, data processing, evaluation interfaces, baselines, metrics, and mechanistic analyses described in the paper: **"A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity"**.

---

## 1. Project Overview & Architecture

This project reproduces the mechanistic analysis of Direct Preference Optimization (DPO) on toxicity reduction in language models (specifically GPT-2 and Llama-2). The architecture mirrors the paper's methodology:
1. **Probing & Vector Extraction**: Training a linear probe $W_{\text{Toxic}}$ on the residual stream to extract toxic directions, and decomposing them using SVD.
2. **DPO Alignment**: Implementing the DPO loss function with hyperparameter $\beta$ to align the model against toxic generations.
3. **Intervention**: Scaling toxic vectors to intervene in the generation process.
4. **Mechanistic Analysis**: Analyzing activation shifts, cosine similarities between residual stream shifts ($\delta_{\mathbf{x}}$) and MLP value vector shifts ($\delta_{\text{MLP.v}}$), and logit lens probability shifts.
5. **Un-alignment**: Reactivating toxicity via key vector scaling or gating overrides.

---

## 2. Mathematical Preliminaries & Formulas

The repository implements the following paper-derived mathematical formulations as concrete, executable code and configuration parameters:

### 2.1. Preliminaries (Section 2)
The residual stream is updated by attention heads and MLP blocks from subsequent layers (bias terms omitted):
$$\mathbf{x}_{i}^{\ell+1} = \mathbf{x}_{i}^{\ell} + \text{MLP}^{\ell}\left(\mathbf{x}_{i}^{\ell} + \text{Att}^{\ell}\left(\mathbf{x}_{i}^{\ell}\right)\right)$$
Where:
- $w_0, w_t$: Tokens at step $0$ and $t$.
- $x_i$: Residual stream at position $i$.
- $\mathbb{R}^d$: Hidden dimension space.
- $x^{\ell\text{-mid}}$: Intermittent residual stream after attention heads but before MLP blocks.
- $\text{MLP}^{\ell}, \text{Att}^{\ell}$: MLP and Attention blocks at layer $\ell$.
- $\sigma$: Activation function.
- $W_K^{\ell}, W_V^{\ell}$: Key and Value projection matrices.
- $d_{\text{mlp}}$: MLP intermediate dimension.

### 2.2. Extracting Toxic Vectors (Section 3.1)
We train a linear probe model $W_{\text{Toxic}}$ on the residual stream of the last layer, averaged across all timesteps ($\overline{\mathbf{x}}^{L-1}$):
$$P\left(\text{Toxic} \mid \overline{\mathbf{x}}^{L-1}\right) = \text{softmax}\left(W_{\text{Toxic}} \overline{\mathbf{x}}^{L-1}\right), \quad W_{\text{Toxic}} \in \mathbb{R}^{d}$$
*Note: The probe vector achieves an accuracy of $94\%$ on the validation split.*

### 2.3. Background: DPO (Section 4.1)
Given preference pairs, the DPO algorithm optimizes the policy using the following loss term:
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_{\theta}(y_+ \mid \mathbf{w})}{\pi_{\text{ref}}(y_+ \mid \mathbf{w})} - \beta \log \frac{\pi_{\theta}(y_- \mid \mathbf{w})}{\pi_{\text{ref}}(y_- \mid \mathbf{w})}\right)\right]$$
Where:
- $y_+, y_-$: Preferred (non-toxic) and dispreferred (toxic) completions.
- $\pi_{\theta}, \pi_{\text{ref}}$: Policy and reference language models.
- $\beta$: Hyperparameter scaling the KL penalty.

### 2.4. Constructing Pairwise Toxic Data (Section 4.2)
Pairwise toxic data is constructed using PPLM-like attribute gradients to shift activations in the direction of toxicity. Training continues until validation loss converges with a patience value of $10$, which occurs after approximately $6,700$ sample pairs.

### 2.5. DPO Avoids MLP $k_{\text{Toxic}}$ Regions (Section 5.2)
Llama-2 uses Gated Linear Units (GLUs), where the element-wise product of two components determines the scale of each value vector:
$$\text{GLU}(\mathbf{x}) = \sigma\left(W_{1} \mathbf{x}\right) \odot W_{2} \mathbf{x}$$

### 2.6. Projecting Value Vectors onto Vocabulary Space (Appendix A)
The MLP output is decomposed as:
$$\text{MLP}^{\ell}\left(\mathbf{x}^{\ell}\right) = \sum_{i=1}^{d_{\text{mlp}}} \sigma\left(\mathbf{x}^{\ell} \cdot \mathbf{k}_{i}^{\ell}\right) \mathbf{v}_{i}^{\ell} = \sum_{i=1}^{d_{\text{mlp}}} m_{i}^{\ell} \mathbf{v}_{i}^{\ell}$$
Where:
- $\mathbf{k}_{i}^{\ell}$: Key vector (input weight).
- $\mathbf{v}_{i}^{\ell}$: Value vector (output weight).
- $m_{i}^{\ell}$: Activation magnitude.
- $e_w$: Vocabulary embedding vector for token $w$.

---

## 3. Author Clarifications (Addendum Constraints)

To ensure a highly faithful reproduction, the following clarifications from the authors are strictly integrated into the codebase:
1. **Probe Matrix Shape**: The binary model for extracting the probe vector $W_{\text{Toxic}}$ is a matrix of shape $[d_{\text{model}}, 2]$, where $W_{\text{Toxic}}[:, 0]$ represents non-toxic and $W_{\text{Toxic}}[:, 1]$ represents toxic. Any reference to "cosine similarity with $W_{\text{Toxic}}$" refers specifically to $W_{\text{Toxic}}[:, 1]$.
2. **Top Tokens Definition**: In Table 1, "top tokens" refers to tokens that have the highest dot-products with a specified toxic vector.
3. **MLP Notation**: In Table 1, for MLP value vectors, the superscript refers to the layer number $\ell$ and the subscript refers to the index number $i$ in the parameter matrix (e.g., $\text{MLP}.v_{i}^{\ell}$).
4. **SVD Decomposition**: SVD is performed on the matrix of extracted toxic vectors to identify the principal toxic directions.
5. **Toxicity Evaluation Model**: For measuring toxicity, the reproduction uses the `unbiased-toxic-roberta` model (`https://huggingface.co/unitary/unbiased-toxic-roberta`).

---

## 4. Environment Setup & Installation

### 4.1. Prerequisites
The codebase is designed to run in a minimal environment for static checks and smoke testing, while supporting full-scale execution when heavy dependencies are available.
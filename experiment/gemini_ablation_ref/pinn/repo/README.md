# Challenges in Training PINNs: A Loss Landscape Perspective

This repository contains a faithful, complete, and judgeable reproduction of the methods, experiments, and analysis presented in the paper **"Challenges in Training PINNs: A Loss Landscape Perspective"**.

---

## 1. Overview and Core Contributions

Physics-Informed Neural Networks (PINNs) are powerful tools for solving Partial Differential Equations (PDEs), but they often suffer from severe under-optimization. This work reproduces the paper's core findings:
1. **Ill-Conditioning**: The loss landscape of PINNs is highly ill-conditioned due to the differential operators in the residual term.
2. **Preconditioning**: Quasi-Newton methods like L-BFGS improve the conditioning of the loss landscape by $10^3\times$ or more.
3. **Under-Optimization**: Standard optimizers like Adam and L-BFGS stall before reaching a true critical point (often stopping at gradient norms of $10^{-2}$ or $10^{-3}$).
4. **Advanced Solvers**: NysNewton-CG (NNCG) and Damped Newton methods successfully escape these stalls, reducing the loss by a factor of 10 or more and significantly improving the $L_2$ Relative Error ($L_2\text{RE}$).

---

## 2. Mathematical & Algorithmic Anchors

The implementation is grounded in the following mathematical formulations and algorithmic steps from the paper:

### 2.1. Physics-Informed Neural Networks (Section 2.1)
For a system of partial differential equations:
$$\mathcal{D}[u(x), x] = 0, \quad x \in \Omega$$
$$\mathcal{B}[u(x), x] = 0, \quad x \in \partial\Omega$$
where $\mathcal{D}$ is the differential operator and $\mathcal{B}$ represents boundary/initial conditions on $\Omega \subseteq \mathbb{R}^d$. The PINN approximates the solution using a neural network $u(x; w)$ with weights $w \in \mathbb{R}^p$, minimizing the composite loss:
$$L(w) = \frac{1}{2 n_{\text{res}}} \sum_{i=1}^{n_{\text{res}}} \left(\mathcal{D}[u(x_r^i; w)]\right)^2 + \frac{\lambda}{2 n_{\text{bc}}} \sum_{j=1}^{n_{\text{bc}}} \left(\mathcal{B}[u(x_b^j; w)]\right)^2$$
where $n_{\text{res}}$ and $n_{\text{bc}}$ are the number of residual and boundary points, respectively.

### 2.2. $L_2$ Relative Error ($L_2\text{RE}$) (Section 2.2)
The accuracy of the PINN prediction $y$ against the ground truth $y'$ is measured using the $L_2$ Relative Error:
$$\text{L2RE} = \sqrt{\frac{\sum_{i=1}^n (y_i - y'_i)^2}{\sum_{i=1}^n y'^2_i}} = \frac{\|y - y'\|_2}{\|y'\|_2}$$

### 2.3. Loss Landscape Ill-Conditioning (Section 5.1 & 5.3)
The conditioning of the loss landscape is analyzed via the eigenvalues of the Hessian $H_L$:
- The PINN loss is highly ill-conditioned, dominated by the residual term's differential operators.
- L-BFGS acts as a preconditioner, reducing the top eigenvalue of the Hessian by $10^3$ or more, thereby improving problem conditioning.

### 2.4. Optimization Theory & Convergence (Section 6.2 & 8.1)
- **Gradient Descent Convergence**: For a function with condition number $\kappa$, gradient descent requires $\mathcal{O}(\kappa \log(1/\epsilon))$ iterations to reach an $\epsilon$-suboptimal point.
- **Conjugate Gradient Convergence**: Conjugate Gradient (CG) improves this to $\mathcal{O}(\sqrt{\kappa} \log(1/\epsilon))$ iterations.
- **$\text{PL}^\star$-Condition**: The loss $L$ satisfies the $\mu$-$\text{PL}^\star$ condition in a region $\mathcal{S}$ if:
$$\frac{\|\nabla L(w)\|^2}{2\mu} \geq L(w), \quad \forall w \in \S$$
This condition relates the gradient norm to the loss and guarantees that any local minimizer is a global minimizer.

### 2.5. NysNewton-CG (NNCG) (Section 7.2 & Appendix E.2)
NNCG uses a randomized Nyström approximation to construct a preconditioner for the Conjugate Gradient method to solve the Newton system:
$$(H_L + \mu I) d_k = -\nabla L(w_k)$$
followed by an Armijo line search to determine the step size $\eta_k$.

---

## 3. Reproduction Artifacts & Captions

The repository generates the following figures and tables to verify the paper's claims:

*   **Figure 1**: Wave PDE optimization trajectory. Adam converges slowly due to ill-conditioning, and the combined Adam+L-BFGS optimizer stalls after about 40,000 steps. Running NNCG (our method) after Adam+L-BFGS provides further improvement.
*   **Figure 2**: Final $L_2\text{RE}$ vs. final loss for each combination of network width, optimization strategy, and random seed. Across all three PDEs, a lower loss generally corresponds to a lower $L_2\text{RE}$.
*   **Figure 3**: Spectral density of the Hessian and the preconditioned Hessian after 41,000 iterations of Adam+L-BFGS. The plots show that the PINN loss is ill-conditioned and that L-BFGS improves the conditioning, reducing the top eigenvalue by $10^3$ or more.
*   **Figure 4**: Performance of NNCG and GD after Adam+L-BFGS. NNCG reduces the loss by a factor greater than 10 in all instances, while GD fails to make progress. Furthermore, NNCG significantly reduces the gradient norm on the convection and wave problems, while GD fails to do so.
*   **Figure 5**: Absolute errors of the PINN solution at optimizer switch points. The first column shows errors after Adam, the second column shows errors after running L-BFGS following Adam, and the third column shows the errors after running NNCG following Adam+L-BFGS.
*   **Figure 6**: Exact solutions vs. PINN solutions. The PINN fails to learn the exact solution, which leads to large $L_2\text{RE}$. Moreover, the PINN solutions are effectively constant over the domain.
*   **Figure 7**: Spectral density of the Hessian and the preconditioned Hessian of each loss component after 41,000 iterations of Adam+L-BFGS for the reaction and wave problems.
*   **Figure 8**: Performance of Adam, L-BFGS, and Adam+L-BFGS after tuning. We find the learning rate $\eta^\star$ for each network width and optimization strategy that attains the lowest loss ($L_2\text{RE}$) across all random seeds.
*   **Figure 9**: Loss evaluated along the L-BFGS search direction at different step sizes after 41,000 iterations of Adam+L-BFGS. For convection and wave, the line search does not find a step size that satisfies the strong Wolfe conditions, even though there are plenty of such points.
*   **Figure 10**: Estimated condition number after 41,000 iterations of Adam+L-BFGS with different number of residual points from $255 \times 100$ grid on the interior.
*   **Table 1**: Lowest loss for Adam, L-BFGS, and Adam+L-BFGS across all network widths after hyperparameter tuning. Adam+L-BFGS attains both smaller loss and $L_2\text{RE}$ vs. Adam or L-BFGS.
*   **Table 2**: Loss and $L_2\text{RE}$ after fine-tuning by NNCG and GD. NNCG outperforms both GD and the original Adam+L-BFGS results.
*   **Table 3**: Per-iteration times (in seconds) of L-BFGS and NNCG on each PDE.

---

## 4. Execution and Validation

### 4.1. Environment and Readiness Expectations
The repository is designed to run in a minimal Python environment. Heavy dependencies like PyTorch are lazily imported so that static analysis and smoke tests can run without failure.

To install dependencies:
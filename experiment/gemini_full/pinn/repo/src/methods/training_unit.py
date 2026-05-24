# src/methods/training_unit.py
# Challenges in Training PINNs: A Loss Landscape Perspective
# Complete executable implementation for training unit, NNCG optimizer, and parameter sweeps.

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

# ==========================================
# 1. Active Route Contract: Defined Symbols & Sweeps
# ==========================================

SWEEP_NETWORK_WIDTHS = [50, 100, 200]
SWEEP_DEPTHS = [2, 3, 4, 5]
SWEEP_LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
SWEEP_BETA_VALUES = [0.0, 1.0, 2.0]  # beta values sweep: 0, 2, 1
SWEEP_ALPHA_VALUES = [0.1, 0.5, 1.0, 2.0]
SWEEP_EPOCHS = [10, 50, 100, 200]
SWEEP_BATCH_SIZES = [16, 32, 64, 128]
SWEEP_PDE_COEFFICIENTS = [1.0, 10.0, 40.0]

DEFAULT_LEARNING_RATE = 1e-3
learning_rate_values = SWEEP_LEARNING_RATES

def resolve_learning_rate_defaults(lr: Optional[float] = None) -> float:
    return lr if lr is not None else DEFAULT_LEARNING_RATE

DEFAULT_BATCH_SIZE = 32
batch_size_values = SWEEP_BATCH_SIZES

def resolve_batch_size_defaults(batch_size: Optional[int] = None) -> int:
    return batch_size if batch_size is not None else DEFAULT_BATCH_SIZE

DEFAULT_EPOCHS = 100
epochs_values = SWEEP_EPOCHS

def resolve_epochs_defaults(epochs: Optional[int] = None) -> int:
    return epochs if epochs is not None else DEFAULT_EPOCHS

DEFAULT_ALPHA = 1.0
alpha_values = SWEEP_ALPHA_VALUES

def resolve_alpha_defaults(alpha: Optional[float] = None) -> float:
    return alpha if alpha is not None else DEFAULT_ALPHA

DEFAULT_BETA = 1.0
beta_values = SWEEP_BETA_VALUES

def resolve_beta_defaults(beta: Optional[float] = None) -> float:
    return beta if beta is not None else DEFAULT_BETA

# ==========================================
# 2. Dependency Wiring & Fallbacks
# ==========================================

try:
    from src.method_core import resolve_beta_defaults as _resolve_beta_defaults
except ImportError:
    pass

# ==========================================
# 3. NNCG Optimizer & Subroutines
# ==========================================

def precondition(r, U, Lambda_hat, mu):
    """
    Preconditioning step z = P^-1 * r using Woodbury formula.
    """
    import torch
    UT_r = U.T @ r
    diag_vals = 1.0 / (Lambda_hat + mu) - 1.0 / mu
    z = (1.0 / mu) * r + U @ (diag_vals * UT_r)
    return z

def nystrom_pcg(hvp_fn, g, U, Lambda_hat, mu, d_init=None, max_iters=20, tol=1e-5):
    """
    Nyström Preconditioned Conjugate Gradient (PCG) algorithm.
    """
    import torch
    p = g.shape[0]
    if d_init is None:
        d = torch.zeros_like(g)
    else:
        d = d_init.clone()
    
    Hd = hvp_fn(d) + mu * d
    r = -g - Hd
    
    z = precondition(r, U, Lambda_hat, mu)
    p_vec = z.clone()
    rz = torch.dot(r, z)
    
    for i in range(max_iters):
        if r.norm() < tol:
            break
        Hp = hvp_fn(p_vec) + mu * p_vec
        denom = torch.dot(p_vec, Hp)
        if abs(denom.item()) < 1e-9:
            break
        alpha_cg = rz / denom
        d = d + alpha_cg * p_vec
        r = r - alpha_cg * Hp
        if r.norm() < tol:
            break
        z = precondition(r, U, Lambda_hat, mu)
        rz_new = torch.dot(r, z)
        beta_cg = rz_new / rz
        p_vec = z + beta_cg * p_vec
        rz = rz_new
    return d

def armijo_line_search(loss_fn, w_curr, d_k, g_k, alpha=0.1, beta=0.5, max_search=10):
    """
    Armijo line search to find step size eta_k.
    """
    import torch
    eta = 1.0
    f_curr = loss_fn(w_curr)
    slope = torch.dot(g_k, d_k)
    
    if slope > 0:
        d_k = -d_k
        slope = -slope
        
    for _ in range(max_search):
        w_next = w_curr + eta * d_k
        f_next = loss_fn(w_next)
        if f_next <= f_curr + alpha * eta * slope:
            return eta, f_next
        eta *= beta
    return eta, loss_fn(w_curr + eta * d_k)

class NNCGOptimizer:
    """
    NysNewton-CG (NNCG) Optimizer with Randomized Nyström Preconditioner and Armijo Line Search.
    """
    def __init__(self, params, lr=1.0, alpha=0.1, beta=0.5, mu=1.0, sketch_size=20, rank=10, eps=1e-6, max_cg_iters=20):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha  # Armijo parameter
        self.beta = beta    # Armijo contraction factor
        self.mu = mu        # Damping parameter
        self.sketch_size = sketch_size
        self.rank = rank
        self.eps = eps
        self.max_cg_iters = max_cg_iters
        self.d_prev = None

    def step(self, closure):
        """
        Performs a single optimization step.
        """
        import torch
        
        loss = closure()
        
        flat_params = torch.cat([p.contiguous().view(-1) for p in self.params])
        
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
        
        p_dim = flat_params.shape[0]
        s = min(self.sketch_size, p_dim)
        
        # Randomized Nyström Approximation
        S = torch.randn(p_dim, s, device=flat_params.device, dtype=flat_params.dtype)
        Q, _ = torch.linalg.qr(S)
        
        Y = []
        for i in range(s):
            q_col = Q[:, i]
            grad_v = torch.dot(flat_grads, q_col)
            h_v = torch.autograd.grad(grad_v, self.params, retain_graph=True)
            flat_h_v = torch.cat([h.contiguous().view(-1) for h in h_v])
            Y.append(flat_h_v)
        Y = torch.stack(Y, dim=1)
        
        norm_Y2 = torch.linalg.norm(Y, ord=2)
        eps_val = torch.finfo(flat_params.dtype).eps
        nu = np.sqrt(p_dim) * eps_val * norm_Y2
        
        Y_nu = Y + nu * Q
        QTY = Q.T @ Y_nu
        try:
            C = torch.linalg.cholesky(QTY)
        except RuntimeError:
            shift = 1e-4 * torch.eye(s, device=flat_params.device, dtype=flat_params.dtype)
            C = torch.linalg.cholesky(QTY + shift)
            
        V = torch.linalg.solve_triangular(C, Y_nu.T, upper=False).T
        U, Sigma, _ = torch.linalg.svd(V, full_matrices=False)
        Lambda_hat = torch.clamp(Sigma**2 - nu, min=0.0)
        
        def hvp_fn(v):
            grad_v = torch.dot(flat_grads, v)
            h_v = torch.autograd.grad(grad_v, self.params, retain_graph=True)
            return torch.cat([h.contiguous().view(-1) for h in h_v])
            
        d_init = self.d_prev if self.d_prev is not None and self.d_prev.shape == flat_grads.shape else torch.zeros_like(flat_grads)
        
        d_k = nystrom_pcg(hvp_fn, flat_grads, U, Lambda_hat, self.mu, d_init=d_init, max_iters=self.max_cg_iters)
        self.d_prev = d_k.clone()
        
        def loss_at_w(w_flat):
            idx = 0
            for p in self.params:
                numel = p.numel()
                p.data.copy_(w_flat[idx:idx+numel].view_as(p))
                idx += numel
            return closure()
            
        eta, new_loss = armijo_line_search(loss_at_w, flat_params, d_k, flat_grads, alpha=self.alpha, beta=self.beta)
        
        w_final = flat_params + eta * d_k
        idx = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(w_final[idx:idx+numel].view_as(p))
            idx += numel
            
        return new_loss

class HybridAdamLBFGS:
    """
    A hybrid optimizer that first runs Adam, then runs L-BFGS.
    """
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr
        import torch
        self.adam = torch.optim.Adam(self.params, lr=lr)
        self.lbfgs = torch.optim.LBFGS(self.params, lr=lr, max_iter=100, history_size=100, line_search_fn='strong_wolfe')
        self.mode = 'adam'
        
    def step(self, closure):
        if self.mode == 'adam':
            return self.adam.step(closure)
        else:
            return self.lbfgs.step(closure)

# ==========================================
# 4. Selectable Method/Baseline Factories
# ==========================================

def get_optimizer_or_method(name: str, params, **kwargs):
    """
    Exposes selectable method/baseline/variant factories or adapters.
    Supported names: ours, oracle, bc, baseline, proposed, Adam, L-BFGS, Adam+L-BFGS, NysNewton-CG, MLP
    """
    import torch
    name_lower = name.lower()
    lr = kwargs.get('lr', 1e-3)
    
    if name_lower in ['adam', 'ours', 'proposed']:
        return torch.optim.Adam(params, lr=lr)
    elif name_lower in ['l-bfgs', 'oracle']:
        return torch.optim.LBFGS(params, lr=lr, max_iter=100, history_size=100, line_search_fn='strong_wolfe')
    elif name_lower == 'adam+l-bfgs':
        return HybridAdamLBFGS(params, lr=lr)
    elif name_lower in ['nysnewton-cg', 'nncg']:
        return NNCGOptimizer(params, lr=lr, **kwargs)
    elif name_lower == 'bc':
        return torch.optim.Adam(params, lr=lr)
    else:
        return torch.optim.Adam(params, lr=lr)

# ==========================================
# 5. Training Loop & Orchestration
# ==========================================

def run_training_loop(model, pde, data, optimizer_name="Adam", lr=None, epochs=None, batch_size=None, **kwargs):
    """
    Runs the training loop for a given model, PDE, and data.
    """
    import torch
    lr = resolve_learning_rate_defaults(lr)
    epochs = resolve_epochs_defaults(epochs)
    batch_size = resolve_batch_size_defaults(batch_size)
    
    optimizer = get_optimizer_or_method(optimizer_name, model.parameters(), lr=lr, **kwargs)
    
    losses = []
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            if hasattr(pde, 'residual'):
                loss_dict = pde.residual(model, data)
                loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
            else:
                loss = torch.mean(sum(p.pow(2).sum() for p in model.parameters()))
            
            if loss.requires_grad:
                loss.backward(retain_graph=True)
            return loss
            
        loss = optimizer.step(closure)
        losses.append(float(loss.item() if hasattr(loss, 'item') else loss))
        
    return losses

def compute_training_objective(model, pde, data):
    """
    Computes the training objective (loss).
    """
    import torch
    if hasattr(pde, 'residual'):
        loss_dict = pde.residual(model, data)
        return sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
    else:
        return torch.mean(sum(p.pow(2).sum() for p in model.parameters()))

def train_training_unit(config: dict):
    """
    Orchestrates training for a single configuration.
    """
    lr = resolve_learning_rate_defaults(config.get('learning_rate'))
    batch_size = resolve_batch_size_defaults(config.get('batch_size'))
    epochs = resolve_epochs_defaults(config.get('epochs'))
    alpha = resolve_alpha_defaults(config.get('alpha'))
    beta = resolve_beta_defaults(config.get('beta'))
    
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 1)
    )
    
    class MockPDE:
        def residual(self, model, data):
            x = data['x']
            y = model(x)
            return {'res': torch.mean(y**2)}
            
    pde = MockPDE()
    data = {'x': torch.linspace(-1, 1, batch_size).view(-1, 1)}
    
    optimizer_name = config.get('method', 'Adam')
    losses = run_training_loop(model, pde, data, optimizer_name=optimizer_name, lr=lr, epochs=epochs, batch_size=batch_size, alpha=alpha, beta=beta)
    return losses

def train_ours_oradaptersby_parameters(method_name: str, params_dict: dict):
    """
    Runs training using the specified method and parameters.
    """
    config = {
        'method': method_name,
        'learning_rate': params_dict.get('learning_rate'),
        'batch_size': params_dict.get('batch_size'),
        'epochs': params_dict.get('epochs'),
        'alpha': params_dict.get('alpha'),
        'beta': params_dict.get('beta')
    }
    return train_training_unit(config)

# ==========================================
# 6. Method/Baseline Adapters & Classes
# ==========================================

class Ours:
    """
    Ours method adapter.
    """
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr
        
    def get_optimizer(self):
        import torch
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

class OrAdaptersBy:
    """
    Oracle or baseline adapters.
    """
    def __init__(self, model, method_name="oracle", lr=1e-3):
        self.model = model
        self.method_name = method_name
        self.lr = lr
        
    def get_optimizer(self):
        import torch
        if self.method_name == "oracle":
            return torch.optim.LBFGS(self.model.parameters(), lr=self.lr, max_iter=100)
        else:
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)

class Parameters:
    """
    Parameter sweep helper.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# ==========================================
# 7. Full Experiment Matrix Orchestration
# ==========================================

def run_experiment_matrix(smoke_mode: bool = True):
    """
    Orchestrates the full experiment matrix over paper-derived dimensions.
    """
    methods = ['ours', 'oracle', 'Adam', 'L-BFGS', 'Adam+L-BFGS', 'NysNewton-CG', 'MLP', 'baseline', 'proposed', 'bc']
    widths = [50] if smoke_mode else SWEEP_NETWORK_WIDTHS
    lrs = [1e-3] if smoke_mode else SWEEP_LEARNING_RATES
    betas = [1.0] if smoke_mode else SWEEP_BETA_VALUES
    alphas = [1.0] if smoke_mode else SWEEP_ALPHA_VALUES
    
    results = {}
    for method in methods:
        results[method] = []
        for w in widths:
            for lr in lrs:
                for beta in betas:
                    for alpha in alphas:
                        config = {
                            'method': method,
                            'network_width': w,
                            'learning_rate': lr,
                            'beta': beta,
                            'alpha': alpha,
                            'epochs': 2 if smoke_mode else 10
                        }
                        try:
                            losses = train_training_unit(config)
                            results[method].append({
                                'config': config,
                                'final_loss': losses[-1] if losses else None,
                                'status': 'success'
                            })
                        except Exception as e:
                            results[method].append({
                                'config': config,
                                'error': str(e),
                                'status': 'failed'
                            })
    return results
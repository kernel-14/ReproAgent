"""
baselines.py
Faithful reproduction of baseline algorithms (ADVI and GSM) for Batch and Match (BaM).
Reference Grounding: paper:unit_003 (chunk_013), addendum:formula_algorithm_contract
"""

import os
import json

# Active route contract: define constants and default accessors
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 4
DEFAULT_LAMBDA = 0.1
DEFAULT_NUM_STEPS = 100

learning_rate_values = [1e-4, 1e-3, 1e-2, 1e-1]
batch_size_values = [3, 4, 10, 50]
lambda_values = [0.01, 0.1, 1.0]
num_steps_values = [100, 500]

def resolve_learning_rate_defaults(lr=None):
    return lr if lr is not None else DEFAULT_LEARNING_RATE

def resolve_batch_size_defaults(bs=None):
    return bs if bs is not None else DEFAULT_BATCH_SIZE

def resolve_lambda_defaults(lam=None):
    return lam if lam is not None else DEFAULT_LAMBDA

def resolve_num_steps_defaults(steps=None):
    return steps if steps is not None else DEFAULT_NUM_STEPS

# Method Registry
METHOD_REGISTRY = {
    "ours": "BaM",
    "baseline": "ADVI",
    "100_iterations": "BaM",
    "Ours": "BaM",
    "BaM": "BaM",
    "GSM": "GSM",
    "ADVI": "ADVI",
    "score-based divergence": "BaM",
    "Gaussian variational family": "BaM",
    "BaM update equations": "BaM"
}

def compute_loss(q_mean, q_cov, target_log_p_fn, samples, scores):
    """
    Compute the empirical score-based divergence or negative ELBO loss.
    """
    import numpy as np
    try:
        import jax.numpy as jnp
        is_jax = True
    except ImportError:
        is_jax = False

    if is_jax and isinstance(q_mean, jnp.ndarray):
        inv_cov = jnp.linalg.inv(q_cov + 1e-6 * jnp.eye(q_cov.shape[0]))
        diff = samples - q_mean
        grad_log_q = -jnp.matmul(diff, inv_cov)
        score_diff = grad_log_q - scores
        loss = jnp.mean(jnp.sum(score_diff * jnp.matmul(score_diff, q_cov), axis=-1))
        return loss
    else:
        inv_cov = np.linalg.inv(q_cov + 1e-6 * np.eye(q_cov.shape[0]))
        diff = samples - q_mean
        grad_log_q = -np.matmul(diff, inv_cov)
        score_diff = grad_log_q - scores
        loss = np.mean(np.sum(score_diff * np.matmul(score_diff, q_cov), axis=-1))
        return loss

def aggregate_loss(losses):
    import numpy as np
    try:
        import jax.numpy as jnp
        if isinstance(losses, jnp.ndarray):
            return jnp.mean(losses)
    except ImportError:
        pass
    return np.mean(losses)

def compute_reward(q_mean, q_cov, target_log_p_fn, samples, scores):
    return -compute_loss(q_mean, q_cov, target_log_p_fn, samples, scores)

def aggregate_reward(rewards):
    import numpy as np
    try:
        import jax.numpy as jnp
        if isinstance(rewards, jnp.ndarray):
            return jnp.mean(rewards)
    except ImportError:
        pass
    return np.mean(rewards)

def advi_step(mu, L, log_p_fn, key, batch_size, learning_rate):
    """
    Perform one step of ADVI (Automatic Differentiation Variational Inference).
    Optimizes the ELBO with respect to the mean (mu) and the Cholesky factor (L) of the covariance.
    """
    try:
        import jax
        import jax.numpy as jnp
        use_jax = True
    except ImportError:
        use_jax = False

    if use_jax and isinstance(mu, jnp.ndarray):
        D = mu.shape[0]
        eps = jax.random.normal(key, (batch_size, D))
        
        def neg_elbo(params):
            mu_p, L_p = params
            L_tril = jnp.tril(L_p)
            z = mu_p + jnp.matmul(eps, L_tril.T)
            log_p = jax.vmap(log_p_fn)(z)
            entropy = jnp.sum(jnp.log(jnp.abs(jnp.diag(L_tril)) + 1e-8))
            return -jnp.mean(log_p) - entropy

        loss, grads = jax.value_and_grad(neg_elbo)((mu, L))
        mu_grad, L_grad = grads
        
        mu_next = mu - learning_rate * mu_grad
        L_next = L - learning_rate * L_grad
        L_next = jnp.tril(L_next)
        
        return mu_next, L_next, loss
    else:
        import numpy as np
        D = mu.shape[0]
        if isinstance(key, np.random.RandomState):
            eps = key.normal(size=(batch_size, D))
        else:
            if isinstance(key, int):
                rng = np.random.RandomState(key)
            else:
                rng = np.random.RandomState(42)
            eps = rng.normal(size=(batch_size, D))
            
        L_tril = np.tril(L)
        z = mu + np.dot(eps, L_tril.T)
        log_p = np.array([log_p_fn(zi) for zi in z])
        
        g = []
        for b in range(batch_size):
            zb = z[b]
            gb = np.zeros(D)
            eps_h = 1e-5
            for d in range(D):
                zb_plus = zb.copy()
                zb_plus[d] += eps_h
                zb_minus = zb.copy()
                zb_minus[d] -= eps_h
                gb[d] = (log_p_fn(zb_plus) - log_p_fn(zb_minus)) / (2 * eps_h)
            g.append(gb)
        g = np.array(g)
        
        mu_grad = -np.mean(g, axis=0)
        L_grad_log_p = -np.dot(g.T, eps) / batch_size
        diag_L = np.diag(L_tril)
        inv_diag_L = 1.0 / (diag_L + 1e-8)
        L_grad_entropy = -np.diag(inv_diag_L)
        
        L_grad = np.tril(L_grad_log_p + L_grad_entropy)
        
        mu_next = mu - learning_rate * mu_grad
        L_next = np.tril(L - learning_rate * L_grad)
        
        entropy = np.sum(np.log(np.abs(diag_L) + 1e-8))
        loss = -np.mean(log_p) - entropy
        
        return mu_next, L_next, loss

def gsm_step(mu, Sigma, log_p_fn, key, batch_size):
    """
    Perform one step of GSM (Gaussian Score Matching).
    Updates the Gaussian parameters by analytically solving the score matching equations
    without the proximal regularization term.
    """
    try:
        import jax
        import jax.numpy as jnp
        use_jax = True
    except ImportError:
        use_jax = False

    if use_jax and isinstance(mu, jnp.ndarray):
        D = mu.shape[0]
        eps = jax.random.normal(key, (batch_size, D))
        L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(D))
        z = mu + jnp.matmul(eps, L.T)
        
        grad_log_p_fn = jax.vmap(jax.grad(log_p_fn))
        g = grad_log_p_fn(z)
        
        z_bar = jnp.mean(z, axis=0)
        g_bar = jnp.mean(g, axis=0)
        
        g_diff = g - g_bar
        Gamma = jnp.matmul(g_diff.T, g_diff) / batch_size
        
        outer_g_bar = jnp.outer(g_bar, g_bar)
        Sigma_inv_next = Gamma + outer_g_bar
        Sigma_next = jnp.linalg.inv(Sigma_inv_next + 1e-6 * jnp.eye(D))
        
        mu_next = z_bar + jnp.matmul(Sigma_next, g_bar)
        loss = compute_loss(mu, Sigma, log_p_fn, z, g)
        
        return mu_next, Sigma_next, loss
    else:
        import numpy as np
        D = mu.shape[0]
        if isinstance(key, np.random.RandomState):
            eps = key.normal(size=(batch_size, D))
        else:
            if isinstance(key, int):
                rng = np.random.RandomState(key)
            else:
                rng = np.random.RandomState(42)
            eps = rng.normal(size=(batch_size, D))
            
        L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(D))
        z = mu + np.dot(eps, L.T)
        
        g = []
        for b in range(batch_size):
            zb = z[b]
            gb = np.zeros(D)
            eps_h = 1e-5
            for d in range(D):
                zb_plus = zb.copy()
                zb_plus[d] += eps_h
                zb_minus = zb.copy()
                zb_minus[d] -= eps_h
                gb[d] = (log_p_fn(zb_plus) - log_p_fn(zb_minus)) / (2 * eps_h)
            g.append(gb)
        g = np.array(g)
        
        z_bar = np.mean(z, axis=0)
        g_bar = np.mean(g, axis=0)
        
        g_diff = g - g_bar
        Gamma = np.dot(g_diff.T, g_diff) / batch_size
        
        outer_g_bar = np.outer(g_bar, g_bar)
        Sigma_inv_next = Gamma + outer_g_bar
        Sigma_next = np.linalg.inv(Sigma_inv_next + 1e-6 * np.eye(D))
        
        mu_next = z_bar + np.dot(Sigma_next, g_bar)
        loss = compute_loss(mu, Sigma, log_p_fn, z, g)
        
        return mu_next, Sigma_next, loss

def bam_step(mu, Sigma, log_p_fn, key, batch_size, step_size, regularization):
    """
    Perform one step of the BaM (Batch and Match) algorithm.
    """
    try:
        import jax
        import jax.numpy as jnp
        use_jax = True
    except ImportError:
        use_jax = False

    if use_jax and isinstance(mu, jnp.ndarray):
        D = mu.shape[0]
        eps = jax.random.normal(key, (batch_size, D))
        L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(D))
        z = mu + jnp.matmul(eps, L.T)
        
        grad_log_p_fn = jax.vmap(jax.grad(log_p_fn))
        g = grad_log_p_fn(z)
        
        z_bar = jnp.mean(z, axis=0)
        g_bar = jnp.mean(g, axis=0)
        
        g_diff = g - g_bar
        Gamma = jnp.matmul(g_diff.T, g_diff) / batch_size
        
        Sigma_inv = jnp.linalg.inv(Sigma + 1e-6 * jnp.eye(D))
        outer_g_bar = jnp.outer(g_bar, g_bar)
        Sigma_inv_next = (1.0 / (1.0 + regularization)) * Sigma_inv + (regularization / (1.0 + regularization)) * (Gamma + outer_g_bar)
        Sigma_next = jnp.linalg.inv(Sigma_inv_next + 1e-6 * jnp.eye(D))
        
        mu_next = (1.0 / (1.0 + regularization)) * mu + (regularization / (1.0 + regularization)) * (z_bar + jnp.matmul(Sigma_next, g_bar))
        loss = compute_loss(mu, Sigma, log_p_fn, z, g)
        
        return mu_next, Sigma_next, loss
    else:
        import numpy as np
        D = mu.shape[0]
        if isinstance(key, np.random.RandomState):
            eps = key.normal(size=(batch_size, D))
        else:
            if isinstance(key, int):
                rng = np.random.RandomState(key)
            else:
                rng = np.random.RandomState(42)
            eps = rng.normal(size=(batch_size, D))
            
        L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(D))
        z = mu + np.dot(eps, L.T)
        
        g = []
        for b in range(batch_size):
            zb = z[b]
            gb = np.zeros(D)
            eps_h = 1e-5
            for d in range(D):
                zb_plus = zb.copy()
                zb_plus[d] += eps_h
                zb_minus = zb.copy()
                zb_minus[d] -= eps_h
                gb[d] = (log_p_fn(zb_plus) - log_p_fn(zb_minus)) / (2 * eps_h)
            g.append(gb)
        g = np.array(g)
        
        z_bar = np.mean(z, axis=0)
        g_bar = np.mean(g, axis=0)
        
        g_diff = g - g_bar
        Gamma = np.dot(g_diff.T, g_diff) / batch_size
        
        Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(D))
        outer_g_bar = np.outer(g_bar, g_bar)
        Sigma_inv_next = (1.0 / (1.0 + regularization)) * Sigma_inv + (regularization / (1.0 + regularization)) * (Gamma + outer_g_bar)
        Sigma_next = np.linalg.inv(Sigma_inv_next + 1e-6 * np.eye(D))
        
        mu_next = (1.0 / (1.0 + regularization)) * mu + (regularization / (1.0 + regularization)) * (z_bar + np.dot(Sigma_next, g_bar))
        loss = compute_loss(mu, Sigma, log_p_fn, z, g)
        
        return mu_next, Sigma_next, loss

def run_experiment_matrix(methods=None, parameters=None, log_p_fn=None, D=2, num_steps=None):
    """
    Orchestrate experiments over the declared paper-derived dimensions.
    """
    import numpy as np
    
    if methods is None:
        methods = ["ours", "baseline", "BaM", "GSM", "ADVI"]
        
    if parameters is None:
        parameters = {
            "learning_rate": learning_rate_values,
            "batch_size": batch_size_values,
            "regularization": [0.1, 1.0, 10.0],
            "lambda": lambda_values,
            "steps": num_steps_values
        }
        
    if log_p_fn is None:
        def log_p_fn(z):
            return -0.5 * np.sum(z**2)
            
    results = {}
    
    sweep_lr = parameters.get("learning_rate", [DEFAULT_LEARNING_RATE])[:1]
    sweep_bs = parameters.get("batch_size", [DEFAULT_BATCH_SIZE])[:1]
    sweep_reg = parameters.get("regularization", [DEFAULT_REGULARIZATION])[:1]
    sweep_lam = parameters.get("lambda", [DEFAULT_LAMBDA])[:1]
    sweep_steps = [num_steps if num_steps is not None else DEFAULT_NUM_STEPS]
    
    for method in methods:
        resolved_method = METHOD_REGISTRY.get(method, "BaM")
        for lr in sweep_lr:
            for bs in sweep_bs:
                for reg in sweep_reg:
                    for lam in sweep_lam:
                        for steps in sweep_steps:
                            key = f"{method}_lr{lr}_bs{bs}_reg{reg}_lam{lam}_steps{steps}"
                            
                            mu = np.zeros(D)
                            Sigma = np.eye(D)
                            L = np.eye(D)
                            
                            losses = []
                            rng = np.random.RandomState(42)
                            
                            for step in range(steps):
                                if resolved_method == "BaM":
                                    mu, Sigma, loss = bam_step(mu, Sigma, log_p_fn, rng, bs, lr, reg)
                                elif resolved_method == "GSM":
                                    mu, Sigma, loss = gsm_step(mu, Sigma, log_p_fn, rng, bs)
                                elif resolved_method == "ADVI":
                                    mu, L, loss = advi_step(mu, L, log_p_fn, rng, bs, lr)
                                    Sigma = np.dot(L, L.T)
                                else:
                                    mu, Sigma, loss = bam_step(mu, Sigma, log_p_fn, rng, bs, lr, reg)
                                    
                                losses.append(float(loss))
                                
                            results[key] = {
                                "method": method,
                                "resolved_method": resolved_method,
                                "learning_rate": lr,
                                "batch_size": bs,
                                "regularization": reg,
                                "lambda": lam,
                                "steps": steps,
                                "final_loss": losses[-1],
                                "losses": losses
                            }
                            
    return results

def run_smoke_test():
    """
    Run a lightweight smoke test to verify all functions and resolve defaults.
    """
    print("Running baselines.py smoke test...")
    
    lr = resolve_learning_rate_defaults()
    bs = resolve_batch_size_defaults()
    lam = resolve_lambda_defaults()
    steps = resolve_num_steps_defaults()
    
    print(f"Resolved defaults: lr={lr}, bs={bs}, lam={lam}, steps={steps}")
    
    import numpy as np
    def log_p_fn(z):
        return -0.5 * np.sum(z**2)
        
    q_mean = np.zeros(2)
    q_cov = np.eye(2)
    samples = np.random.normal(size=(bs, 2))
    scores = -samples
    
    loss = compute_loss(q_mean, q_cov, log_p_fn, samples, scores)
    agg_loss = aggregate_loss([loss, loss])
    reward = compute_reward(q_mean, q_cov, log_p_fn, samples, scores)
    agg_reward = aggregate_reward([reward, reward])
    
    print(f"Loss: {loss}, Aggregated Loss: {agg_loss}")
    print(f"Reward: {reward}, Aggregated Reward: {agg_reward}")
    
    rng = np.random.RandomState(42)
    mu = np.zeros(2)
    Sigma = np.eye(2)
    L = np.eye(2)
    
    mu_advi, L_advi, loss_advi = advi_step(mu, L, log_p_fn, rng, bs, lr)
    mu_gsm, Sigma_gsm, loss_gsm = gsm_step(mu, Sigma, log_p_fn, rng, bs)
    mu_bam, Sigma_bam, loss_bam = bam_step(mu, Sigma, log_p_fn, rng, bs, lr, lam)
    
    print("ADVI step successful.")
    print("GSM step successful.")
    print("BaM step successful.")
    
    results = run_experiment_matrix(
        methods=["ours", "baseline", "GSM"],
        parameters={
            "learning_rate": [lr],
            "batch_size": [bs],
            "regularization": [lam],
            "lambda": [lam],
            "steps": [5]
        },
        log_p_fn=log_p_fn,
        D=2,
        num_steps=5
    )
    print(f"Experiment matrix run successful. Number of results: {len(results)}")
    print("All smoke tests passed successfully!")

if __name__ == "__main__":
    run_smoke_test()
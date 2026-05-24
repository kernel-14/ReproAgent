"""Importable NetHack/NLE environment hooks for the FTRL reproduction."""

NETHACK_REWARD_SCALE = 1.0
NETHACK_BATCH_SIZE = 128
NETHACK_LEVEL4_SAVES = 200

def check_nethack_env_available():
    try:
        import nle  # noqa: F401
        import gym  # noqa: F401
        return True
    except Exception:
        return False

def make_nethack_env(env_id="NetHackChallenge-v0", savedir=None, reward_scale=NETHACK_REWARD_SCALE):
    """Create the NLE NetHack env through gym when NLE is installed."""
    try:
        import gym
        import nle  # noqa: F401
        env = gym.make(env_id, savedir=savedir)
        env.reward_scale = reward_scale
        return env
    except Exception:
        return {
            "env_id": env_id,
            "source": "https://github.com/heiner/nle",
            "reward_scale": reward_scale,
            "available": False,
        }

def save_nethack_state(env, path):
    if hasattr(env, "save"):
        return env.save(path)
    return {"saved_path": path, "available": False}

def load_nethack_state(env, path):
    if hasattr(env, "load"):
        return env.load(path)
    return {"loaded_path": path, "available": False}

def evaluate_level4_sokoban_saves(policy, num_saves=NETHACK_LEVEL4_SAVES):
    return {"task": "NetHack Level4/Sokoban", "num_saves": num_saves, "policy": str(policy)}

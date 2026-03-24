"""
smoke_test.py  –  Verify the environment and reward shaping work
                  without needing GloVe or SB3.

Run from repo root:
    python scripts/smoke_test.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from environment    import CodenamesEnv
from reward_shaping import PotentialShapingWrapper
from guesser        import GreedySpymaster


def make_random_sim(vocab_size=200, seed=42):
    rng  = np.random.default_rng(seed)
    vecs = rng.standard_normal((vocab_size, 50)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    return (vecs @ vecs.T).astype(np.float32)


def test_env_basic():
    V            = 200
    sim_matrix   = make_random_sim(V)
    board_indices = np.arange(25, dtype=np.int32)

    env = CodenamesEnv(sim_matrix, board_indices, vocab_size=V, seed=0)

    obs, info = env.reset()
    assert "observation"   in obs
    assert "achieved_goal" in obs
    assert "desired_goal"  in obs
    assert obs["observation"].shape == (675,), obs["observation"].shape

    # Random action
    action = env.action_space.sample()
    obs2, reward, term, trunc, info2 = env.step(action)
    assert isinstance(reward, float)
    print(f"  [env_basic] reward={reward:.2f}  terminated={term}  "
          f"guessed={info2['guessed']}")
    print("  PASS")


def test_reward_shaping():
    V             = 200
    sim_matrix    = make_random_sim(V)
    board_indices = np.arange(25, dtype=np.int32)

    base_env    = CodenamesEnv(sim_matrix, board_indices, vocab_size=V, seed=1)
    shaped_env  = PotentialShapingWrapper(base_env, gamma=0.99, scale=1.0)

    obs, _ = shaped_env.reset()
    action  = shaped_env.action_space.sample()
    _, reward, _, _, info = shaped_env.step(action)

    assert "shaping_bonus" in info
    assert "potential"     in info
    print(f"  [shaping] reward={reward:.3f}  bonus={info['shaping_bonus']:.3f}  "
          f"potential={info['potential']:.3f}")
    print("  PASS")


def test_greedy_spymaster():
    V             = 200
    sim_matrix    = make_random_sim(V)
    board_indices = np.arange(25, dtype=np.int32)

    env     = CodenamesEnv(sim_matrix, board_indices, vocab_size=V, seed=2)
    greedy  = GreedySpymaster(sim_matrix, board_indices, vocab_size=V, top_k=50)

    obs, _ = env.reset()
    wins   = 0

    for episode in range(10):
        obs, _ = env.reset()
        done   = False
        while not done:
            clue, count = greedy.select_action(env.labels, env.remaining_mask)
            action = greedy.action_to_flat(clue, count)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
        fr = np.sum((env.labels == 0) & env.remaining_mask)
        if fr == 0:
            wins += 1

    print(f"  [greedy] win rate over 10 episodes: {wins}/10")
    print("  PASS")


def test_her_compute_reward():
    V             = 200
    sim_matrix    = make_random_sim(V)
    board_indices = np.arange(25, dtype=np.int32)
    env           = CodenamesEnv(sim_matrix, board_indices, vocab_size=V)

    # Achieved = desired → reward should be 0
    g = np.zeros(25, dtype=np.float32)
    g[[0, 2]] = 1.0
    r = env.compute_reward(g.copy(), g.copy(), {})
    assert r == 0.0, r

    # Achieved ≠ desired → reward should be -1
    achieved = np.zeros(25, dtype=np.float32)
    achieved[0] = 1.0
    r2 = env.compute_reward(achieved, g.copy(), {})
    assert r2 == -1.0, r2

    print("  [HER compute_reward] PASS")


if __name__ == "__main__":
    print("Running smoke tests…\n")
    test_env_basic()
    test_reward_shaping()
    test_greedy_spymaster()
    test_her_compute_reward()
    print("\nAll tests passed ✓")

import gymnasium
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision = 3)

TARGET_EPS = 0.01
GAMMA = 0.9
Q_INIT = 0.0
EPS_INIT = 1
MIN_EPS = 0.01
MAX_STEPS = 2000
HORIZON = 10

episodes_per_iteration = [1, 10, 50]
decays = [1, 2, 5]
seeds = np.arange(50)


env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.unwrapped.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

P = P * (1.0 - T[..., None])  # next state probability for terminal transitions is 0

OPTIMAL_PI = np.zeros((n_states, n_actions))
OPTIMAL_PI[0, 1] = 1
OPTIMAL_PI[3, 1] = 1
OPTIMAL_PI[1, 2] = 1
OPTIMAL_PI[4, 2] = 1
OPTIMAL_PI[6, 2] = 1
OPTIMAL_PI[7, 2] = 1
OPTIMAL_PI[5, 3] = 1
OPTIMAL_PI[8, 3] = 1
OPTIMAL_PI[2, 4] = 1

def bellman_q(pi, gamma):
    I = np.eye(n_states * n_actions)
    P_under_pi = (
        P[..., None] * pi[None, None]
    ).reshape(n_states * n_actions, n_states * n_actions)
    return (
        R.ravel() * np.linalg.inv(I - gamma * P_under_pi)
    ).sum(-1).reshape(n_states, n_actions)

def episode(env, Q, eps, seed):
    data = dict()
    data["s"] = []
    data["a"] = []
    data["r"] = []
    data['p'] = []
    s, _ = env.reset(seed=seed)
    done = False
    while not done:
        a, p = eps_greedy_action(Q, s, eps)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)
        data['p'].append(p)
        s = s_next
    return data


def eps_greedy_action(Q, s, eps):
    Q_s = Q[s]
    if eps > np.random.rand():
        action = np.random.randint(len(Q_s))
        prob = eps / n_actions
    else:
        action = np.argmax(Q_s)
        prob = 1 - eps
    return action, prob

def eps_greedy_policy(Q, eps):

    optimal_actions = np.argmax(Q, -1)
    PI = np.zeros((n_states, n_actions)) + (eps/n_actions)
    PI[np.arange(n_states), optimal_actions] += 1 - eps
    
    return PI
    
def monte_carlo(env, Q, eps_decay, episodes_per_iteration, use_is, seed):

    total_steps = 0
    eps = EPS_INIT
    
    C = np.zeros((n_states, n_actions))
    bellman_errors = []

    if use_is:
        PI = eps_greedy_policy(Q, TARGET_EPS)
    else:
        PI = eps_greedy_policy(Q, eps)

    True_Q = bellman_q(PI, GAMMA)

    be = np.sum(np.abs(True_Q - Q))
    bellman_errors.append(be)

    
    while total_steps < MAX_STEPS:
                    
        states, actions, rewards, probs = [], [], [], []
        iteration_steps = 0
        
        for epi in range(episodes_per_iteration):
            
            data = episode(env, Q, eps, seed = seed)

            iteration_steps += len(data['s'])
            
            states.append(data['s'])
            actions.append(data['a'])
            rewards.append(data['r'])
            probs.append(data['p'])
            
            eps = max(eps - (len(data['s']) * eps_decay / MAX_STEPS), MIN_EPS)
                
        for epi in range(episodes_per_iteration):
            
            G = 0
            W = 1
            
            for t in range(len(states[epi]))[::-1]:
                
                s, a, r, p = states[epi][t], actions[epi][t], rewards[epi][t], probs[epi][t]
                
                G = (GAMMA * G) + r
                
                C[s, a] += W
                Q[s, a] += ((W / C[s,a]) * (G - Q[s, a]))
                
                if use_is:
                    num = TARGET_EPS / n_actions
                    if a == np.argmax(Q[s]):
                        num += 1 - TARGET_EPS                 
                    W = W * (num / p)
        
        if use_is:
            PI = eps_greedy_policy(Q, TARGET_EPS)
        else:
            PI = eps_greedy_policy(Q, eps)

        True_Q = bellman_q(PI, GAMMA)
        
        be = np.sum(np.abs(True_Q - Q))
        for _ in range(iteration_steps):
            bellman_errors.append(be)
        
        total_steps += iteration_steps

    return Q, bellman_errors[:MAX_STEPS]

def error_shade_plot(ax, data, stepsize, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())


results = np.zeros((
    len(episodes_per_iteration),
    len(decays),
    len(seeds),
    MAX_STEPS,
))

fig, axs = plt.subplots(1, 2)
plt.ion()
plt.show()

use_is = True  # repeat with True
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Absolute Bellman Error")
    env = gymnasium.make(
        "Gym-Gridworlds/Penalty-3x3-v0",
        max_episode_steps=HORIZON,
        reward_noise_std=reward_noise_std,
    )
    for j, episodes in enumerate(episodes_per_iteration):
        for k, decay in enumerate(decays):
            for seed in seeds:
                # print(f"episodes: {episodes}; decay: {decay}; seed: {seed}")
                np.random.seed(seed)
                Q = np.zeros((n_states, n_actions)) + Q_INIT                
                Q, be = monte_carlo(env, Q, decay, episodes, use_is, int(seed))
                # print(f"Bellman Error: {be[-1]}")
                # print()
                
                results[j, k, seed] = be
            error_shade_plot(
                ax,
                results[j, k],
                stepsize=1,
                label=f"Episodes: {episodes}, Decay: {decay}",
            )
            ax.legend()
            plt.draw()
            plt.pause(0.001)

plt.ioff()
plt.show()
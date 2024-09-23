import gymnasium
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)


HORIZON = 10
GAMMA = 0.99
MAX_STEPS = 10000
EPS_INIT = 1.0
LR_INIT = 0.1


init_values = [-10, 0.0,  10]
algs = ["QL", "SARSA", "Exp_SARSA"]
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

def bellman_q(pi, gamma, max_iter=1000):
    delta = np.inf
    iter = 0
    Q = np.zeros((n_states, n_actions))
    be = np.zeros((max_iter))
    while delta > 1e-5 and iter < max_iter:
        Q_new = R + (np.dot(P, gamma * (Q * pi)).sum(-1))
        delta = np.abs(Q_new - Q).sum()
        be[iter] = delta
        Q = Q_new
        iter += 1
    return Q

def eps_greedy_probs(Q, s = None, eps = 0):
    if s is not None:
        Q_s = Q[s]
        pi = np.zeros(n_actions) + (eps/n_actions)
        best_action = np.argmax(Q_s)
        pi[best_action] += 1 - eps
    else:
        pi = np.zeros((n_states, n_actions)) + (eps/n_actions)
        optimal_actions = np.argmax(Q, -1)
        pi[np.arange(n_states), optimal_actions] += 1 - eps
    return pi

def eps_greedy_action(Q, s, eps):
    Q_s = Q[s]
    if eps > np.random.rand():
        action = np.random.randint(len(Q_s))
    else:
        action = np.argmax(Q_s)
    return action


def expected_return(env, Q, gamma, episodes=10):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            a = eps_greedy_action(Q, s, 0.0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()

def td(env, env_eval, Q, alg, seed):
    
    tot_steps = 0
    
    all_td_errors = []
    all_eval_returns = []
    all_bellman_errors = []
    
    eps = EPS_INIT
    lr = LR_INIT
    
    while tot_steps < MAX_STEPS:
        
        s, _ = env.reset(seed = seed)
        done = False
        steps = 0
        
        a = eps_greedy_action(Q, s, eps)
        
        while not done:
            
            # algs = ["QL", "SARSA", "Exp_SARSA"]
            
            steps += 1
            tot_steps += 1
            
            if alg in ["QL", "Exp_SARSA"]:
                a = eps_greedy_action(Q, s, eps)
                
            s_next, r, t1, t2, _ = env.step(a)
            
            eps = max(eps - (1.0 / MAX_STEPS), 0.01)
            lr = max(lr - (0.1 / MAX_STEPS), 0.001)
                        
            if alg == "QL":
                a_next = np.argmax(Q[s_next])
                boostrap = Q[s_next, a_next]
            
            elif alg == "SARSA":
                a_next = eps_greedy_action(Q, s_next, eps)                    
                boostrap = Q[s_next, a_next]
                
            else:
                a_next = np.argmax(Q[s_next])
                probs = eps_greedy_probs(Q, s_next, eps)
                boostrap = np.sum([probs[i] * Q[s_next, i] for i in range(n_actions)])
            
            td_error = (r + (GAMMA * boostrap * (1 - t1)) - Q[s, a])
            Q[s, a] += lr * td_error
            all_td_errors.append(np.abs(td_error))
            
            if tot_steps % 100 == 0:
                
                if alg in ["SARSA", "Exp_SARSA"]:
                    pi = eps_greedy_probs(Q, None, eps)
                else:
                    pi = eps_greedy_probs(Q, None, 0)
                
                true_q = bellman_q(pi, GAMMA)
                bellman_error = np.mean(np.abs(true_q - Q))
                all_bellman_errors.append(bellman_error)

                ret = expected_return(env_eval, Q, GAMMA)
                all_eval_returns.append(ret)                
            
            s = s_next
            a = a_next
            
            done = t1 or t2
            

    return Q, np.array(all_bellman_errors), np.array(all_td_errors)[:MAX_STEPS], np.array(all_eval_returns)



def double_td(env, env_eval, Q, alg, seed):
    
    Q1, Q2 = Q.copy(), Q.copy()
    
    tot_steps = 0
    
    all_td_errors = []
    all_eval_returns = []
    all_bellman_errors = []
    
    eps = EPS_INIT
    lr = LR_INIT
    
    while tot_steps < MAX_STEPS:
        
        s, _ = env.reset(seed = seed)
        done = False
        steps = 0
        
        a = eps_greedy_action(Q1 + Q2, s, eps)
        
        while not done:
            
            # algs = ["QL", "SARSA", "Exp_SARSA"]
            
            steps += 1
            tot_steps += 1
            
            if alg in ["QL", "Exp_SARSA"]:
                a = eps_greedy_action(Q1+Q2, s, eps)
                
            s_next, r, t1, t2, _ = env.step(a)
            
            eps = max(eps - (1.0 / MAX_STEPS), 0.01)
            lr = max(lr - (0.1 / MAX_STEPS), 0.001)
            
            
            if alg == "QL":
                if np.random.rand() > 0.5:
                    a_next = np.argmax(Q2[s_next])
                    td_error = (r + (GAMMA * Q1[s_next, a_next] * (1 - t1)) - Q2[s, a])
                    Q2[s, a] += lr * td_error
                else:
                    a_next = np.argmax(Q1[s_next])
                    td_error = (r + (GAMMA * Q2[s_next, a_next] * (1 - t1)) - Q1[s, a])
                    Q1[s, a] += lr * td_error

            elif alg == "SARSA":
                if np.random.rand() > 0.5:
                    a_next = eps_greedy_action(Q2, s_next, eps)                    
                    td_error = (r + (GAMMA * Q1[s_next, a_next] * (1 - t1)) - Q2[s, a])
                    Q2[s, a] += lr * td_error
                else:
                    a_next = eps_greedy_action(Q1, s_next, eps)                    
                    td_error = (r + (GAMMA * Q2[s_next, a_next] * (1 - t1)) - Q1[s, a])
                    Q1[s, a] += lr * td_error
            else:
                a_next = np.argmax(Q1[s_next] + Q2[s_next])
                if np.random.rand() > 0.5:
                    probs = eps_greedy_probs(Q2, s_next, eps)
                    boostrap = np.sum([probs[i] * Q1[s_next, i] for i in range(n_actions)])
                    td_error = (r + (GAMMA * boostrap * (1 - t1)) - Q2[s, a])
                    Q2[s, a] += lr * td_error
                else:
                    probs = eps_greedy_probs(Q1, s_next, eps)
                    boostrap = np.sum([probs[i] * Q2[s_next, i] for i in range(n_actions)])
                    td_error = (r + (GAMMA * boostrap * (1 - t1)) - Q1[s, a])
                    Q1[s, a] += lr * td_error
            
            all_td_errors.append(np.abs(td_error))
            
            if tot_steps % 100 == 0:
                
                if alg in ["SARSA", "Exp_SARSA"]:
                    pi = eps_greedy_probs(Q1+Q2, None, eps)
                else:
                    pi = eps_greedy_probs(Q1+Q2, None, 0)
                
                true_q = bellman_q(pi, GAMMA)
                bellman_error = np.mean(np.abs(true_q - ((Q1+Q2)/2)))
                all_bellman_errors.append(bellman_error)

                ret = expected_return(env_eval, (Q1+Q2)/2, GAMMA)
                all_eval_returns.append(ret)                
            
            s = s_next
            a = a_next
            
            done = t1 or t2
            

    return (Q1+Q2)/2, np.array(all_bellman_errors), np.array(all_td_errors)[:MAX_STEPS], np.array(all_eval_returns)


# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha = 0.1, linewidth=0.0, color=line.get_color())


results_be = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    MAX_STEPS // 100,
))
results_tde = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    MAX_STEPS,
))
results_exp_ret = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    MAX_STEPS // 100,
))

fig, axs = plt.subplots(1, 3)
plt.ion()
plt.show()

# reward_noise_std = 3.0
reward_noise_std = 0.0

for ax in axs:
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")

env = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=HORIZON,
    reward_noise_std=reward_noise_std,
)

env_eval = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=HORIZON,
)

for i, init_value in enumerate(init_values):
    for j, alg in enumerate(algs):
        for seed in seeds:
            np.random.seed(seed)
            Q = np.zeros((n_states, n_actions)) + init_value
            Q, be, tde, exp_ret = double_td(env, env_eval, Q, alg, int(seed))
            # Q, be, tde, exp_ret = td(env, env_eval, Q, alg, int(seed))
            results_be[i, j, seed] = be
            results_tde[i, j, seed] = tde
            results_exp_ret[i, j, seed] = exp_ret
        label = f"$Q_0$: {init_value}, Alg: {alg}"
        axs[0].set_title("TD Error")
        error_shade_plot(
            axs[0],
            results_tde[i, j],
            stepsize=1,
            smoothing_window=50,
            label=label,
        )
        axs[0].legend()
        axs[0].set_ylim([0, 5])
        axs[1].set_title("Bellman Error")
        error_shade_plot(
            axs[1],
            results_be[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[1].legend()
        axs[1].set_ylim([0, 50])
        axs[2].set_title("Expected Return")
        error_shade_plot(
            axs[2],
            results_exp_ret[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[2].legend()
        axs[2].set_ylim([-5, 1])
        plt.draw()
        plt.pause(0.001)

plt.ioff()
plt.show()




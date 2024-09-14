import gymnasium
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
env.reset()

n_states = env.observation_space.n
n_actions = env.action_space.n


R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated


PI = np.zeros((n_states, n_actions))

PI[0, 1] = 1
PI[3, 1] = 1
PI[1, 2] = 1
PI[4, 2] = 1
PI[6, 2] = 1
PI[7, 2] = 1
PI[5, 3] = 1
PI[8, 3] = 1
PI[2, 4] = 1


def bellman_update(V, Q, PI, P, R, T, gamma):

    PI_R = np.multiply(PI, R).sum(-1) # Shape: (S,)

    P_V = np.dot(P, V) # Shape - (S, A)
    P_V = np.multiply(P_V, (1 - T))
    PI_P_V = np.multiply(PI, P_V).sum(-1) # Shape: (S,)
    
    NEW_V = PI_R + (gamma * PI_P_V)
    NEW_Q = R + (gamma * P_V)
    
    V_bellman_error = np.sum(np.abs(V - NEW_V))
    Q_bellman_error = np.sum(np.abs(Q - NEW_Q))
    
    return NEW_V, NEW_Q, V_bellman_error, Q_bellman_error



value_init = -10
gamma = 0.99


for value_init in [-10, 0, 10]:
    for gamma in [0.01, 0.5, 0.99]:
        
        matplotlib.rcdefaults()
        fig, ax = plt.subplots()
        
        print(f"Running V: {value_init}; Gamma: {gamma}")
        

        V = np.zeros(n_states) + value_init
        Q = np.zeros((n_states, n_actions)) + value_init

        V_errors = []
        Q_errors = []
        
        total_iters = 0
        
        V_error = 100
        while V_error > 1e-4:
            V, Q, V_error, Q_error = bellman_update(V, Q, PI, P, R, T, gamma)
            V_errors.append(V_error)
            Q_errors.append(Q_error)
            total_iters += 1


        grid_size = int(np.sqrt(n_states))
        sns.heatmap(V.reshape((grid_size, grid_size)), annot = True)
        plt.title(f"VALUE FUNCTION (V/Q_Init: {value_init}; Gamma: {gamma}) -> Iters {total_iters}")
        plt.savefig(f"Results/V_{value_init}_{gamma}.jpg")
        plt.clf()

        plt.plot(V_errors)
        plt.title(f"ABS V-FUNC ERROR (V/Q_Init: {value_init}; Gamma: {gamma}) -> Iters {total_iters}")
        plt.savefig(f"Results/VE_{value_init}_{gamma}.jpg")
        plt.clf()

        plt.plot(Q_errors)
        plt.title(f"ABS Q-FUNC ERROR (V/Q_Init: {value_init}; Gamma: {gamma}) -> Iters {total_iters}")
        plt.savefig(f"Results/QE_{value_init}_{gamma}.jpg")
        plt.clf()

        fig, axes = plt.subplots(1, n_actions, figsize= ((n_actions * 7, 5)))
        grid_size = int(np.sqrt(n_states))
        action_list = ['LEFT', 'DOWN', 'RIGHT', 'UP', 'STAY']

        for action, ax in enumerate(axes.flatten()):
            sns.heatmap(Q[:, action].reshape((grid_size, grid_size)), annot = True, ax = ax)
            ax.title.set_text(action_list[action])
            
        plt.suptitle(f"ACTION VALUE FUNCTION (V/Q_Init: {value_init}; Gamma: {gamma}) -> Iters {total_iters}")
        plt.savefig(f"Results/Q_{value_init}_{gamma}.jpg")
        plt.clf()
        
    
    
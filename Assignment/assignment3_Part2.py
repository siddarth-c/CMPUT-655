
import gymnasium
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


def random_argmax(a, axis=None, out=None):
        
    max_val = np.max(a, axis=axis, keepdims=True)
    
    max_mask = (a == max_val)
    
    if axis is None:
        max_mask = max_mask.ravel()    
        rand_idx = np.random.choice(np.where(max_mask)[0])
    else:
        rand_idx = np.apply_along_axis(
            lambda x: np.random.choice(np.where(x)[0]),
            axis, max_mask)
    
    if out is not None:
        out[...] = rand_idx
    else:
        out = rand_idx
    
    return out


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


# VALUE_INIT = 10
gamma = 0.99

def reinit(VALUE_INIT):
    
    PI = np.ones((n_states, n_actions)) * (1/n_actions)
    Q = np.zeros((n_states, n_actions)) + VALUE_INIT
    
    return PI, Q


def policy_eval(Q, PI, R, T, gamma, steps = None):
    
    def bellman_update(Q, PI, P, R, T, gamma):        
        
        PI_Q = np.multiply(PI, Q).sum(-1) # Shape: (S,)
        P_PI_Q = np.dot(P, PI_Q) # Shape: (S, A)
        P_PI_Q = np.multiply(P_PI_Q, (1 - T))
        
        NEW_Q = R + (gamma * P_PI_Q)
        
        bellman_error = np.sum(np.abs(Q - NEW_Q))
        
        return NEW_Q, bellman_error
    
    bellman_errors = []
    
    if steps is None:
        error = 100
        while error > 1e-4:
            Q, error = bellman_update(Q, PI, P, R, T, gamma)
            bellman_errors.append(error)
    else:
        for _ in range(steps):
            Q, error = bellman_update(Q, PI, P, R, T, gamma)
            bellman_errors.append(error)
            
    return Q, bellman_errors



def policy_imp(Q, PI, R, T, gamma, random_idx = False):
    
    PI_old = PI.copy()
    
    if random_idx:
        optimal_actions = random_argmax(Q, -1)
    else:
        optimal_actions = np.argmax(Q, -1)

    PI = np.zeros((n_states, n_actions))

    PI[np.arange(n_states), optimal_actions] = 1
    
    complete = np.allclose(PI, PI_old)
    return PI, complete


def policy_iter(Q, PI, R, T, gamma):
    
    complete = False
    all_bellman_errors = []
    update_points = []
    
    while not complete:
    
        Q, bellman_errors = policy_eval(Q, PI, R, T, gamma)
        PI, complete = policy_imp(Q, PI, R, T, gamma)
        
        all_bellman_errors.append([e for e in bellman_errors])
        update_points.append(len(bellman_errors))

    all_bellman_errors = list(itertools.chain(*all_bellman_errors))
    
    return Q, PI, all_bellman_errors, update_points


def value_iter(Q, R, T, gamma):
    
    def bellman_update(Q, P, R, T, gamma):
        
        P_Q = np.dot(P, Q) # Shape - (S, A, A)
        BEST_P_Q = np.max(P_Q, -1) # Shape - (S, A)
        BEST_P_Q = np.multiply(BEST_P_Q, (1 - T))
        
        NEW_Q = R + (gamma * BEST_P_Q)
        
        bellman_error = np.sum(np.abs(Q - NEW_Q))
        
        return NEW_Q, bellman_error
    
    bellman_errors = []
    error = 100
    
    while error > 1e-4:
        Q, error = bellman_update(Q, P, R, T, gamma)
        bellman_errors.append(error)

    optimal_actions = np.argmax(Q, -1)

    PI = np.zeros((n_states, n_actions))
    PI[np.arange(n_states), optimal_actions] = 1

        
    return Q, PI, bellman_errors




def GPI(Q, PI, R, T, gamma, eval_steps=None):
    complete = False
    iteration = 0
    all_bellman_errors = []

    while not complete:
        iteration += 1
        Q, bellman_error = policy_eval(Q, PI, R, T, gamma, steps = eval_steps)
        PI, complete = policy_imp(Q, PI, R, T, gamma, random_idx = True)
        
        all_bellman_errors.append([e for e in bellman_error])

    all_bellman_errors = list(itertools.chain(*all_bellman_errors))


    return Q, PI, all_bellman_errors


for VALUE_INIT in [-100, -10, -5, 0, 5, 10, 100]:
    
    print("VALUE INIT - ", VALUE_INIT)
    

    PI_PI, PI_Q = reinit(VALUE_INIT)

    PI_Q, PI_PI, PI_Bellman_Errors, update_points = policy_iter(PI_Q, PI_PI, R, T, gamma)
    PI_STEPS = len(PI_Bellman_Errors)

    print('POLICY ITERATION SUCCESS - ', np.allclose(PI_PI, OPTIMAL_PI))


    VI_PI, VI_Q = reinit(VALUE_INIT)


    VI_Q, VI_PI, VI_Bellman_Errors = value_iter(VI_Q, R, T, gamma)
    VI_STEPS = len(VI_Bellman_Errors)

    print('VALUE ITERATION SUCCESS - ', np.allclose(VI_PI, OPTIMAL_PI))
    
    
    GPI_STEPS = 0
    
    for _ in range(5):
        
        GPI_PI, GPI_Q = reinit(VALUE_INIT)

        GPI_Q, GPI_PI, GPI_Bellman_Errors = GPI(GPI_Q, GPI_PI, R, T, gamma, 5)
        GPI_STEPS += len(GPI_Bellman_Errors)

        print('GENERALIZED POLICY ITERATION SUCCESS - ', np.allclose(GPI_PI, OPTIMAL_PI))
        
    GPI_STEPS = GPI_STEPS //5
    
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 6))
    ax1, ax2, ax3 = axes.flatten()

    ax1.plot(PI_Bellman_Errors, 'b')
    ax1.set_title(f'Policy Iteration: #Iters {(PI_STEPS)}')
    ax2.plot(VI_Bellman_Errors, 'g')
    ax2.set_title(f'Value Iteration: #Iters {(VI_STEPS)}')
    ax3.plot(GPI_Bellman_Errors, 'k')
    ax3.set_title(f'General Policy Iteration: #Iters {(GPI_STEPS)}')

    # plt.show()

    plt.suptitle(f"ACTION VALUE INIT: {VALUE_INIT}", fontweight = 'bold')
    plt.savefig(f"Results/Q_{VALUE_INIT}.jpg")

    plt.clf()
    
    
    print()
    print()
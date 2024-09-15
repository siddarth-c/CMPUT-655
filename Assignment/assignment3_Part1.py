
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
SMALL_NUMBER = 3e-5

def reinit(VALUE_INIT):
    
    PI = np.ones((n_states, n_actions)) * (1/n_actions)
    V = np.zeros(n_states) + VALUE_INIT
    
    return PI, V


def policy_eval(V, PI, R, T, gamma, steps = None):
    
    def bellman_update(V, PI, P, R, T, gamma):

        PI_R = np.multiply(PI, R).sum(-1) # Shape: (S,)

        P_V = np.dot(P, V) # Shape - (S, A)
        P_V = np.multiply(P_V, (1 - T))
        PI_P_V = np.multiply(PI, P_V).sum(-1) # Shape: (S,)
        
        NEW_V = PI_R + (gamma * PI_P_V)
        
        bellman_error = np.sum(np.abs(V - NEW_V))
        
        return NEW_V, bellman_error
    
    bellman_errors = []
    
    if steps is None:
        error = 100
        while error > SMALL_NUMBER:
            V, error = bellman_update(V, PI, P, R, T, gamma)
            bellman_errors.append(error)
    else:
        for _ in range(steps):
            V, error = bellman_update(V, PI, P, R, T, gamma)
            bellman_errors.append(error)
            
            
    return V, bellman_errors



def policy_imp(V, PI, R, T, gamma, random_idx = False):
    
    PI_old = PI.copy()

    P_V = np.dot(P, V) # Shape - (S, A)
    P_V = np.multiply(P_V, (1 - T))
    
    if random_idx:
        optimal_actions = random_argmax(R + (gamma * P_V), -1)
    else:
        optimal_actions = np.argmax(R + (gamma * P_V), -1)
        
    PI = np.zeros((n_states, n_actions))

    PI[np.arange(n_states), optimal_actions] = 1
    
    complete = np.allclose(PI, PI_old)
    return PI, complete


def policy_iter(V, PI, R, T, gamma):
    
    complete = False
    all_bellman_errors = []
    update_points = []
    
    while not complete:
    
        V, bellman_errors = policy_eval(V, PI, R, T, gamma)
        PI, complete = policy_imp(V, PI, R, T, gamma)
        
        all_bellman_errors.append([e for e in bellman_errors])
        update_points.append(len(bellman_errors))

    all_bellman_errors = list(itertools.chain(*all_bellman_errors))
    
    return V, PI, all_bellman_errors, update_points

def value_iter(V, R, T, gamma):
    
    def bellman_update(V, P, R, T, gamma):

        P_V = np.dot(P, V) # Shape - (S, A)
        P_V = np.multiply(P_V, (1 - T))
        
        NEW_V = np.max(R + (gamma * P_V), -1)
        
        bellman_error = np.sum(np.abs(V - NEW_V))
        
        return NEW_V, bellman_error
    
    bellman_errors = []
    error = 100
    
    while error > SMALL_NUMBER:
        V, error = bellman_update(V, P, R, T, gamma)
        bellman_errors.append(error)

    P_V = np.dot(P, V) # Shape - (S, A)
    P_V = np.multiply(P_V, (1 - T))
    optimal_actions = np.argmax(R + (gamma * P_V), -1)

    PI = np.zeros((n_states, n_actions))
    PI[np.arange(n_states), optimal_actions] = 1

        
    return V, PI, bellman_errors




def GPI(V, PI, R, T, gamma, eval_steps=None):
    complete = False
    iteration = 0
    all_bellman_errors = []

    while not complete:
        iteration += 1
        V, bellman_error = policy_eval(V, PI, R, T, gamma, steps = eval_steps)
        PI, complete = policy_imp(V, PI, R, T, gamma, random_idx = True)
        
        all_bellman_errors.append([e for e in bellman_error])

    all_bellman_errors = list(itertools.chain(*all_bellman_errors))


    return V, PI, all_bellman_errors


for VALUE_INIT in [-100, -10, -5, 0, 5, 10, 100]:
    
    print("VALUE INIT - ", VALUE_INIT)
    

    PI_PI, PI_V = reinit(VALUE_INIT)

    PI_V, PI_PI, PI_Bellman_Errors, update_points = policy_iter(PI_V, PI_PI, R, T, gamma)
    PI_STEPS = len(PI_Bellman_Errors)

    print('POLICY ITERATION SUCCESS - ', np.allclose(PI_PI, OPTIMAL_PI))


    VI_PI, VI_V = reinit(VALUE_INIT)


    VI_V, VI_PI, VI_Bellman_Errors = value_iter(VI_V, R, T, gamma)
    VI_STEPS = len(VI_Bellman_Errors)

    print('VALUE ITERATION SUCCESS - ', np.allclose(VI_PI, OPTIMAL_PI))
    
    
    GPI_STEPS = 0
    
    for _ in range(5):

        GPI_PI, GPI_V = reinit(VALUE_INIT)

        GPI_V, GPI_PI, GPI_Bellman_Errors = GPI(GPI_V, GPI_PI, R, T, gamma, 5)
        GPI_STEPS += len(GPI_Bellman_Errors)

        print('GENERALIZED POLICY ITERATION SUCCESS - ', np.allclose(GPI_PI, OPTIMAL_PI))
        
    GPI_STEPS = GPI_STEPS//5

    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 6))
    ax1, ax2, ax3 = axes.flatten()

    ax1.plot(PI_Bellman_Errors, 'b')
    ax1.set_title(f'Policy Iteration - #Iters {(PI_STEPS)}')
    ax2.plot(VI_Bellman_Errors, 'g')
    ax2.set_title(f'Value Iteration - #Iters {(VI_STEPS)}')
    ax3.plot(GPI_Bellman_Errors, 'k')
    ax3.set_title(f'General Policy Iteration - #Iters {(GPI_STEPS)}')

    # plt.show()

    plt.suptitle(f"VALUE INIT: {VALUE_INIT}", fontweight = 'bold')
    plt.savefig(f"Results/V_{VALUE_INIT}.jpg")

    plt.clf()
    
    
    print()
    print()
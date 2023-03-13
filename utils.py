from scipy.stats import norm
import numpy as np
import itertools

def gaussian_expected_top_r_of_n(r, n, mu, sigma, method='approx'):
    '''
    A function to calculate the order statistics of Gaussian
    Two methods implemented, monte-carlo and approximation
    '''
    if r<1:
        return 0
 
    if method == 'monte-carlo':
        Nsamples = 1000
        val = 0
        for i in range(Nsamples):
            scores = mu + sigma*np.random.randn(n,1)
            sorted_scores = np.sort(scores.flatten())
            sorted_scores = sorted_scores[::-1]
            val = val + sorted_scores[r-1]
        val = val/Nsamples
    
    elif method == 'approx':
        alpha = 0.375
        p = (r - alpha)/(n-2*alpha+1)
        val = -norm.ppf(p)
        val = sigma*val+mu
    
    return val

def gaussian_table_filler(N, mean, std, method='approx'):
    '''
    A function to save the order statistics of Gaussian rv.
    '''
    gaussian_table = np.zeros([N+1,N])
    for i in range(N+1):
        for j in range(i):
            gaussian_table[i,j] = gaussian_expected_top_r_of_n(j+1, i, mean, std, method)
    
    return gaussian_table

def get_greedy_reward(gaussian_tables, N, acceptance_ratio, state, action):
    '''
     A function to calculate the greedy reward
    '''
    if any(np.array(state)-np.array(action) < 0) or np.sum(action)!=N*acceptance_ratio:
        greedy_reward = -np.Inf
    else:
        greedy_reward = 0
        for i in range(len(state)):
            greedy_reward += np.sum(gaussian_tables[i][state[i], :int(action[i])])
        greedy_reward = greedy_reward/(N*acceptance_ratio)
    return greedy_reward

def possible_actions(k, N, highs):
    for bars in itertools.combinations(range(N+k-1), k-1):
        vec = np.array([b-a-1 for a, b in zip((-1,) + bars, bars + (N+k-1,))])
        if (vec <= highs).all():
            yield vec

def process_multigroup(N, mean, std, acceptance_ratio, fairness_target, initial_theta, lambda_, eta, num_instances, num_rounds, decay):
    '''
    A function to process the multigroup data

    Parameters
    ----------
    N: int
        Number of applicants
    mean: list
        Mean of the Gaussian distribution of each group
    std: list
        Standard deviation of the Gaussian distribution of each group
    acceptance_ratio: float
        The acceptance ratio of the institution
    fairness_target: list
        The fairness target of each group
    initial_theta: list
        The initial mean parameter of each group
    lambda_: float
        The lambda parameter of the algorithm
    eta: float
        The eta parameter of the algorithm
    num_instances: int
        Number of instances
    num_rounds: int
        Number of rounds
    '''

    num_groups = len(mean)

    applicants = np.zeros([num_instances, num_rounds, num_groups])
    admissions = np.zeros([num_instances, num_rounds, num_groups])
    thetas = np.zeros([num_instances, num_rounds, num_groups])

    gaussian_tables = []
    for i in range(num_groups):
        gaussian_tables.append(gaussian_table_filler(N, mean[i], std[i], method='approx'))

    optimal_actions_table = np.zeros(np.hstack([np.ones(num_groups, dtype=int)*(N+1), num_groups]))

    for iter in range(num_instances):
        print('Processing instance %d' % iter)

        theta = initial_theta
        for t in range(num_rounds):
            # this sampling can be replaced by a more efficient sampling method such as multinomial sampling
            while True:
                state = np.random.poisson(lam=theta*N, size=len(theta))
                if np.sum(state)==N:
                    break

            if not np.any(optimal_actions_table[tuple(state)]):
                action_list = list(possible_actions(num_groups, int(N*acceptance_ratio), state))
                reward_list = np.zeros(len(action_list))
                for i in range(len(action_list)):
                    reward_list[i] = get_greedy_reward(gaussian_tables, N, acceptance_ratio, state, action_list[i])
                    reward_list[i] = reward_list[i] - lambda_*np.square(np.linalg.norm(action_list[i]/(N*acceptance_ratio)-fairness_target))
                optimal_actions_table[tuple(state)] = action_list[np.argmax(reward_list)]
            
            action = optimal_actions_table[tuple(state)]
            theta = np.clip(theta + eta*(action/(N*acceptance_ratio)-state/N),0,1)
            theta = theta/np.sum(theta)

            applicants[iter, t, :] = state/N
            admissions[iter, t, :] = action/(N*acceptance_ratio)
            thetas[iter, t, :] = theta
    
    return applicants, admissions, thetas

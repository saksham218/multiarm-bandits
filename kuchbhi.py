import numpy as np
import matplotlib.pyplot as plt

def std(P):
    x=P
    for i in range(100):
        x=x@P
    return x[0]

def generate_data(num_arms):
    # num_arms: number of arms
    # num_iter: number of iterations
    # return: regret
    num_states =2
    transition_prob=np.zeros((num_arms,num_states,num_states),dtype=np.float32)
    for i in range(num_arms):
        transition_prob[i][0][0]=np.random.uniform(0,1)
        transition_prob[i][0][1]=1-transition_prob[i][0][0]
        transition_prob[i][1][0]=np.random.uniform(0,1)
        transition_prob[i][1][1]=1-transition_prob[i][1][0]

    pi=np.zeros((num_arms,2)) #stationary distribution

    for i in range(num_arms):
        pi[i]=std(transition_prob[i])
    
    mean_reward=np.zeros((num_arms,2))

    for i in range(num_arms):
        mean_reward[i][0]=np.random.random()
        mean_reward[i][1]=np.random.random()
    
    arm_mean_reward=np.zeros((num_arms,1)) #mean reward of each arm

    for i in range(num_arms):
        arm_mean_reward[i]=pi[i]@(mean_reward[i].T)
    
    best_mean_reward=np.max(arm_mean_reward)

    return best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob

def mab_markov_ucb(num_arms,num_iter,L, best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob):
    # num_arms: number of arms
    # num_iter: number of iterations
    # L: a constant
    
    states = np.zeros((num_arms,1),dtype=np.int32)
    T = np.zeros((num_arms,1),dtype=np.int32)   #T[i] is the number of times arm i is pulled
    sum_rewards = np.zeros((num_arms,1))   #sum_rewards[i] is the sum of rewards of arm i
    regret = np.zeros((num_iter,1))
    g = np.zeros((num_arms,1)) #g[i] is the index for arm i
    
    #We start with an exploring start playing al  arms initially in the first iteration

    for arm in range(num_arms):
        reward = np.random.binomial(1,mean_reward[arm][states[arm]])
        sum_rewards[arm] += reward
        T[arm] += 1
        g[arm] = sum_rewards[arm]/T[arm]
        states[arm] = np.random.binomial(1,transition_prob[arm][states[arm]][0])
        regret[0] += best_mean_reward - arm_mean_reward[arm]

    for iter in range(1,num_iter):
        selected_arm = np.argmax(g)
        reward = np.random.binomial(1,mean_reward[selected_arm][states[selected_arm]])
        sum_rewards[selected_arm] += reward
        T[selected_arm] += 1
        g= (sum_rewards/T)+np.sqrt(L*np.log(iter+1)/(T+1))
        states[selected_arm] = np.random.binomial(1,transition_prob[selected_arm][states[selected_arm]][0])
        regret[iter] = regret[iter-1] + best_mean_reward[selected_arm] - arm_mean_reward[selected_arm]
    
    return regret

def kl_divergence(p,q): # KullBack Leiber distance
    
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def kl_bound(t,T,arm):  # The upper bound in the KL_UCB  algorithm
    return (np.log(1 + t*(np.log(t)**2)))/T[arm]


def mab_markov_kl_ucb(num_arms,num_iter,best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob):
    # num_arms: number of arms
    # num_iter: number of iterations
    # L: a constant
    
    states = np.zeros((num_arms,1),dtype=np.int32)
    T = np.zeros((num_arms,1),dtype=np.int32)   #T[i] is the number of times arm i is pulled
    sum_rewards = np.zeros((num_arms,1))   #sum_rewards[i] is the sum of rewards of arm i
    regret = np.zeros((num_iter,1))
    g = np.zeros((num_arms,1)) #g[i] is the index for arm i
    
    #We start with an exploring start playing all  arms initially in the first iteration
    for arm in range(num_arms):
        reward = np.random.binomial(1,mean_reward[arm][states[arm]])
        sum_rewards[arm] += reward
        T[arm] += 1
        g[arm] = sum_rewards[arm]/T[arm]
        states[arm] = np.random.binomial(1,transition_prob[arm][states[arm]][0])
        regret[0] += best_mean_reward - arm_mean_reward[arm]


    for t in range(1,num_iter):
        for arm in range(num_arms):
            exp_r=1 # initializing the value in the range of mean estimate and 1 that would satisfy the inequality 
            while exp_r>=(sum_rewards[arm]/T[arm]): #run loop till we satisfy the inequality
                print("t=",t,"arm=",arm,"exp_r=", exp_r)
                v1 = kl_divergence(sum_rewards[arm]/T[arm],exp_r)
                v2 = kl_bound(t,T,arm)
                if v1<=v2:
                    g[arm]=exp_r
                    break
                exp_r-=0.1
      
        selected_arm = np.argmax(g)
        reward=np.random.binomial(1,mean_reward[selected_arm][states[selected_arm]])
        T[selected_arm]+=1
        sum_rewards[selected_arm]+=reward
        states[selected_arm] = np.random.binomial(1,transition_prob[selected_arm][states[selected_arm]][0]) #transition to state 0 with probability trans_prob or else be in state 1


        regret[t]=regret[t-1]+best_mean_reward-arm_mean_reward[selected_arm] #calculating regret
    
    return regret


def plot_regret(regret):
    plt.plot(regret)
    plt.xlabel('Iterations')
    plt.ylabel('Regret')
    plt.show()
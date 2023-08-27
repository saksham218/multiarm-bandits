import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Online Algorithms for Multi-Armed Bandit Problems",
    page_icon="👋",
)

st.title("Team 1")

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
        states[arm] = np.random.binomial(1,transition_prob[arm][states[arm]][1])
        regret[0] += best_mean_reward - arm_mean_reward[arm]

    for iter in range(1,num_iter):
        selected_arm = np.argmax(g)
        reward = np.random.binomial(1,mean_reward[selected_arm][states[selected_arm]])
        sum_rewards[selected_arm] += reward
        T[selected_arm] += 1
        g= (sum_rewards/T)+np.sqrt(L*np.log(iter+1)/(T+1))
        states[selected_arm] = np.random.binomial(1,transition_prob[selected_arm][states[selected_arm]][1])
        regret[iter] = regret[iter-1] + best_mean_reward[selected_arm] - arm_mean_reward[selected_arm]
    
    return regret

def plot_regret(regret):
    st.pyplot.plot(regret)
    st.pyplot.xlabel('Iterations')
    st.pyplot.ylabel('Regret')
    st.pyplot.show()

if __name__=="__main__":
    #Using Streamitlit create a web app to show the results

    st.title("Online Algorithms for Multi-Armed Bandit Problems")

    num_arms = st.sidebar.slider("Number of arms", 2, 10, 1)
    num_iter = st.sidebar.slider("Number of iterations", 1000, 10000, 1000)

    best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob = generate_data(num_arms)

    regret = mab_markov_ucb(num_arms,num_iter,1, best_mean_reward,arm_mean_reward,mean_reward,pi,transition_prob)

    plot_regret(regret)

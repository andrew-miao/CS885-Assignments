import numpy as np
import MDP

class RL2:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward
        self.reinforce_cul_reward = []
        self.model_rl_reward = []
        self.q_learning_reward = []
        self.ucb_reward = []
        self.eps_greedy_reward = []
        self.thompson_reward = []

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs: 
        action -- sampled action
        '''

        # temporary value to ensure that the code compiles until this
        # function is coded
        action = self.reinforce_sample_action(policyParams, state)
        return action

    def epsilon_greedy(self, state, value_function, avg_reward_estimator, transition_prob_estimator, epsilon=0):
        n_actions = self.mdp.nActions
        n_states = self.mdp.nStates
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            value_action_pairs = []
            for a in range(n_actions):
                next_value = 0
                for next_state in range(n_states):
                    next_value += transition_prob_estimator[a][state][next_state] * value_function[next_state]
                q_values = avg_reward_estimator[a][state] + self.mdp.discount * next_value
                value_action_pairs.append((q_values, a))
            # to select one of the actions of equal Q-value at random due to Python's sort is stable
            np.random.shuffle(value_action_pairs)
            value_action_pairs.sort(key=lambda x:x[0], reverse=True)
            action = value_action_pairs[0][1]
        return action

    def compute_optimal_value(self, state, value_function, avg_reward_estimator, transition_prob_estimator):
        q_values = []
        for a in range(self.mdp.nActions):
            next_value = 0
            for next_state in range(self.mdp.nStates):
                next_value += transition_prob_estimator[a][state][next_state] * value_function[next_state]
            q_values.append(avg_reward_estimator[a][state] + self.mdp.discount * next_value)
        return max(q_values)

    def updateTransitionProb(self, state, action, transition_prob_estimator,
                             action_state_counter, action_state_nextstate_counter):
        for next_state in range(self.mdp.nStates):
            transition_prob_estimator[action][state][next_state] = action_state_nextstate_counter[action][state][next_state] / action_state_counter[action][state]
        return transition_prob_estimator

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)
        transition_prob_estimator = np.copy(defaultT) # transition_prob_estimator shape = (|A|, |S|, |S|)
        avg_reward_estimator = np.copy(initialR) # avg_reward_estimator shape = (|A|, |S|)
        action_state_counter = np.zeros(avg_reward_estimator.shape)
        action_state_nextstate_counter = np.zeros(transition_prob_estimator.shape)
        self.model_rl_reward = [0] * nEpisodes
        for i in range(nEpisodes):
            state = int(np.copy(s0))
            for t in range(nSteps):
                action = self.epsilon_greedy(state, V, avg_reward_estimator, transition_prob_estimator, epsilon)
                [reward, next_state] = self.sampleRewardAndNextState(state, action)
                self.model_rl_reward[i] += (self.mdp.discount ** t) * reward
                action_state_counter[action][state] += 1
                action_state_nextstate_counter[action][state][next_state] += 1
                transition_prob_estimator = self.updateTransitionProb(state, action, transition_prob_estimator,
                                                                      action_state_counter, action_state_nextstate_counter)
                avg_reward_estimator[action][state] += ((reward - avg_reward_estimator[action][state]) / action_state_counter[action][state])
                V[state] = self.compute_optimal_value(state, V, avg_reward_estimator, transition_prob_estimator)
                state = next_state
        # extract optimal policy
        for state in range(self.mdp.nStates):
            policy[state] = self.epsilon_greedy(state, V, avg_reward_estimator, transition_prob_estimator, 0)
        return [V,policy], self.model_rl_reward    

    def epsilonGreedyBandit_SampleAction(self, empiricalMeans, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.mdp.nActions)
        action_values = []
        for action in range(self.mdp.nActions):
            action_values.append(empiricalMeans[action])
        return np.argmax(action_values)

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        action_counter = np.zeros(self.mdp.nActions)
        state = 0
        self.eps_greedy_reward = [None] * nIterations
        for t in range(nIterations):
            epsilon = 1 / (t + 1)
            action = self.epsilonGreedyBandit_SampleAction(empiricalMeans, epsilon)
            [reward, state] = self.sampleRewardAndNextState(state, action)
            self.eps_greedy_reward[t] = reward
            action_counter[action] += 1
            empiricalMeans[action] += (reward - empiricalMeans[action]) / action_counter[action]
        return empiricalMeans, self.eps_greedy_reward

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        state = 0
        self.thompson_reward = [None] * nIterations
        for t in range(nIterations):
            for action in range(self.mdp.nActions):
                empiricalMeans[action] = np.mean(np.random.beta(prior[action][0], prior[action][1], size=k))
            action = np.argmax(empiricalMeans)
            [reward, state] = self.sampleRewardAndNextState(state, action)
            self.thompson_reward[t] = reward
            prior[action][0] += reward
            prior[action][1] += 1 - reward 
        return empiricalMeans, self.thompson_reward

    def UCBSampleAction(self, empiricalMeans, action_counter, timestep, epsilon=1e-08):
        action_values = []
        for action in range(self.mdp.nActions):
            value = empiricalMeans[action] + np.sqrt(2 * np.log(timestep) / (action_counter[action] + epsilon))
            action_values.append(value)
        return np.argmax(action_values)


    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        action_counter = np.zeros(self.mdp.nActions)
        epsilon = 1e-08  # avoid numerical overflow issue
        state = 0
        t = 0
        self.ucb_reward = [None] * nIterations
        for t in range(1, nIterations + 1):
            action = self.UCBSampleAction(empiricalMeans, action_counter, t, epsilon)
            [reward, state] = self.sampleRewardAndNextState(state, action)
            self.ucb_reward[t - 1] = reward
            action_counter[action] += 1
            empiricalMeans[action] += (reward - empiricalMeans[action]) / action_counter[action]
        return empiricalMeans, self.ucb_reward

    def reinforce_softmax(self, policy, state):
        probability = np.exp(policy[:, state]) / np.sum(np.exp(policy[:, state]))
        return probability

    def reinforce_sample_action(self, policy, state):
        probability = self.reinforce_softmax(policy, state)
        action = np.random.choice(self.mdp.nActions, 1, p=probability)
        return action

    def generateEpisode(self, initial_state, policy, nSteps, episode):
        state_buffer, action_buffer, reward_buffer = [], [], []
        state = int(np.copy(initial_state))
        action = self.reinforce_sample_action(policy, state)
        for t in range(nSteps):
            [reward, next_state] = self.sampleRewardAndNextState(state, action)
            self.reinforce_cul_reward[episode] += (self.mdp.discount ** t) * reward
            state_buffer.append(state)
            action_buffer.append(action)
            reward_buffer.append(reward)
            state = next_state
            action = self.reinforce_sample_action(policy, state)
        return state_buffer, action_buffer, reward_buffer

    def computeReturn(self, reward_buffer, time_step):
        Return = 0
        for i in range(time_step, len(reward_buffer)):
            Return += (self.mdp.discount ** (i - time_step)) * reward_buffer[i]
        return Return
    
    def computePolicyGradient(self, policy, action, state):
        # policy gradient of action (if action == a_i) = [-softmax(a_1), ..., 1 - softmax(a_i), ..., -softmax(a_A)]
        softmax = self.reinforce_softmax(policy, state)
        policy_gradient = np.zeros(softmax.shape)
        for a in range(self.mdp.nActions):
            if a == action:
                policy_gradient[a] = 1 - softmax[a]
            else:
                policy_gradient[a] = -softmax[a]
        return policy_gradient

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs: 
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        lr = 0.01
        policyParams = np.copy(initialPolicyParams)
        self.reinforce_cul_reward = [0] * nEpisodes
        for i in range(nEpisodes):
            state_buffer, action_buffer, reward_buffer = self.generateEpisode(s0, policy=policyParams, nSteps=nSteps, episode=i)
            for t in range(nSteps):
                Return = self.computeReturn(reward_buffer, t)
                policy_gradient = self.computePolicyGradient(policyParams, action_buffer[t], state_buffer[t])
                # policyParams[:, state_buffer[t]] +=  lr * Return * policy_gradient
                policyParams[:, state_buffer[t]] +=  lr * (self.mdp.discount ** t) * Return * policy_gradient
        return policyParams, self.reinforce_cul_reward  

    def selectOptimalAction(self, Q, state, epsilon=0, temperature=0):
        n_actions = self.mdp.nActions
        temp = np.zeros(n_actions)
        for action in range(n_actions):
            temp[action] = Q[action][state]
        if epsilon > 0:
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                # to select one of the actions of equal Q-value at random due to Python's sort is stable
                q_action_pairs = []
                for action, q in enumerate(temp):
                    q_action_pairs.append((q, action))
                np.random.shuffle(q_action_pairs)
                q_action_pairs.sort(key=lambda x:x[0], reverse=True)
                action = q_action_pairs[0][1]
            return action
        if temperature > 0:
            p = np.exp(temp / temperature) / np.sum(np.exp(temp / temperature))
            action = np.random.choice(n_actions, p)
            return action
        # to select one of the actions of equal Q-value at random due to Python's sort is stable
        q_action_pairs = []
        for action, q in enumerate(temp):
            q_action_pairs.append((q, action))
        np.random.shuffle(q_action_pairs)
        q_action_pairs.sort(key=lambda x:x[0], reverse=True)
        action = q_action_pairs[0][1]
        return action

    def computeMaxQ(self, Q, state):
        q = 0
        for action in range(self.mdp.nActions):
            q = max(q, Q[action][state])
        return q

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        temperature = float(temperature)
        counts = np.zeros([self.mdp.nActions, self.mdp.nStates])
        Q = np.copy(initialQ)
        ep_rewards = []

        for episode in range(nEpisodes):
            s = np.copy(s0)
            discounted_rewards = 0.
            for step in range(nSteps):
                if np.random.uniform() < epsilon:
                    a = np.random.randint(self.mdp.nActions)
                else:
                    if temperature == 0.:
                        a = np.argmax(Q[:, s])
                    else:
                        prob_a = np.exp(Q[:, s] / temperature) / np.sum(np.exp(Q[:, s] / temperature))
                        a = np.argmax(np.random.multinomial(1, prob_a))

                r, next_s = self.sampleRewardAndNextState(s, a)
                discounted_rewards += self.mdp.discount**step * r

                counts[a, s] += 1.
                Q[a, s] += (1. / counts[a, s]) * (r + self.mdp.discount * np.amax(Q[:, next_s]) - Q[a, s])
                s = np.copy(next_s)
            ep_rewards.append(discounted_rewards)

        policy = np.argmax(Q, axis=0)
        self.q_learning_reward = np.array(ep_rewards)
        return [Q,policy, self.q_learning_reward]
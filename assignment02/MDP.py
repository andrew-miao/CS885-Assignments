import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount

    def getMaxValue(self, value_function, state):
        """value_function: value function vector"""
        v = 0
        for action in range(self.nActions):
            v = max(v, self.R[action][state] + self.discount * np.inner(self.T[action][state], value_function))
        return v
        
    def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0
        flag = True
        while iterId < nIterations and flag:
            epsilon = 0
            for state in range(self.nStates):
                V[state] = self.getMaxValue(initialV, state)
                epsilon = max(epsilon, np.abs(V[state] - initialV[state]))
            initialV = np.copy(V)
            iterId += 1
            if epsilon <= tolerance:
                flag = False
        return [V,iterId,epsilon]

    def getOptimalAction(self, value_function, state):
        """value_function: value function vector"""
        opt_action = None
        v_max = -np.inf
        for action in range(self.nActions):
            temp = self.R[action][state] + self.discount * np.inner(self.T[action][state], value_function)
            if v_max < temp:
                opt_action = action
                v_max = temp
        return opt_action

    def extractPolicy(self, V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates)
        for state in range(self.nStates):
            policy[state] = self.getOptimalAction(V, state)
        return policy 

    def evaluatePolicy(self, policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded

        # build R^pi and T^pi
        reward_pi = np.zeros(self.nStates)
        transition_pi = np.zeros((self.nStates, self.nStates))
        for state in range(self.nStates):
            reward_pi[state] = self.R[policy[state]][state]
            transition_pi[state] = self.T[policy[state]][state]
        inverse = np.linalg.inv(np.identity(self.nStates) - self.discount * transition_pi)
        V = np.dot(inverse, reward_pi)
        return V
        
    def policyImprovement(self, value_function, policy):
        policy_stable = True
        for state in range(self.nStates):
            temp = policy[state]
            policy[state] = self.getOptimalAction(value_function, state)
            if temp != policy[state]:
                policy_stable = False
        return policy_stable, policy

    def policyIteration(self, initialPolicy, nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.copy(initialPolicy)
        V = np.zeros(self.nStates)
        iterId = 0
        flag = True
        while flag and iterId < nIterations:
            iterId += 1
            V = self.evaluatePolicy(policy)
            policy_stable, policy = self.policyImprovement(V, policy)
            flag = not policy_stable
        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0
        flag = True
        while iterId < nIterations and flag:
            epsilon = 0
            for state in range(self.nStates):
                V[state] = self.R[policy[state]][state] + self.discount * np.inner(self.T[policy[state]][state], initialV)
                epsilon = max(epsilon, np.abs(V[state] - initialV[state]))
            initialV = np.copy(V)
            iterId += 1
            if epsilon <= tolerance:
                flag = False
        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.copy(initialPolicy)
        V = np.zeros(initialV.shape)
        iterId = 0
        epsilon = 0
        flag = True
        while flag and iterId < nIterations:
            iterId += 1
            [initialV, _, epsilon] = self.evaluatePolicyPartially(policy, initialV, nEvalIterations, tolerance)
            __, policy = self.policyImprovement(V, policy)
            # Personally, I don't like the following lines, because it destroy the beauty of the 
            # cycle of policy evalution -> <- policy improvement, even these lines guarantee the convergence.
            for state in range(self.nStates):
                V[state] = self.getMaxValue(initialV, state)
            epsilon = np.linalg.norm(V - initialV, np.inf)
            flag = (epsilon > tolerance)
        return [policy,V,iterId,epsilon]
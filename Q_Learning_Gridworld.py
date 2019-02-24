#%%
"""
This is a solution to Gridworld using off policy Q Learning.
As enviroment it emplys LazyProgrammers Grid_world.py https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
Solution is part of The School of AI's Move 37 Course https://www.theschool.ai/courses/move-37-course/
Written as a jupyter cell in visual studio code. Just run the cell. If you want to run using python interpreter directly, replace def main(): to if name == 'main': and remove the last line (call of main()).
"""
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

CONST_ACTION_LST = ('U','D','L','R')
CONST_N_EPISODES = 10000
CONST_START_STATE = (2,0)

class QLearningAgent:
    def __init__(self,states,actions):
        self.actions = actions
        self.states = states
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.Q = {j:{i:0 for i in self.actions} for j in self.states}
        print(self.Q)

    def learn(self,state,action,reward,next_state):
        """ 
        This is the Q Learning off policy algorithm as descibed in Chapt.6.5 Introduction to reinforment learning
        Args: S,A,R,S'
        Return: Nothing, writes to self.Q
        """
        Q_old_value = self.Q[state][action] #Required for new Q value
        TD_Target_value = reward + self.discount_factor*max(self.Q[next_state].values()) #CAVE: Be sure max returns the value not the key of dict
        self.Q[state][action] = self.Q[state][action] + self.learning_rate * (TD_Target_value - Q_old_value) #set new Q value
    
    def get_action(self,state):
        """ 
        Returns either the best action according to Q fuction (argmax) 
        or random aciton with self.epsion setting the probability
        Args: state: the stae
        Return: action (epsion greedy)
        """
        rand_number = np.random.random()
        if rand_number < (1-self.epsilon):
            return np.random.choice(self.actions)
        else:
            return max(self.Q[state],key=self.Q[state].get) #max(dict,key=dict.get) returns the key belong to the largest value in the dict

def main(): #If you want to run using python interpreter directly, replace def main(): to if __name__ == '__main__': 
    #Create enviroment
    env = standard_grid(obey_prob=0.9,step_cost=None)
    #Create agent
    agent = QLearningAgent(env.all_states(), CONST_ACTION_LST)
    #Learn Policy by playing many episodes and Q-Learning adapting the Policy
    for episode in range(10000):
        env.set_state(CONST_START_STATE)
        state = env.current_state()
        while True:
            action = agent.get_action(state)
            reward = env.move(action)
            next_state = env.current_state()
            agent.learn(state,action,reward,next_state)
            if env.game_over():
                break
            state = next_state
    print(agent.Q)

main() #Remove if running with pure python interpreter

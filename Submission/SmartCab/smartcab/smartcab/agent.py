import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from McHersheyHashMcHarshFace import HashFunction
import pandas as pd
from pandas.core import panelnd
from pandas.core import panel4d
import numpy as np
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qt=QTable(0.5,0.5)
        print '-'*80
    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        totalTime=self.env.get_deadline(self)
        self.qt.printVal(totalTime)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        current_state = self.env.sense(self)
        
        deadline = self.env.get_deadline(self)
               # TODO: Update state
        
        
        # TODO: Select action according to your policy
        #action = random.choice([None, 'forward', 'left', 'right'])
        action = self.qt.get_next_action( self.next_waypoint, deadline, current_state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        next_state=self.env.sense(self)
        self.qt.update(self.next_waypoint, deadline, current_state, action, reward)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


class QTable(object):

    def __init__(self, alpha, gamma):
        rnd0=0*np.random.random_sample((4,100,512,2))
        p4d = pd.Panel4D(rnd0)

        Panel5D = panelnd.create_nd_panel_factory(klass_name   = 'Panel5D',
        orders  = [ 'action', 'labels','items','major_axis','minor_axis'],
        slices  = { 'labels' : 'labels', 'items' : 'items',
        'major_axis' : 'major_axis', 'minor_axis' : 'minor_axis'},
        slicer  = pd.Panel4D,
        stat_axis    = 2)

        p5d = Panel5D(dict(AN = p4d,BF=p4d,CR=p4d,DL=p4d))
        self.qtable=p5d
        self.alpha=alpha
        self.gamma=gamma
        self.trial_count=0
        self.exploration_trials=40
        self.reward=np.zeros(4)
        self.total_time=0
        self.trial_steps_df=pd.DataFrame(np.random.randint(1, size=(100,2)))
        pd.set_option("display.max_rows",100)

    
    
    def printVal(self, totalTime):
            self.trial_count += 1
            self.total_time=totalTime
    

    def update(self, next_waypoint, deadline, current_state, action, reward):
       
        self.trial_steps_df.set_value(self.trial_count,1,deadline)
        hashF = HashFunction()
        action_code,nav_code, deadline, traffic_code,light_code = hashF.hash5DState(action, next_waypoint, deadline, current_state)
        

        prev_val = self.qtable.ix[action_code,nav_code, deadline, traffic_code,light_code]
        new_val=(1-self.alpha)*prev_val + self.alpha*reward
        
        self.qtable.ix[action_code,nav_code, deadline, traffic_code,light_code]=new_val

        if(self.trial_count==99 and deadline==20):
            print " STEPS: : ",  self.trial_steps_df
            self.trial_steps_df.plot()
            plt.show()
    
    
    def get_next_action(self, next_waypoint, deadline, current_state):
        
        #print "trial: ", self.trial_count,"  deadline: ", deadline, " of ", self.total_time
        if self.trial_count < self.exploration_trials: # explotation
            return random.choice([None, 'forward', 'right', 'left'])
        else: # exploitation
            hashF = HashFunction()
            nav_code, deadline, traffic_code,light_code=hashF.hash4DState(next_waypoint, deadline, current_state)
           

            self.reward[0]=self.qtable.ix['AN',nav_code, deadline, traffic_code,light_code]
            self.reward[1]=self.qtable.ix['BF',nav_code, deadline, traffic_code,light_code]
            self.reward[2]=self.qtable.ix['CR',nav_code, deadline, traffic_code,light_code]
            self.reward[3]=self.qtable.ix['DL',nav_code, deadline, traffic_code,light_code]
            max_rew=np.argmax(self.reward)
       
            suggested_action=[None, 'forward', 'right', 'left'][max_rew]
        
        
        
        return suggested_action


if __name__ == '__main__':
    run()



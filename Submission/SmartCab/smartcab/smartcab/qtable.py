import random
from McHersheyHashMcHarshFace import HashFunction
import pandas as pd
from pandas.core import panelnd
from pandas.core import panel4d
import numpy as np
import matplotlib.pyplot as plt

class QTable(object):
    
    def __init__(self, alpha, gamma, explore_to_exploit_ratio):
        rnd0=0*np.random.random_sample((4,100,512,2))
        p4d = pd.Panel4D(rnd0)
        
        Panel5D = panelnd.create_nd_panel_factory(
        klass_name   = 'Panel5D',
        orders  = [ 'action', 'labels','items','major_axis','minor_axis'],
        slices  = {'labels' : 'labels', 'items' : 'items','major_axis' : 'major_axis', 'minor_axis' : 'minor_axis'},
        slicer  = pd.Panel4D,
        stat_axis    = 2)

        p5d = Panel5D(dict(AN = p4d,BF=p4d,CR=p4d,DL=p4d))
        self.qtable=p5d
        self.alpha=alpha
        self.gamma=gamma
        self.trial_count=0
        self.exploration_ratio=explore_to_exploit_ratio
        self.exploration_trials=int( self.exploration_ratio * 100)
        self.exploitation_trials = 100 - self.exploration_trials
        print "explore:", self.exploration_trials,", exploit:",self.exploitation_trials
        self.reward=np.zeros(4)
        self.total_time=0
        self.trial_steps_df=pd.DataFrame(np.random.randint(1, size=(101,3)))
        pd.set_option("display.max_rows",200)
        self.exploration_success=0.0
        self.exploitation_success=0.0
    



    def printVal(self, totalTime):
        self.trial_count += 1
        self.total_time=totalTime
        #print "----- PRINT --trial_count:",self.trial_count, ",total_time:",self.total_time,",totalTime",totalTime
    
    
    def update(self, next_waypoint, deadline, current_state, action, reward, agent, env):
        hashF = HashFunction()
        action_code,nav_code, deadline, traffic_code,light_code = hashF.hash5DState(action, next_waypoint, deadline, current_state)
        prev_val = self.qtable.ix[action_code,nav_code, deadline, traffic_code,light_code]
        
        #Set new Q-Value for State, Action, based on  Reward
        new_val=(1-self.alpha)*prev_val + self.alpha*reward
        
        self.qtable.ix[action_code,nav_code, deadline, traffic_code,light_code]=new_val
        
        agent_state = env.agent_states[agent]
        destination = agent_state['destination']
        location = agent_state['location']
        heading = agent_state['heading']
        
        #print "self.trial_count: ",self.trial_count
        prev_reward=self.trial_steps_df.ix[self.trial_count,2]
        new_reward = prev_reward + reward
        self.trial_steps_df.set_value(self.trial_count,2,new_reward)
        
        #print "self.trial_count: ",self.trial_count, " , reward: ", reward,", Prev reward:", prev_reward, ", new_reward:",new_reward
        
        #Calculate success rate during Exploration and Exploitation phase
        if destination == location:
            #print "self.trial_count: ",self.trial_count, ", reached:\n",
            self.trial_steps_df.set_value(self.trial_count,1,(self.total_time - deadline))
            if self.trial_count < self.exploration_trials:
                self.exploration_success += 1
            else:
                self.exploitation_success += 1
            
            
        if self.trial_count == 99 and (destination == location or deadline== 0):
            #print " STEPS: : ",  self.trial_steps_df
            print "Exploration to Exploitation Ratio:", self.exploration_ratio
            print ",  Exploration Success Rate:", (self.exploration_success / self.exploration_trials)
            print ",  Exploitation Success Rate:", (self.exploitation_success / self.exploitation_trials)
            print "Steps needed to reach target (Zero Means not reached), and Rewards Collected in each trial:\n", self.trial_steps_df
            
    
    
    def get_next_action(self, next_waypoint, deadline, current_state):

        if self.trial_count < self.exploration_trials: # explotation
            return random.choice([None, 'forward', 'right', 'left'])
        else: # exploitation
            hashF = HashFunction()
            nav_code, deadline, traffic_code,light_code = hashF.hash4DState(next_waypoint, deadline, current_state)
            
            #Get Q values for all 4 possible next states.
            self.reward[0]=self.qtable.ix['AN',nav_code, deadline, traffic_code,light_code]
            self.reward[1]=self.qtable.ix['BF',nav_code, deadline, traffic_code,light_code]
            self.reward[2]=self.qtable.ix['CR',nav_code, deadline, traffic_code,light_code]
            self.reward[3]=self.qtable.ix['DL',nav_code, deadline, traffic_code,light_code]
            max_rew=np.argmax(self.reward)
            
            # Take the action that leads to best next state
            # Add some randomization when still time to
            #if(deadline > 10 and deadline % 5 == 0):
            #    suggested_action=random.choice([None, 'forward', 'right', 'left'])
            #else:
            suggested_action=[None, 'forward', 'right', 'left'][max_rew]
        
        
        
        return suggested_action
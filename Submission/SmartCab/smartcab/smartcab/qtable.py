import random
from McHersheyHashMcHarshFace import HashFunction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

class QTable(object):
    
    def __init__(self, alpha, gamma):
        rnd0= 0.5+np.zeros((4,64,2))
        p3d = pd.Panel(rnd0)
        #print "p3d: ",p3d.values
        
        p4d = pd.Panel4D(dict(AN = p3d,BF=p3d,CR=p3d,DL=p3d))
        self.qtable=p4d
        #print "self.qtable: ",self.qtable.values
        self.alpha=alpha
        self.gamma=gamma
        self.trial_count=0
        self.qvalue=np.zeros(4)
        self.next_qvalue=np.zeros(4)
        self.total_time=0
        self.trial_steps_df=pd.DataFrame(np.random.randint(1, size=(101,3)))
        pd.set_option("display.max_rows",200)
        self.success_rate=0.0
    



    def printVal(self, totalTime):
        self.trial_count += 1
        self.total_time=totalTime
        #print "----- PRINT --trial_count:",self.trial_count, ",total_time:",self.total_time,",totalTime",totalTime
    
    
    def update(self, next_waypoint, deadline, current_state, action, reward, next_state_value, next_state_waypoint, agent, env):
        hashF = HashFunction()
        action_code,nav_code, traffic_code,light_code = hashF.hash5DState(action, next_waypoint, current_state)
        prev_val = self.qtable.ix[action_code,nav_code, traffic_code,light_code]
        
        #GEt Qvalues of possible next states
        next_nav_code, next_traffic_code, next_light_code = hashF.hash4DState(next_state_waypoint,  next_state_value)
            
        #Get Q values for all 4 possible next states.
        self.next_qvalue[0]=self.qtable.ix['AN',next_nav_code,  next_traffic_code,next_light_code]
        self.next_qvalue[1]=self.qtable.ix['BF',next_nav_code,  next_traffic_code,next_light_code]
        self.next_qvalue[2]=self.qtable.ix['CR',next_nav_code,  next_traffic_code,next_light_code]
        self.next_qvalue[3]=self.qtable.ix['DL',next_nav_code , next_traffic_code,next_light_code]
        next_max_qvalue=np.mean(self.next_qvalue)

        
        #Set new Q-Value for State, Action, based on  Reward
        new_val=(1-(1.0/self.trial_count))*prev_val + (1.0/self.trial_count)* (reward + self.gamma * next_max_qvalue)
        
        self.qtable.ix[action_code,nav_code,  traffic_code,light_code]=new_val
        
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
        #if destination == location:
            #print "self.trial_count: ",self.trial_count, ", reached:\n",
            #self.trial_steps_df.set_value(self.trial_count,1,(self.total_time - deadline))
            #self.success_rate += 1
        
            
            #if self.trial_count == 99 and (destination == location or deadline== 0):
            #print " STEPS: : ",  self.trial_steps_df
            #print "Success Rate:", self.success_rate
            #print "Steps needed to reach target (Zero Means not reached), and Rewards Collected in each trial:\n", self.trial_steps_df
            #print "Q-Table", self.qtable.values





    
    
    def get_next_action(self, next_waypoint, deadline, current_state):


        hashF = HashFunction()
        nav_code, traffic_code,light_code = hashF.hash4DState(next_waypoint, current_state)
        
        #Get Q values for all 4 possible next states.
        self.qvalue[0]=self.qtable.ix['AN',nav_code,  traffic_code,light_code]
        self.qvalue[1]=self.qtable.ix['BF',nav_code,  traffic_code,light_code]
        self.qvalue[2]=self.qtable.ix['CR',nav_code, traffic_code,light_code]
        self.qvalue[3]=self.qtable.ix['DL',nav_code, traffic_code,light_code]
        max_qvalue=np.argmax(self.qvalue)
        
        # Take the action that leads to best next state
        # Add some randomization when still time to
        
        suggested_action=[None, 'forward', 'right', 'left'][max_qvalue]


        return suggested_action
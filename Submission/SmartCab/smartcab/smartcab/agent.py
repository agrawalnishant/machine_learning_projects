import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
from pandas.core import panelnd
from pandas.core import panel4d
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qt=QTable(0.5,0.5)
        print "qt: ",self.qt
    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        current_state = self.env.sense(self)
        #print "current_state:", current_state
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        
        
        # TODO: Select action according to your policy
        action = random.choice([None, 'forward', 'left', 'right'])

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        next_state=self.env.sense(self)
        print current_state, action, reward, next_state
        self.qt.update(current_state, action, reward, next_state)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, current_state, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=1)  # press Esc or close pygame window to quit


class QTable(object):

    def __init__(self, alpha, gamma):
        rnd=0*np.random.random_sample((2,4,4,4))
        p4d = pd.Panel4D(rnd)

        Panel5D = panelnd.create_nd_panel_factory(klass_name   = 'Panel5D',
        orders  = [ 'action', 'labels','items','major_axis','minor_axis'],
        slices  = { 'labels' : 'labels', 'items' : 'items',
        'major_axis' : 'major_axis', 'minor_axis' : 'minor_axis'},
        slicer  = pd.Panel4D,
        stat_axis    = 2)

        p5d = Panel5D(dict(N = p4d,F=p4d,L=p4d,R=p4d))
        self.qtable=p5d
        self.alpha=alpha
        self.gamma=gamma
        print "p5d:", p5d

    def update(self, current_state, action, reward, next_state):
        current_state_df=pd.DataFrame(current_state.items())
        if current_state_df.iloc[0,1] == 'green':
            light_code=0
        else:
            light_code=1

        if current_state_df.iloc[1,1] == None:
            oncoming_code=0
        elif current_state_df.iloc[1,1] == 'forward':
            oncoming_code=1
        elif current_state_df.iloc[1,1] == 'left':
            oncoming_code=2
        elif current_state_df.iloc[1,1] == 'right':
            oncoming_code=3

        if current_state_df.iloc[2,1] == None:
            left_code=0
        elif current_state_df.iloc[2,1] == 'forward':
            left_code=1
        elif current_state_df.iloc[2,1] == 'left':
            left_code=2
        elif current_state_df.iloc[2,1] == 'right':
            left_code=3

        if current_state_df.iloc[3,1] == None:
            right_code=0
        elif current_state_df.iloc[3,1] == 'forward':
            right_code=1
        elif current_state_df.iloc[3,1] == 'left':
            right_code=2
        elif current_state_df.iloc[3,1] == 'right':
            right_code=3

        if action == None:
            action_code='N'
        elif action == 'forward':
            action_code='F'
        elif action == 'left':
            action_code='L'
        elif action == 'right':
            action_code='R'
        
        print "action_code:",action_code,",light_code:",light_code,",oncoming_code:",oncoming_code,",left_code:",left_code,",right_code:",right_code,

        prev_val = self.qtable.ix[action_code,light_code,oncoming_code,left_code,right_code]
        new_val=(1-self.alpha)*prev_val + self.alpha*reward
        
        print "prev_val: ",prev_val,", next_val:", new_val
        
        self.qtable.ix[action_code,light_code,oncoming_code,left_code,right_code]=new_val

        print "q-Table:", self.qtable.values

if __name__ == '__main__':
    run()



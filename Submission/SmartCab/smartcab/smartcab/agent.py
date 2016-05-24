import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from McHersheyHashMcHarshFace import HashFunction
from qtable import QTable
import pandas as pd
from pandas.core import panelnd
from pandas.core import panel4d
import numpy as np
import matplotlib.pyplot as plt


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env,**kwargs):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        alpha=0.5
        gamma=0.002
        add_total=0
        add_total=False
        self.success=0
        self.total=0
        self.simulated_annealing_counter=5
        self.qt=QTable(alpha,gamma)
        print '-'*80
    
    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        totalTime=self.env.get_deadline(self)
        self.qt.printVal(totalTime)
        self.simulated_annealing_counter=5

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        current_state = self.env.sense(self)
        self.state=current_state
        
        deadline = self.env.get_deadline(self)
        # TODO: Update state
        
        
        # TODO: Select action according to your policy
        #action = random.choice([None, 'forward', 'left', 'right'])
        action = self.qt.get_next_action( self.next_waypoint, deadline, current_state, self.simulated_annealing_counter)
        if action == None:
            self.simulated_annealing_counter -= 1
        else:
            self.simulated_annealing_counter=5

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        add_total = False
        if deadline == 0:
            add_total = True
        if reward > 10:
            self.success += 1
            add_total = True
        if add_total:
            self.total += 1
            print("success: {} / {}".format(self.success, self.total))

        # TODO: Learn policy based on state, action, reward
        next_state_value=self.env.sense(self)
        next_state_deadline=self.env.get_deadline(self)
        next_state_waypoint=self.planner.next_waypoint()
        self.qt.update(self.next_waypoint, deadline, current_state, action, reward, next_state_value, next_state_waypoint,  self, self.env)


def run():
    """Run the agent for a finite number of trials."""

    ## Run with diff exploration to exploitation ratio
    ratios = [0.05]
    # Set up environment and agent
    for ratio in ratios:
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent, ratio_value=ratio)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

        # Now simulate it
        sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
        sim.run(n_trials=100)  # press Esc or close pygame window to quit





if __name__ == '__main__':
    run()



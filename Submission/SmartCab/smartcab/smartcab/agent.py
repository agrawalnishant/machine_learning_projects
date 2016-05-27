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
import math


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env,**kwargs):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        add_total=0
        add_total=False
        self.success=0
        self.total=0
        self.counter=0
        self.epsilon_reset_counter=0
        self.trial_counter=0.0
        self.min_epsilon=0.001
        self.eps_freq=1.0
        self.filled_cell_count=0
        self.total_cell_count=0
        self.updated_func_counter=0
        
        for key, value in kwargs.iteritems():
            print "%s = %s" %(key,value)
            if key == 'alp':
                self.alpha=value
            elif key == 'gma':
                self.gamma=value
            elif key == 'eps':
                self.epsl=value
        self.epsilon=self.epsl
        print "epsilon: ",self.epsilon
        self.qt=QTable(self.alpha,self.gamma)
        print '-'*80
    
    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        totalTime=self.env.get_deadline(self)
        self.qt.printVal(totalTime)
        self.trial_counter += 1.0
        if self.epsilon > self.min_epsilon:
            self.epsilon = (5.0 * self.epsl) / self.trial_counter
            self.eps_freq = math.ceil(1.0 / self.epsilon)
            print "self.epsilon:",self.epsilon, ", self.eps_freq: ", self.eps_freq, "\n"

    
    def update(self, t):
        self.counter+=1
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        current_state = self.env.sense(self)
        self.state=current_state
        
        deadline = self.env.get_deadline(self)
        # TODO: Update state
        
        
        # TODO: Select action according to your policy
        
        #action = random.choice([None, 'forward', 'left', 'right'])
        #if self.total > 0 and self.total % self.epsilon_freq == 0.0:
        #    print "simulated annealing at ", self.total
        #    action = random.choice([None, 'forward', 'left', 'right'])
        #else:
        if self.epsilon > self.min_epsilon and deadline!=0 and deadline != self.eps_freq and math.floor(deadline % self.eps_freq ) == 0.0:
            #self.epsilon_reset_counter += 1
            action = random.choice([None, 'forward', 'left', 'right'])
            print "annealing now.", "self.epsilon:",self.epsilon, ", action: ",action, ", deadline:", deadline

        else:
            #print "self.counter: ", self.counter, ", multiplier:", (self.counter * self.epsilon)
            action = self.qt.get_next_action( self.next_waypoint, deadline, current_state)
        

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
                
        if self.total == 100:
            
            for item, frame in self.qt.qtable.iteritems():
                for item2, frame2 in frame.iteritems():
                    for item3, frame3 in frame2.iteritems():
                        for item4, frame4 in frame3.iteritems():
                            self.total_cell_count +=1
                            #print("f4:", frame4)
                            if frame4 != 0.0:
                                #print "\n"
                                self.printNav(item2)
                                self.printTraffic(item3)
                                self.printTrafficLight(item4)
                                self.printAction(item)
                                print "Q-Val: {0:.5f}".format(frame4)
                                self.filled_cell_count+=1
            print '-'*80
            print "updated cells: ", self.filled_cell_count, ", self.total_cell_count:",self.total_cell_count, ", updated_func_counter:",self.updated_func_counter
            print "self.alpha:", self.alpha, "self.gamma:", self.gamma,", self.epsilon:",self.epsl,", success:", self.success
            print '_'*80
                        #    print '_'*20
        # TODO: Learn policy based on state, action, reward
        next_state_value=self.env.sense(self)
        next_state_deadline=self.env.get_deadline(self)
        next_state_waypoint=self.planner.next_waypoint()
        self.qt.update(self.next_waypoint, deadline, current_state, action, reward, next_state_value, next_state_waypoint,  self, self.env)
        self.updated_func_counter += 1

    def printAction(self, code):
        print '|',
        if code=='AN':
            print "Action: None",
        elif code=='BF':
            print "Action: Forward",
        elif code=='CR':
            print "Action: Right",
        elif code=='DL':
            print "Action: Left",
        print '|',

    def printNav(self, code):
        print '|',
        if code==0:
            print "Nav: None",
        elif code==1:
            print "Nav: Forward",
        elif code==2:
            print "Nav: Right",
        elif code==3:
            print "Nav: Left",

    def printTraffic(self, code):
        left_mask=0b000011
        right_mask=0b001100
        oncoming_mask=0b110000

        left_filtered = code & left_mask
        right_filtered = code & right_mask
        oncoming_filtered = code & oncoming_mask

        print '| Traffic state: ',
        if left_filtered == 0:
            print "Left: None",
        elif left_filtered == 1:
            print "Left: Forward",
        elif left_filtered == 2:
            print "Left: Right",
        elif left_filtered == 3:
            print "Left: Left",
        print '-+-',

        if right_filtered == 0:
            print "Right: None",
        elif right_filtered == 4:
            print "Right: Forward",
        elif right_filtered == 8:
            print "Right: Right",
        elif right_filtered == 12:
            print "Right: Left",
        print '-+-',

        if oncoming_filtered == 0:
            print "Oncoming: None",
        elif oncoming_filtered == 16:
            print "Oncoming: Forward",
        elif oncoming_filtered == 32:
            print "Oncoming: Right",
        elif oncoming_filtered == 48:
            print "Oncoming: Left",

    def printTrafficLight(self, code):
        print '| ',
        if code == 0:
            print "Light: Red",
        else:
            print "Light: Green",



def run():
    """Run the agent for a finite number of trials."""

    ## Run with diff exploration to exploitation ratio
    alphas = [0.7, 0.5,0.3]
    gammas=[0.005,.05, 0.1]
    epsilon_probabilities=[0.5,0.2,0.1]
    # Set up environment and agent
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilon_probabilities:
                e = Environment()  # create environment (also adds some dummy traffic)
                a = e.create_agent(LearningAgent, alp=alpha, gma=gamma,eps=epsilon)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

                # Now simulate it
                sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
                sim.run(n_trials=100)  # press Esc or close pygame window to quit





if __name__ == '__main__':
    run()



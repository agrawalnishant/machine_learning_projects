import pandas as pd


class HashFunction(object):
    """ Hashing Function to find index based on state"""

    def hash4DState(self, next_waypoint,  current_state):
        """Find Index into Q-Table, for 4 Dimentions of state: Navigation, Traffic, and Light. """
        
        current_state_df=pd.DataFrame(current_state.items())
        
        if current_state_df.iloc[0,1] == 'red':
            light_code=0
        else:
            light_code=1
        
        if current_state_df.iloc[1,1] == None:
            oncoming_code=0
        elif current_state_df.iloc[1,1] == 'forward':
            oncoming_code=32
        elif current_state_df.iloc[1,1] == 'right':
            oncoming_code=64
        elif current_state_df.iloc[1,1] == 'left':
            oncoming_code=96
        
        if current_state_df.iloc[2,1] == None:
            right_code=0
        elif current_state_df.iloc[2,1] == 'forward':
            right_code=8
        elif current_state_df.iloc[2,1] == 'right':
            right_code=16
        elif current_state_df.iloc[2,1] == 'left':
            right_code=24
        
        if current_state_df.iloc[3,1] == None:
            left_code=0
        elif current_state_df.iloc[3,1] == 'forward':
            left_code=1
        elif current_state_df.iloc[3,1] == 'right':
            left_code=2
        elif current_state_df.iloc[3,1] == 'left':
            left_code=3
        
        traffic_code = oncoming_code + right_code + left_code
        
        if next_waypoint == None:
            nav_code=0
        elif next_waypoint == 'forward':
            nav_code=1
        elif next_waypoint == 'right':
            nav_code=2
        elif next_waypoint == 'left':
            nav_code=3

        return nav_code,  traffic_code,light_code


    def hash5DState(self, action, next_waypoint,  current_state):
        """Find Index into Q-Table, for 4 Dimentions of state: Navigation, Traffic, and Light. """

        if action == None:
            action_code='AN'
        elif action == 'forward':
            action_code='BF'
        elif action == 'right':
            action_code='CR'
        elif action == 'left':
            action_code='DL'

        nav_code,  traffic_code,light_code = self.hash4DState(next_waypoint,  current_state)


        return action_code, nav_code, traffic_code, light_code
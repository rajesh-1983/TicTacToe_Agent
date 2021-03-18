from gym import spaces
import numpy as np
from numpy import nan
import random
from itertools import groupby
from itertools import product

state_shape = 3
win_total = 15

class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
       
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        won = False
        #Transform array to 2D matrix
        curr_state_matrix = np.array(curr_state).reshape(state_shape, state_shape)
        
        for index in range(state_shape):
            #capture row vector
            row_vector = curr_state_matrix[index, :]

            #capture column vector
            col_vector = curr_state_matrix[:, index]

            if((np.nan not in row_vector) and np.sum(row_vector) == win_total) or\
                    ((np.nan not in col_vector) and (np.sum(col_vector) == win_total)):
                won = True
                break
        
        if not won:      
            #capture diagnonal elements
            right_diagonal = np.diag(curr_state_matrix) 
            left_diagonal = np.diag(np.fliplr(curr_state_matrix))
          
            if (np.nan not in right_diagonal and np.sum(right_diagonal) == win_total) or\
             (np.nan not in left_diagonal and np.sum(left_diagonal) == win_total):
                won = True
                   
        return won            
            

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up
        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) == 0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)


    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state  

    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
             
        reward = 0
        # Get next state based current state and action
        next_state = self.state_transition(curr_state, curr_action)

        # Check whether we terminal state within environment
        check_terminal_state = self.is_terminal(next_state)
        # print(check_terminal_state)
        
        if check_terminal_state[0]:
            if check_terminal_state[1] == 'Win':
                reward = 10
            else:
                reward = 0
        else:
            # Select random action from environment
            env_action_space = list(self.action_space(next_state)[1])
            random_action_index = np.random.choice(len(env_action_space))
            random_action = env_action_space[random_action_index]

            # Apply random action on next_state 
            next_state = self.state_transition(next_state, random_action)
            # Check whether we terminal state within environment
            check_terminal_state = self.is_terminal(next_state)

            if check_terminal_state[0]:
                if check_terminal_state[1] == 'Win':
                    reward = -10
                else:
                     reward = 0
            else:
                 reward = -1
         
        # Return tuple represent the next-state, reward and is_terminal state    
        return (next_state, reward, check_terminal_state[0])
                 
             

    def reset(self):
        return self.state

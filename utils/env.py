import time 
import numpy as np 
from IPython.display import clear_output

import matplotlib.pyplot as plt 
import seaborn as sns 

from .viz import viz

################################################
#                                              #
#         THE FROZEN LAKE ENVIRONMENT          # 
#                                         @ZF  #
################################################

layout = [
    "S.......",
    "........",
    "...H....",
    ".....H..",
    "...H....",
    ".HH...H.",
    ".H..H.H.",
    "...H...G"
]


class frozen_lake:
    n_row = 8
    n_col = 8

    def __init__(self, layout=layout, eps=.2, seed=1234):

        # get occupancy 
        self.rng = np.random.RandomState(seed)
        self.layout = layout
        self.get_occupancy()
        self.eps = eps 
        # define MDP 
        self._init_S()
        self._init_A()
        self._init_P()
        self._init_R()
        

    def get_occupancy(self):
        # get occupancy, current state and goal 
        map_dict = {
            'H': .7,
            '.': 0,
            'S': 0, 
            'G': 0,
        }
        self.occupancy = np.array([list(map(lambda x: map_dict[x], row)) 
                                   for row in self.layout])
        self.goal = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='G'))
        self.curr_cell = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='S'))
        holes = np.array([list(row) for row in self.layout])=='H'
        self.hole_cells = [h for h in np.vstack(np.where(holes)).T]
        
    def cell2state(self, cell):
        return cell[0]*self.occupancy.shape[1] + cell[1]
    
    def state2cell(self, state):
        n = self.occupancy.shape[1]
        return np.array([state//n, state%n])
        
    # ------------------------------------ #
    #            Define the MDP            #
    # ------------------------------------ #

    def _init_S(self):
        '''Define the state space
        '''
        # all possible state
        self.nS = frozen_lake.n_row*frozen_lake.n_col
        self.S  = list(range(self.nS))
        self.state = self.cell2state(self.curr_cell)
        # define the terminal states: goal, holes
        self.goal_state = self.cell2state(self.goal)
        self.hole_states = [self.cell2state(h) for h in self.hole_cells]
        self.s_termination = self.hole_states+[self.goal_state]

    def _init_A(self,):
        '''Define the action space 
        '''
        self.directs = [
            np.array([-1, 0]), # up
            np.array([ 1, 0]), # down
            np.array([ 0,-1]), # left
            np.array([ 0, 1]), # right
        ]
        self.nA = len(self.directs)
        self.A  = list((range(self.nA)))

    def _init_P(self):
        '''Define the transition function, P(s'|s,a)

            P(s'|s,a) is a probability distribution that
            maps s and a to a distribution of s'. 
        '''

        def p_s_next(s, a):
            p_next = np.zeros([self.nS])
            cell = self.state2cell(s)
            # if the current state is terminal state
            # state in the current state 
            if s in self.s_termination:
                p_next[s] = 1 
            else:
                for j in self.A:
                    s_next = self.cell2state(
                        np.clip(cell + self.directs[j],
                        0, frozen_lake.n_row-1))
                    # the agent is walking on a surface of frozen ice, they cannot always
                    # successfully perform the intended action. For example, attempting to move "left"
                    # may result in the agent moving to the left with a probability of 1-ε.
                    # With probability ε, the agent will randomly move in one of the 
                    # other possible directions.
                    if j == a: 
                        p_next[s_next] += 1-self.eps
                    else:
                        p_next[s_next] += self.eps / (self.nA-1)
                
            return p_next
        
        self.p_s_next = p_s_next

    def _init_R(self):
        '''Define the reward function, R(s')

        return:
            r: reward
            done: if terminated 
        '''
        def R(s_next):
            if s_next == self.goal_state:
                return 1, True
            elif s_next in self.hole_states:
                return -1, True
            else:
                return 0, False
        self.r = R
        
    # ------------ visualize the environment ----------- #

    def reset(self):
        '''Reset the environment

            Bring the agent back to the starting point
        '''
        self.curr_cell = np.hstack(np.where(np.array([list(row) 
                        for row in self.layout])=='S'))
        self.state = self.cell2state(self.curr_cell)
        self.done = False
        self.act = None

        return self.state, None, self.done 

    def render(self, ax):
        '''Visualize the current environment
        '''
        occupancy = np.array(self.occupancy)
        sns.heatmap(occupancy, cmap=viz.mixMap, ax=ax,
                    vmin=0, vmax=1, 
                    lw=.5, linecolor=[.9]*3, cbar=False)
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=occupancy.shape[0], color='k',lw=5)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=occupancy.shape[1], color='k',lw=5)
        ax.text(self.goal[1]+.15, self.goal[0]+.75, 'G', color=viz.Red,
                    fontweight='bold', fontsize=10)
        ax.text(self.curr_cell[1]+.25, self.curr_cell[0]+.75, 'O', color=viz.Red,
                    fontweight='bold', fontsize=10)
        r, _ = self.r(self.state)
        ax.set_title(f'Reward: {r}, done: {self.done}')
        ax.set_axis_off()
        ax.set_box_aspect(1)

    def show_pi(self, ax, pi):
        '''Visualize your policy π(a|s)
        '''
        #self.reset()
        self.render(ax)
        for s in self.S:
            if s not in self.s_termination:
                cell = self.state2cell(s)
                a = pi[s].argmax()
                next_cell = self.directs[a]*.25
                ax.arrow(cell[1]+.5, cell[0]+.5, 
                        next_cell[1], next_cell[0],
                        width=.01, color='k')
        ax.set_title('Policy')

    def show_v(self, ax, V):
        '''Visualize the value V(s) for each state given a policy
        '''
        v_mat = V.reshape([frozen_lake.n_row, frozen_lake.n_col])
        sns.heatmap(v_mat, cmap=viz.RedsMap, ax=ax,
                    lw=.5, linecolor=[.9]*3, cbar=False)
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=v_mat.shape[0], color='k',lw=5)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=v_mat.shape[1], color='k',lw=5)
        for s in self.S:
            if s not in self.s_termination:
                    cell = self.state2cell(s)
                    v = V[s].round(2)
                    ax.text(cell[1]+.15, cell[0]+.65,
                            str(v), color='k',
                            fontweight='bold', fontsize=8)
        ax.set_title('Value')
        ax.set_axis_off()
        ax.set_box_aspect(1)

    # ------------ interact with the environment ----------- #
    
    def step(self, act):
        '''Update the state of the environment
        '''
        p_s_next = self.p_s_next(self.state, act)
        self.state = self.rng.choice(self.S, p=p_s_next)
        self.curr_cell = self.state2cell(self.state)
        rew, self.done = self.r(self.state)
        self.act = None 
        return self.state, rew, self.done
    
################################################
#                                              #
#                TWO STAGE TASK                # 
#                                         @ZF  #
################################################

class two_stage_task:
    '''The two stage task

    The task reported in Daw et al., 2011 is a
    two-stage MDP. The task is written in the gym
    format. Here we will define the 4-tuple
    for this MDP (S, A, T, R)

    S: the state space, 
    A: the action space, 
    P: the transition fucntion, 
    R: the reward function,
    '''
    nS = 3 
    nA = 3

    def __init__(self, rho=.7, seed=2023):
        self.rho   = rho  # transition probability
        self.rng   = np.random.RandomState(seed)
        # define MDP 
        self._init_S()
        self._init_A()
        self._init_P()
        self._init_R()

    # -------- Define the task -------- #
        
    def _init_S(self):
        self.S = [0, 1, 2]

    def _init_A(self):
        self.A = [0, 1]

    def _init_P(self):
        '''The transition function

        The transition matrix is:

                 s0     s1      s2      
        s0-a0    0      t       1-t   
        s0-a1    0      1-t     t  

        s1-a0    1      0       0        
        s1-a1    1      0       0   
    
        s2-a0    1      0       0      
        s2-a1    1      0       0    
        '''
        self.P = np.zeros([self.nS, self.nA, self.nS])
        # state == 0 
        self.P[0, 0, :] = [0, self.rho, 1-self.rho]
        self.P[0, 1, :] = [0, 1-self.rho, self.rho]
        # state != 0 
        self.P[1:, :, 0] = 1
        # common state 
        self.common_state = 1 if self.rho > .5 else 2
        def p_s_next(s, a):
            return self.P[s, a, :].copy()
        self.p_s_next = p_s_next

    def _init_R(self):
        '''The reward function

            the probability of getting reward

                    p(r|s, a)      
            s0-a0      0              
            s0-a1      0             

            s1-a0     .9               
            s1-a1     .4             

            s2-a0     .1           
            s2-a1     .6        
        '''
        def r_fn(s, a):
            r_mat = np.zeros([3, 2])
            r_mat[1, 0] = .9
            r_mat[1, 1] = .4
            r_mat[2, 0] = .1 
            r_mat[2, 1] = .6
            p = r_mat[s, a]
            return self.rng.choice([0, 1], [1-p, p])
        self.r_fn = r_fn
    
    # -------- Run the task -------- #

    def reset(self):
        '''Reset the task, always start with state=0
        '''
        self.s = 0
        self.t = -1
        self.r = 0 
        info = {'stage': 0}
        return self.s, info

    def render(self, ax):
        occupancy = np.zeros([3, 5])
        occupancy[1, 1] = 1
        occupancy[1, 3] = 1
        cmaps = [viz.GreensMap, viz.RedsMap, viz.BluesMap]
        sns.heatmap(occupancy, cmap=cmaps[self.s],
                    vmin=0, vmax=1, cbar=False,
                    ax=ax)
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=occupancy.shape[0], color='k',lw=5)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=occupancy.shape[1], color='k',lw=5)
        ax.set_title(f'State: {self.s}, Reward: {self.r}')
        ax.set_axis_off()
        ax.set_box_aspect(3/5)

    def step(self, a):
        '''For each trial 

        Args:
            a: take the action conducted by the agent 

        Outputs:
            s_next: the next state
            rew: reward 
            info: some info for analysis 
        '''
        # Rt(St, At)
        self.r = self.r_fn[self.s, a]
        # St, At --> St+1 
        s_next = self.rng.choice(self.nS, p=self.T[self.s, a, :])
        # if the state is common 
        if s_next != 0: self.t += 1
        # if it is the end of the trial
        done = 1 if s_next == 0 else 0
        # info
        info = {
            'stage': 1 if s_next==0 else 0,
            'common': 'common' if a+1 == s_next else 'rare',
            'rewarded': 'rewarded' if self.r else 'unrewarded',
        }
        # now at the next state St
        self.s = s_next 
        return s_next, self.r, done, info
    
if __name__ == '__main__':

    env = frozen_lake()
    env.reset()
    env.step(2)

    print(1)
        
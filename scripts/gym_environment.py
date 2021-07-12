import numpy as np
import gym
from tensorforce import Environment
from tf_environment import action_values

"""
Gym environment that operates f1tenth_gym
Necessary for custom rewards and interface differences
Supports parallel execution with multiple instances at the same time
State is now self-contained instead of pass-by-reference from train.py
The original tf_environment is then used for testing & visualization
"""
class Gym_Environment(Environment):
    def __init__(self, main_state):
        #Initiate f1tenth_gym
        self.configs = main_state.configs
        self.main_state = main_state
        self.gym_env = gym.make('f110_gym:f110-v0', map='berlin', map_ext='.png', num_agents = 1)
        #Generate actions
        self.agent_actions = action_values(main_state.configs)
        #Load waypoints for accurate lap counting
        #and reward shaping
        #The agent always spawns at 0,0. The map orig changes between maps
        self.waypoints = main_state.configs["waypoints"]
        self.waypoints = np.asarray(self.waypoints)
        self.waypoints = self.waypoints.astype("float64")
        self.num_waypoints = self.waypoints.shape[0]
        self.resolution = main_state.configs["map_resolution"]
        self.map_orig = np.asarray(main_state.configs["map_orig"])
        #Translate waypoints to continuous coordinates from pixels
        for i in range(self.num_waypoints):
            self.waypoints[i, :] *= self.resolution
            self.waypoints[i, :] += self.map_orig
        #State variables
        self.cur_waypoint = 0
        self.prev_waypoints = 0
        self.odom = np.zeros(5)
        self.lap_finish = False
        self.turned_back = False
        self.crashed = False
        self.prev_distance = 0.0
        self.cur_distance = 0.0
        self.steps = 0
    
    """
    Mandatory action space description to be recognized by Tensorforce
    The int actions start from 0
    """
    def actions(self):
        agent_actions = {
            "velocity": {"type": "int", "num_values": self.configs["NUM_VEL_CHOICES"]},
            "turning_angle": {"type": "int", "num_values": self.configs["NUM_TURN_ANG"]}
        }
        return agent_actions
    
    """
    Mandatory state space description to be recognized by Tensorforce
    """
    def states(self):
        return {
            'odom': {'type':'float', 'shape':[5]},
            'laser_scan': {'type':'float', 'shape':[1080]}
        }
    
    """
    The Tensorforce reset supplies no arguments,
    but the f1tenth_gym env reset requires a position argument,
    which is for most purposes 0,0
    """
    def reset(self):
        observation, reward, done, info = self.gym_env.reset(np.zeros((1,3)))
        self.cur_waypoint = 0
        self.prev_waypoint = 0
        self.cur_distance = 0.0
        self.prev_distance = 0.0
        self.turned_back = False
        self.crashed = False
        self.lap_finish = False
        self.steps = 0
        return self.translate_state(observation)
    
    """
    Translate the f1tenth_gym state format
    into vectors for Tensorforce
    """
    def translate_state(self, observation):
        ego_id = 0
        assert len(observation['scans']) == 1
        lin_vel = np.sqrt(observation['linear_vels_x'][ego_id] ** 2 + \
                          observation['linear_vels_y'][ego_id] ** 2)
        odom = np.asarray([
            observation['poses_x'][ego_id],
            observation['poses_y'][ego_id],
            observation['poses_theta'][ego_id],
            lin_vel,
            observation['ang_vels_z'][ego_id]
        ])
        cur_state = {
            'odom': odom,
            'laser_scan': observation['scans'][ego_id]
        }
        return cur_state

    """
    Execute the action and determine waypoint position
    """
    def execute(self, actions):
        #Execute the step in the gym environment
        for i in range(5):
            #observation, reward, done, info = \
            #    self.gym_env.step(np.asarray( \
            #    [[self.agent_actions['turning_angle'][actions['turning_angle']], \
            #    self.agent_actions['velocity'][actions['velocity']]]]))
            observation, reward, done, info = \
                self.gym_env.step(np.asarray( \
                [[self.agent_actions['turning_angle'][actions['turning_angle']], \
                3.0]]))
            self.steps += 1
            if(done):
                break
        cur_state = self.translate_state(observation)
        #Done would only be true if crashed
        self.crashed = done
        #Determine the current waypoint
        distances = self.waypoints - cur_state['odom'][0:2]
        distances = np.sum(np.square(distances), axis=1)
        cur_wp = np.argmin(distances)
        self.prev_waypoint = self.cur_waypoint
        self.cur_waypoint = cur_wp
        #Decide distance to next waypoint for reward shaping
        next_wp = (cur_wp + 1) % self.num_waypoints
        self.prev_distance = self.cur_distance
        self.cur_distance = distances[next_wp]

        #Check if lap finished
        if((self.cur_waypoint == 0)\
            and (self.prev_waypoint==\
            self.num_waypoints - 1)):
            self.lap_finish = True
            done = True
        #Check if car had turned back
        if((not self.lap_finish)\
            and (self.cur_waypoint<\
            self.prev_waypoint)):
            self.turned_back = True
        if((self.prev_waypoint==0)\
            and (self.cur_waypoint > 2)):
            self.turned_back = True
        done = done or self.turned_back
        #Calculate a custom reward
        reward = self.reward()
        return cur_state, done, reward
    
    def reward(self):
        if(self.turned_back):
            if(self.main_state.verbose):
                print("Turned_back, reset simulation")
            return -50
        
        reward = 0.0
        if(self.cur_waypoint > self.prev_waypoint):
            reward = 100
            if(self.main_state.verbose):
                print("Reward for reaching waypoint " + str(self.cur_waypoint))
            
        if((self.main_state.ds_reward) and\
            (self.cur_waypoint == self.prev_waypoint)):
            if(self.prev_distance!=0.0):
                reward = (self.prev_distance - self.cur_distance) * 5
        
        if(self.lap_finish):
            #The default time step of f1tenth_gym is 0.01
            reward = (40-self.steps * 0.01)*self.configs["RW_MULT"]
            if(self.main_state.verbose):
                print(self.steps * 0.01)
        elif(self.crashed):
            reward = -1
        if(reward==0.0):
            reward = self.main_state.default_reward
        return reward
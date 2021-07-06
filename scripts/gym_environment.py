import gym
from tensorforce import Environment
from tf_environment import action_values

class Gym_Environment(Environment):
    def __init__(self, main_state):
        self.configs = main_state.configs
        self.main_state = main_state
        self.gym_env = gym.make('f110_gym:f110-v0')
        self.agent_actions = action_values(main_state.configs)
    
    def actions(self):
        agent_actions = {
            "velocity": {"type": "int", "num_values": self.configs["NUM_VEL_CHOICES"]},
            "turning_angle": {"type": "int", "num_values": self.configs["NUM_TURN_ANG"]}
        }
        return agent_actions
    
    def states(self):
        return {
            'odom': {'type':'float', 'shape':[5]},
            'laser_scan': {'type':'float', 'shape':[1080]}
        }
    
    def reset(self):
        self.gym_env.reset(np.zeros(2))
    
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


    def execute(self, actions):
        #Execute the step in the gym environment
        observation, reward, done, info = \
            self.gym_env.step(np.asarray( \
            [agent_actions['turning_angle'][actions['turning_angle']], \
            agent_actions['velocity'][actions['velocity']]]))
        cur_state = self.translate_state(observation)
        return cur_state, done, reward
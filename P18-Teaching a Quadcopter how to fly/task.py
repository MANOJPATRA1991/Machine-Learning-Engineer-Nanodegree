import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent"""
    def __init__(self, 
                 init_pose=None, 
                 init_velocities=None, 
                 init_angle_velocities=None, 
                 runtime=5., 
                 target_pos=None):
        """
            Initialize a Task object
            Args:
                init_pose: Initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
                init_velocities: Initial velocity of the quadcopter in (x,y,z) dimensions
                init_angle_velocities: Initial radians/second for each of the three Euler angles
                runtime: Time limit for each episode
                target_pos: Target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        
        # Save the start position (x,y,z) in an instance variable
        self.start_pos = self.sim.pose[:3]
        
        # For each timestep of the agent, we step the simulation action_repeat timesteps
        self.action_repeat = 3

        # 'x', 'y', 'z', 'phi', 'theta', 'psi' -> 6 
        # 'x_velocity', 'y_velocity', 'z_velocity' -> 3
        # 'phi_velocity', 'theta_velocity', 'psi_velocity' -> 3
        self.state_size = self.action_repeat * (6 + 3 + 3)
        
        # Minimum action value
        self.action_low = 0
        
        # Maximum action value
        self.action_high = 900
        
        # 4-dimensional action space, with one entry for each rotor
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current position of simulator to return reward"""
        penalty, reward = 0, 0
        
        current_position = self.sim.pose[:3]
        
        # Penalty for Euler angles to consider stable take up
        penalty += abs(self.sim.pose[3:6]).sum()
        
        # Penalty for distance from target, to make sure the agent moves towards the target
        penalty += abs(current_position[0] - self.target_pos[0])**2 
        penalty += abs(current_position[1] - self.target_pos[1])**2
        
        # 10 times more penalty along z-axis
        penalty += 10*abs(current_position[2] - self.target_pos[2])**2
        
        # Penalty for velocity and residual distance
        penalty += abs(abs(current_position - self.target_pos).sum() - abs(self.sim.v).sum())
                       
        distance = np.sqrt((current_position[0] - self.target_pos[0])**2 + \
                           (current_position[1] - self.target_pos[1])**2 + \
                           (current_position[2] - self.target_pos[2])**2)
                       
        
        # Extra reward for flying near the target
        if distance < 10:
            reward += 1000
        
        # Constant reward for flying
        reward += 100
                      
        return reward - penalty * 0.0002

    def step(self, rotor_speeds):
        """
            Uses action to obtain next state, reward, done
            Args:
                rotor_speeds: A list of floating-point numbers representing the speed of each rotor
            Returns:
                next_state: Next state
                reward: Reward as a result of current action taken
                done: Boolean indicating the end of an episode
            
        """
        reward = 0
        pose_all = []
        
        # For each timestep of the agent, we step the simulation action_repeat timesteps
        for _ in range(self.action_repeat):
            # Update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            
            reward += self.get_reward() 
            pose_all.append(self.current_state())
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

   
    def current_state(self):
        """
            Returns information about current position, velocity and angular velocity
            Returns:
                A numpy array of 9 items
        """
        state = np.concatenate([np.array(self.sim.pose),
                                np.array(self.sim.v),
                                np.array(self.sim.angular_v)])
        return state
                       
    def reset(self):
        """
            Reset the sim to start a new episode
            Returns:
                A numpy array of 9 items
        """
        self.sim.reset()
        state = np.concatenate([self.current_state()] * self.action_repeat) 
        return state
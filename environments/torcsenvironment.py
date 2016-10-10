from gym_torcs import TorcsEnv
import numpy as np
import cv2
imwrite = cv2.imwrite

class TorcsEnvironment(object):
    def __init__(self, vision=True, action_space=1, observation_dims=[64,64], n_action_repeat = 2):
        #set throttle to be true if 2dim action space is expected
        self.env = TorcsEnv(vision=vision, throttle=False)
        self.action_space = action_space
        self.observation_dims = observation_dims
        self.episode_count = 10
        self.max_steps = 50
        self.reward = 0
        self.done = False
        self.step = 0
        self.n_action_repeat = n_action_repeat


    def new_game(self):
        self.ob = self.env.reset(relaunch=True)
        #print(self.ob)
        self.total_reward = 0.
        return self.preprocess(self.ob.img), self.total_reward, False

    def make_step(self, action, is_training):
        if action == -1:
          # Step with random action
          action = np.tanh(np.random.randn(self.action_space))
          print("random action:",action)

        cumulated_reward = 0

        for _ in range(self.n_action_repeat):
          screen, reward, terminal, _ = self.env.step(action)
          cumulated_reward += reward


        return self.preprocess(screen.img), reward, terminal, {}

    def preprocess(self,raw_screen):
        y = 0.2126 * raw_screen[0] + 0.7152 * raw_screen[1] + 0.0722 * raw_screen[2]
        y = y.astype(np.uint8)
        #print(y.shape)
        y_screen = cv2.resize(y, (self.observation_dims[0],self.observation_dims[1]))
        
        return y_screen

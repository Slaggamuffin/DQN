#!/usr/bin/env python
# coding: utf-8

# In[7]:


import gym
import csv
import copy
import random
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torch.autograd import Variable
import numpy as np
import cv2
from IPython.display import clear_output
import time


# In[2]:


class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self,env_name, lr=0.01):
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        #self.device = torch.device("cpu")
        
        #Replay memory
        self.replay_memory = []
        self.replay_size = 128
        self.capacity = 10000
        
        #State Variables
        self.ROWS = 160
        self.COLS = 240
        self.REM_STEP = 4
        self.EPISODES = 4
        self.image_memory = np.zeros((self.REM_STEP,self.ROWS, self.COLS))
        
        #Greedy Variables
        self.epsilon = 0.95
        self.eps_decay = 0.99
        self.gamma = 0.99

        # Model variables    
        self.action_dim = self.env.action_space.n
        self.linear_input=147264
        #self.linear_input = self.out_rows*self.out_cols*16      
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=self.REM_STEP
                            ,out_channels=64
                            ,kernel_size=4
                            ,stride=2
                        ),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(
                            in_channels=64
                            ,out_channels=64
                            ,kernel_size=3
                            ,stride=2
                            ),
                        torch.nn.ReLU(),
                        torch.nn.Flatten(),
                        torch.nn.Linear(147264,512),
                        torch.nn.ReLU(),
                        torch.nn.Linear(512,128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128,self.action_dim)          
                    )
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
         
        
    # Model functions  
    def update(self, states, targets):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(states).to(self.device))
        loss = self.criterion(y_pred, Variable(torch.Tensor(targets).to(self.device)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state).to(self.device))

        
    def replay(self):
        ''' Add experience replay to the DQL network class.'''
        if len(self.replay_memory) >= self.replay_size:
            # Sample experiences from the agent's memory
            data = random.sample(self.replay_memory, self.replay_size)
            states = []
            targets = []
            for state, action, next_state, reward, done in data:
                states.append(state.squeeze(0))
                q_values = self.predict(state).squeeze(0)
                if done:
                    q_values[action] = reward
                else:
                    q_values_next = self.predict(next_state)
                    q_values[action] = reward + self.gamma * torch.max(q_values_next).item()

                targets.append(q_values.tolist())
            
            self.update(states, targets)
            
    def memory_append(self,memory):
        """ Adds a maximum capacity to replay memory """
        count = 0
        
        if len(self.replay_memory) < self.capacity:
            self.replay_memory.append(memory)
        else:
            self.replay_memory[count % self.capacity] = memory #if replay memory is over capacity will start to re-write
        count += 1
            
            
            
 # State functions
    def imshow(self, image, rem_step=0):
        cv2.imshow(env_name+str(rem_step), image[rem_step,...])
        cv2.waitKey(200)
        cv2.destroyAllWindows()
        return
            
       
    def GetImage(self):
        img = self.env.render(mode='rgb_array')
  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255

        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0,:,:] = img_rgb_resized
    
        #self.imshow(self.image_memory,0)
        return np.expand_dims(self.image_memory, axis=0)
    
    def reset(self):
        self.env.reset()
        for i in range(self.REM_STEP):
            state = self.GetImage()
            
        return state

    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.GetImage()
        return next_state, reward, done, info

# Action function
    def act(self,state):
        q_values = self.predict(state)
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = torch.argmax(q_values).item()
        return action
    
# Main loop
    def run(self):
        final = []
        for episode in range(self.EPISODES):
            # Get_state
            state = self.reset()
            done = False
            count = 0
            sum_total_replay_time = 0
            while not done:               
                # Select action via Greedy_strategy
                action = self.act(state)
                # Get next_state
                next_state, reward, done, info = self.step(action)
                self.memory_append((state, action, next_state, reward, done))
                # predict and target q_values and update them
                t0=time.time()
                self.replay()
                t1=time.time()
                sum_total_replay_time+=(t1-t0)
                count += 1
            
                if done:
                    final.append(count)
                    break
                    
                state = next_state    
                
            #update epsilon        
            self.epsilon = max(self.epsilon * self.eps_decay, 0.01)
            print("episode: {}, total reward: {}".format(episode, count))
            print("replay time: {}".format(sum_total_replay_time/count))
        self.env.close()  
        return final
        


# In[3]:


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = DQN(env_name)
    episodes = agent.run()
    
'''
Credits: Alberto Castrignanò, s281689, Politecnico di Torino
'''

'''
imports here
common:

import numpy as np
import gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
import PIL.Image as Image
import random
import pygame
import time
import tensorflow as tf
import math
import subprocess
import xml.etree.ElementTree as ET
import tensorflow as tf
import matplotlib.pyplot as plt
'''

import pygame
import time

'''
put here variables useful for the environment but not outside
common:
world size, actors stats(speed, acceleration, position, angles), bounds
'''

class EnvironmentName():
    '''
    -   width and height will be necessary to generate images for the neural network
    -   lr will be used for the neural network
    -   gamma is the gamma parameter in the a2c algorithm
    -   preload_ parameters must be set together to restore a previous condition:
        preload_model_weights = x
        preload_model_scores = y
        preload_observations = True
        N.B. x and y must be multiple of 200(check if this number is coherent with the one in train.py)
    -   output_dir will be the name where data mentioned above will be saved
    '''
    def __init__(self, width=300, height=300, lr=0.001, gamma=0.99,
    preload_model_weights=None, preload_model_scores=None, preload_observations=None, output_dir="output_dir_name"):
    #other parameters needed by the environment can be added here
        self.num_continue = 0 #change with number of needed continue
        self.num_discrete = 0 #change with number of needed discrete actions
        self.range_continue = None #put here a list of bounds for the continue actions, e.g.:
        #self.range_continue = [(-1,1),(-1,1)] -> both continue actions will be bounded inside (-1,1)
        self.dim_discrete = None #put here a list of sizes for the discrete actions, e.g.:
        #self.dim_discrete = [2,3] -> first action will have two possible values, second will have three possible values
        self.epochs = 300 #put here number of epochs to run
        self.iterations = 1001 #put here number of max iterations for a single epoch
        self.width = width
        self.height = height
        self.channels = 3 #for RGB images
        self.lr = lr
        self.gamma = gamma
        self.preload_model_weights = preload_model_weights
        self.preload_model_scores = preload_model_scores
        self.preload_observations = preload_observations
        self.output_dir = output_dir
    
    def reset(self):
        observation = None
        #observation must be a numpy array that complies with the shape (1,self.width,self.height,self.channels), e.g.:
        #observation = tf.keras.preprocessing.image.img_to_array(image).reshape((1,self.width,self.height,3))
        #observation = np.asarray(image).reshape((1,self.width,self.height,3)).copy()
        return observation
    
    def render(self):
        image = None
        #image must be a PIL.Image object, e.g.:
        #image = Image.fromarray(array, 'RGB').copy().resize([self.width,self.height])
        return image
    
    def adapt_actions(self, discrete_actions=None, continue_actions=None):
        #the training process will provide this function two tensors:
        #discrete_actions will contain a list of size (num_discrete,1)
        #continue_actions will contain a list of size (num_continue,1)
        #here you can put together them in a way that will comply with how the step function accepts the actions, e.g.:
        #return [discrete_actions, continue_actions]
        #return discrete_actions
        #return continue_actions
        #return discrete_actions+continue_actions
        return None
    
    def step(self, action):
        #action will be structured in the way defined in adapt_actions
        observation = None
        #observation must be a numpy array that complies with the shape (1,self.width,self.height,self.channels), e.g.:
        #observation = tf.keras.preprocessing.image.img_to_array(image).reshape((1,self.width,self.height,3))
        #observation = np.asarray(image).reshape((1,self.width,self.height,3)).copy()
        reward = 1
        #reward depends on the environment mechanisms
        done = False
        #done depends on whether the environment has reached a final state or not
        return observation, reward, done, None
        #last thing returned can be useful infos. until now infos use has not been implemented

if __name__=="__main__":
    images = []

    env = EnvironmentName()
    env.reset()
    observation = env.render()
    images.append(observation)

    screen = pygame.display.set_mode((env.width,env.height))
    pygame.display.flip()

    done = False
    while not done:
        action = None
        observation, reward, done, _ = env.step(action) #this observation is a numpy array
        observation = env.render() #this observation is a PIL.Image object
        images.append(observation)
    
    print("Press any key!")
    input() #press a key and then get prepared to see the video...
    print("wait 5 seconds...")
    time.sleep(5) #...after 5 seconds
    
    for image in images:
        raw = image.tobytes("raw", "RGB")
        pygame_surface = pygame.image.fromstring(raw, (env.width,env.height), "RGB") 
        screen.blit(pygame_surface, (0,0))
        #pygame.display.update()
        pygame.display.flip()
        time.sleep(0.1) #change accordingly on wanted visualization speed
        for event in pygame.event.get(): #avoids freezing the pygame window
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    
    pygame.quit()
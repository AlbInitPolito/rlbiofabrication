'''
Credits: Alberto Castrignanò, s281689, Politecnico di Torino
'''

import numpy as np
import pygame
import time
import env.test_env.TestEnv as tenv
import env.test_env.CartPoleEnv as cpenv
import tensorflow as tf
from model import ActorCritic
import os
from watch_images import generate_video_from_array

#use cpu if problems while testing on laptop
tf.config.set_visible_devices([], 'GPU') 
print(os.getcwd())
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
setup the environment
choose the environment that you want!
'''
#env = tenv.TestEnv()
env = cpenv.CartPoleEnv()
observation = env.reset() #observations from step and reset functions must be returned as numpy array of shape (1,size,size,ch)
observation = observation/255
observation = tf.convert_to_tensor(observation)

observations = []
observations_to_save = []
'''
setup the neural network
'''
num_continue = env.num_continue
num_discrete = env.num_discrete
range_continue = env.range_continue
dim_discrete = env.dim_discrete

preload_model_weights = "out/"+env.output_dir+"/"+env.preload_model_weights+".h5"
model = ActorCritic(num_continue=num_continue,num_discrete=num_discrete,range_continue=range_continue,dim_discrete=dim_discrete)
model.build((1,env.width,env.height,env.channels))
model.summary()
if preload_model_weights:
    model.load_weights(preload_model_weights)

'''
setup the play
'''
GAMMA = env.gamma
screen_width = 400
screen_height = 400
step_time = []
model_time = []
full_observations = []
threshold = 25 #set the minimum that you want to consider as result
filename = "testplay" #SUBSTITUTE WITH A NAME FOR THE MP4 VIDEO

'''
start!
'''
for _ in range(10): #choose how many plays to watch
    Qval = 0
    while Qval<=threshold:
        elapsed_time = time.time()

        done = False
        rewards = []
        observations = []
        env.reset()
        observations.append(env.render())

        for i in range(env.iterations):
            temp_model_time = time.time()
            '''
            obtain actions
            '''
            discrete_actions, continue_actions, normals, value = model(observation)
            model_time.append(time.time()-temp_model_time)
            d_acts = []
            if discrete_actions:
                for da in discrete_actions:
                    probs = da[0].numpy().astype('float64')
                    action = np.random.choice(len(da[0]), size=1, p=(probs/sum(probs))) #obtain a random action based on the probability given by each discrete action
                    d_acts.append(action[0])
            temp_step_time = time.time()
            '''
            act on the environment
            '''
            observation, reward, done, info = env.step(env.adapt_actions(d_acts, continue_actions))
            step_time.append(time.time()-temp_step_time)
            '''
            store the observation
            '''
            observations.append(env.render())
            observation = observation/255
            observation = tf.convert_to_tensor(observation)
            rewards.append(reward)
            if done:
                break
        '''
        compute the play score
        '''
        Qval = 0
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
        if Qval>threshold:
            print("Qval: ",Qval)
            full_observations = full_observations+observations

'''
watch the plays
'''
generate_video_from_array(full_observations,filename,env.width,env.height)
print("OK")
print("Press any key!")
input() #press a key and then get prepared to see the video...
print("wait 5 seconds...")
time.sleep(5) #...after 5 seconds

screen = pygame.display.set_mode((screen_width,screen_height))
pygame.display.flip()
for image in full_observations:
    raw = image.resize((screen_width,screen_height)).tobytes("raw", "RGB")
    pygame_surface = pygame.image.fromstring(raw, (screen_width,screen_height), "RGB") 
    screen.blit(pygame_surface, (0,0))
    #pygame.display.update()
    pygame.display.flip()
    time.sleep(0.05)

#while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

pygame.quit()

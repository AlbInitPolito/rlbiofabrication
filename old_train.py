'''
Credits: Alberto Castrignanò, s281689, Politecnico di Torino
'''

import numpy as np
import pygame
import time
import env.test_env.TestEnv as tenv
import env.test_env.CartPoleEnv as cpenv
import env.test_env.LunarLanderEnv as lenv
import env.Palacell.PalacellEnv as penv
import tensorflow as tf
from model import ActorCritic
import os
import watch_images as wi
import checks
import sys
import env.Palacell.PalacellEnv as penv

#use cpu if problems while testing on laptop
tf.config.set_visible_devices([], 'GPU')
print(os.getcwd())
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

verbose = 1

'''
setup the environment
choose the environment that you want!
'''
#env = tenv.TestEnv(200,200,0.01,0.99)
env = cpenv.CartPoleEnv(lr=0.001)
#env = lenv.LunarLanderEnv(lr=0.0001)
#env = penv.PalacellEnv(lr = 0.001)
checks.check_env(env)
observation = env.reset() #observations from step and reset functions must be returned as numpy array of shape (1,size,size,ch)


'''
setup the neural network
'''
num_continue = env.num_continue
num_discrete = env.num_discrete
range_continue = env.range_continue
dim_discrete = env.dim_discrete
epochs = env.epochs
iterations = env.iterations
width = env.width
height = env.height
channels = env.channels

#load weights, scores, observations
if env.preload_model_weights and env.preload_model_scores and env.preload_observations:
    preload_model_weights = "out/"+env.output_dir+"/model_at_epoch_"+str(env.preload_model_weights)+".h5"
    preload_model_scores = "out/"+env.output_dir+"/scores_at_epoch_"+str(env.preload_model_scores)+".npy"
    preload_observations = "out/"+env.output_dir+"/observations.npy"
elif env.preload_model_weights:
    preload_model_weights = "out/"+env.output_dir+"/model_at_epoch_"+str(env.preload_model_weights)+".h5"
else:
    preload_model_weights = None
    preload_model_scores = None
    preload_observations = None

model = ActorCritic(num_continue=num_continue,num_discrete=num_discrete,range_continue=range_continue,dim_discrete=dim_discrete)
model.build((1,width,height,channels))
model.summary()

'''
setup training
'''
starting_epoch = -1
save_every_epoch = 200
GAMMA = env.gamma

step_time = []
model_time = []
scores = [-10000000]
observations = []
observations_to_save = []

output_dir = env.output_dir
if not os.path.exists("out/"+output_dir):
    os.makedirs("out/"+output_dir)

if preload_model_weights and env.preload_model_scores and env.preload_observations:
    model.load_weights(preload_model_weights)
    starting_epoch = int(preload_model_weights.split("_")[-1].split(".")[0])
    scores = np.load(preload_model_scores).tolist()
    observations = np.load(preload_observations, allow_pickle=True).tolist()
elif preload_model_weights:
    model.load_weights(preload_model_weights)
    starting_epoch = int(preload_model_weights.split("_")[-1].split(".")[0])
optimizer = tf.keras.optimizers.Adam(learning_rate=env.lr)

'''
start training!
'''
print("GO!")
for j in range (starting_epoch+1,epochs):
    with tf.GradientTape() as tape:
        elapsed_time = time.time()

        done = False

        rewards = []
        values = []
        log_probs = []
        discrete_log_probs = []
        observations = []

        observation = env.reset()
        observation = observation/255
        observation = tf.convert_to_tensor(observation)
        observations.append(env.render())

        for iter in range(iterations):
            temp_model_time = time.time()
            '''
            obtain actions and generate log probs
            '''
            discrete_actions, continue_actions, normals, value = model(observation)
            values.append(value[0][0])
            model_time.append(time.time()-temp_model_time)

            #discrete actions
            d_acts = []
            if discrete_actions:
                for da in discrete_actions:
                    probs = da[0].numpy().astype('float64')
                    action = np.random.choice(len(da[0]), size=1, p=(probs/sum(probs))) #obtain a random action based on the probability given by each discrete action
                    discrete_log_prob = tf.math.log(da[0][action[0]]) #log of probability of given action
                    discrete_log_probs.append(discrete_log_prob)
                    d_acts.append(action[0])
            #continue actions
            if continue_actions:
                temp_cont = []
                for (i,nd) in enumerate(normals):
                    log_prob = nd.log_prob(continue_actions[i]) #log of probability of given action
                    log_probs.append(log_prob)
                    temp_cont.append(tf.clip_by_value(continue_actions[i],env.range_continue[i][0],env.range_continue[i][1]))
                
                continue_actions = tf.convert_to_tensor(temp_cont)

            temp_step_time = time.time()

            '''
            act on the environment
            '''
            observation, reward, done, info = env.step(env.adapt_actions(d_acts, continue_actions))
            step_time.append(time.time()-temp_step_time)
            rewards.append(reward)
            observations.append(env.render())
            observation = observation/255
            observation = tf.convert_to_tensor(observation)

            if done:
                break
            
            if iter%20==0 and verbose>1:
                print("Iteration: ", iter)
                print("Info: ",env._get_info())
                print("Elapsed: ",str(time.time()-elapsed_time))


        '''
        compute loss and backpropagte
        '''
        #compute Q-values
        #_, _, _, Qval = model(last_observation)
        Qval = 0
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
        
        #normalize Qvals
        #Qvals = (Qvals - np.mean(Qvals)) / (np.std(Qvals)+np.finfo(np.float32).eps.item())

        ##transform values, Qvals  into keras tensors   
        Qvals = tf.convert_to_tensor(Qvals)
        values = tf.convert_to_tensor(values)

        #compute advantage
        advantage = Qvals - values #ADVANTAGE IN TAKING ACTION A WRT ACTIONS IN THAT STATE (removes more noise)
        
        #compute actor loss
        if num_continue>0:
            log_probs = tf.convert_to_tensor(log_probs)
            actor_continue_loss = 0
            for i in range(num_continue):
                temp_log_probs = [-log_probs[j] for j in range(len(log_probs)) if (j+i)%num_continue==0]
                actor_continue_loss += tf.math.reduce_mean(temp_log_probs*advantage)
        if num_discrete>0:
            discrete_log_probs = tf.convert_to_tensor(discrete_log_probs)
            actor_discrete_loss = tf.math.reduce_mean([-discrete_log_probs[i]*advantage[int(i/num_discrete)] for i in range(len(discrete_log_probs))])
        
        #compute critic loss and sum up everything
        critic_loss = 0.5 * tf.math.reduce_mean(advantage**2) ##MEAN SQUARE ERROR
        ac_loss = critic_loss
        if num_continue>0:
            ac_loss += actor_continue_loss
        if num_discrete>0:
            ac_loss += actor_discrete_loss
        ac_loss = tf.convert_to_tensor(ac_loss)

        #compute gradients and backpropagate
        grads = tape.gradient(ac_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        '''
        print epoch stats and save weights, scores, observations
        '''
        print("epoch: ",j,", loss: ",ac_loss.numpy())
        print("Qval: ", Qvals[0].numpy())
        print("Last reward: ", Qvals[-1].numpy())
        if Qval>scores[-1]:
            scores.append(Qval)
            observations_to_save = observations_to_save+observations
            model.save_weights("out/"+output_dir+"/model_at_epoch_"+str(j)+".h5")
        elif Qval>=scores[int(len(scores)/2)]:
            observations_to_save = observations_to_save+observations
        print("elapsed time for epoch: ",time.time()-elapsed_time)
        print(len([i for i in grads if i==None]))
        inds = [i for (i,j) in enumerate(grads) if j==None]
        for i in inds:
            print(model.trainable_variables)
   
    '''
    save weights, scores, observations
    '''
    if j%save_every_epoch==0:
        np.save("out/"+output_dir+"/observations",observations_to_save)
        wi.generate_video("out/"+output_dir+"/observations")
        model.save_weights("out/"+output_dir+"/model_at_epoch_"+str(j)+".h5")
        np.save("out/"+output_dir+"/scores_at_epoch_"+str(j),scores)

np.save("out/"+output_dir+"/observations",observations_to_save)
model.save_weights("out/"+output_dir+"/last_model.h5")
np.save("out/"+output_dir+"/full_scores"+str(j),scores)

print("OK")
input()
time.sleep(5)
screen = pygame.display.set_mode((400,400))
pygame.display.flip()
for image in observations:
    raw = image.reshape((400,400)).tobytes("raw", "RGB")
    pygame_surface = pygame.image.fromstring(raw, (400,400), "RGB") 
    screen.blit(pygame_surface, (0,0))
    #pygame.display.update()
    pygame.display.flip()
    time.sleep(0.01)

#while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

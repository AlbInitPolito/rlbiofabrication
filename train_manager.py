'''
Credits: Alberto Castrignanò, s281689, Politecnico di Torino
'''

import numpy as np
import time
#import env.test_env.TestEnv as tenv
#import env.test_env.CartPoleEnv as cpenv
#import env.test_env.LunarLanderEnv as lenv
import env.Palacell.PalacellEnv as penv
import tensorflow as tf
from model import ActorCritic
import os
from train import Train
import itertools
from multiprocessing import Process

lr_list = [0.001, 0.0001]
gamma_list = [0.99, 0.95]

def parallel_train():
    #use cpu if problems while testing on laptop
    tf.config.set_visible_devices([], 'GPU')
    print(os.getcwd())
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    envs = []
    models = []
    trains = []

    for _ in range(len(lr_list)*len(gamma_list)):
        #choose the environment!
        #env = tenv.TestEnv(200,200,0.01,0.99)
        #env = cpenv.CartPoleEnv(lr=0.001)
        #env = lenv.LunarLanderEnv(lr=0.0001)
        env = penv.PalacellEnv()
        envs.append(penv.PalacellEnv())

        '''
        setup the neural network #TODO
        '''
        #load weights, scores, observations
        #if env.preload_model_weights:
        #    preload_model_weights = "out/"+env.output_dir+"/model_at_epoch_"+str(env.preload_model_weights)+".h5"
        #else:
        #    preload_model_weights = None

        num_continue = env.num_continue
        num_discrete = env.num_discrete
        range_continue = env.range_continue
        dim_discrete = env.dim_discrete
        width = env.width
        height = env.height
        channels = env.channels

        model = ActorCritic(num_continue=num_continue,num_discrete=num_discrete,range_continue=range_continue,dim_discrete=dim_discrete)
        model.build((1,width,height,channels))
        model.summary()

        #if preload_model_weights:
        #    model.load_weights(preload_model_weights)
        models.append(model)

    processes = []

    combs = itertools.product(lr_list, gamma_list)
    for i, (lr, gamma) in enumerate(combs):
        train = Train(envs[i], models[i], lr, gamma)
        proc = Process(target=train.train, args=[20, False])
        proc.start()
        #train.train(verbose=False, save_every=20)
        trains.append(train)
        processes.append(proc)
        
    
    while True:
        print("Select a number between 0 and ",len(trains)-1," to get infos: ")
        try:
            ind = input()
            if ind=="exit":
                for proc in processes:
                    proc.exit()
            else:
                for s in trains[ind].get_infos():
                    print(s)
        except Exception as e:
            print("insert a valid index!")
            print(e)

def single_train():
    #use cpu if problems while testing on laptop
    tf.config.set_visible_devices([], 'GPU')
    print(os.getcwd())
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    #choose the environment!
    #env = tenv.TestEnv(200,200,0.01,0.99)
    #env = cpenv.CartPoleEnv(lr=0.001)
    #env = lenv.LunarLanderEnv(lr=0.0001)
    env = penv.PalacellEnv()

    '''
    setup the neural network
    '''
    #load weights, scores, observations
    if env.preload_model_weights:
        preload_model_weights = "out/"+env.output_dir+"/model_at_epoch_"+str(env.preload_model_weights)+".h5"
    else:
        preload_model_weights = None

    num_continue = env.num_continue
    num_discrete = env.num_discrete
    range_continue = env.range_continue
    dim_discrete = env.dim_discrete
    width = env.width
    height = env.height
    channels = env.channels

    model = ActorCritic(num_continue=num_continue,num_discrete=num_discrete,range_continue=range_continue,dim_discrete=dim_discrete)
    model.build((1,width,height,channels))
    model.summary()

    if preload_model_weights:
        model.load_weights(preload_model_weights)

    for lr in lr_list:
        for gamma in gamma_list:
            env = penv.PalacellEnv(iters = 20)
            train = Train(env,model,lr,gamma)
            train.train()

if __name__=='__main__':
    single_train()
    #parallel_train() #DA FINIRE
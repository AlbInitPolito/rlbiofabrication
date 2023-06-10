'''
Credits: Alberto Castrignanò, s281689, Politecnico di Torino
'''

import numpy as np
import time
import tensorflow as tf
import os
import checks

class Train:
    def __init__(self, env, model, lr, gamma):
        self.env = env
        self.model = model
        self.lr = lr
        self.gamma = gamma

    def get_infos(self):
        strings = []
        strings.append("scores: ", str(self.scores))
        strings.append("losses: ", self.loss)
        strings.append("epoch: ", self.epoch)
        strings.append("env infos: ", self.env._get_info())
        strings.append("")
        return

    def train(self, save_every=10, verbose=True):
        env = self.env
        model = self.model
        lr = self.lr
        gamma = self.gamma
        #use cpu if problems while testing on laptop
        tf.config.set_visible_devices([], 'GPU')
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        #check for env integrity
        checks.check_env(env)

        '''
        setup the environment
        '''
        num_continue = env.num_continue
        num_discrete = env.num_discrete

        '''
        setup training
        '''
        GAMMA = gamma
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        epochs = env.epochs
        iterations = env.iterations

        step_time = []
        model_time = []
        self.scores = []
        self.loss = []

        output_dir = env.output_dir+"_"+str(lr)+"_"+str(gamma)
        if not os.path.exists("out/"+output_dir):
            os.makedirs("out/"+output_dir)

        for j in range (0,epochs):
            elapsed_time = time.time()
            self.epoch = j
            with tf.GradientTape() as tape:
                done = False

                rewards = []
                values = []
                log_probs = []
                discrete_log_probs = []

                observation = env.reset()
                observation = observation/255
                observation = tf.convert_to_tensor(observation)
                for iter in range(iterations):
                    '''
                    obtain actions and generate log probs
                    '''
                    discrete_actions, continue_actions, normals, value = model(observation)
                    values.append(value[0][0])

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

                        '''
                        act on the environment
                        '''
                        observation, reward, done, info = env.step(env.adapt_actions(d_acts, continue_actions))
                        rewards.append(reward)
                        observation = observation/255
                        observation = tf.convert_to_tensor(observation)

                        if done:
                            break

                        if iter%20==0 and verbose:
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
                if verbose:
                    print("epoch: ", j, ", loss: ", ac_loss.numpy(), " lr: ", lr, " gamma: ", gamma)
                    print("Qval: ", Qvals[0].numpy())
                    print("Last reward: ", Qvals[-1].numpy())
                    print("Elapsed epoch time: ",str(time.time()-elapsed_time))
                self.loss.append(ac_loss)
                if len(self.scores)==0 or Qval>self.scores[-1]:
                    self.scores.append(Qval)
                    model.save_weights("out/"+output_dir+"/model_at_epoch_"+str(j)+".h5")
                if verbose:
                    print(len([i for i in grads if i==None]))
                #inds = [i for (i,j) in enumerate(grads) if j==None]
                #for i in inds:
                #    print(model.trainable_variables)
        
            '''
            save weights, scores, observations
            '''
            if j%save_every==0:
                model.save_weights("out/"+output_dir+"/model_at_epoch_"+str(j)+".h5")
                np.save("out/"+output_dir+"/scores_at_epoch_"+str(j),self.scores)
                np.save("out/"+output_dir+"/losses_at_epoch_"+str(j),self.loss)

        model.save_weights("out/"+output_dir+"/last_model.h5")
        np.save("out/"+output_dir+"/full_scores",self.scores)
        np.save("out/"+output_dir+"/full_losses",self.loss)
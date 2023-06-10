'''
Credits: Alberto Castrignanò, s281689, Politecnico di Torino
'''

import numpy as np
import time
import tensorflow as tf
import os
import checks
from model import ActorCritic

class ProlifTrain:
    #def __init__(self, env, model, lr, gamma, id):
    def __init__(self, env, lr, gamma):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.scores = []
        self.losses = []
        self.loss = 0
        self.epoch = 0

    
    def get_infos(self):
        strings = ["inner: "]
        #strings.append("scores: "+str(self.scores))
        strings.append("losses: "+str(self.loss))
        strings.append("epoch: "+str(self.epoch))
        #strings.append("times: "+str(self.times))
        strings.append("env infos: "+self.env._get_info())

        strings.append("")
        for s in strings:
            print(s)
        return strings

    def train(self, save_every=10, verbose=True, starting_epoch=0, pipe=None):
        lr = self.lr
        gamma = self.gamma
        #use cpu if problems while testing on laptop
        tf.config.set_visible_devices([], 'GPU')
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        num_continue = self.env.num_continue
        num_discrete = self.env.num_discrete
        range_continue = self.env.range_continue
        dim_discrete = self.env.dim_discrete
        model = ActorCritic(num_continue=num_continue,num_discrete=num_discrete,range_continue=range_continue,dim_discrete=dim_discrete)
        self.model = model

        checks.check_env(self.env)

        num_continue = self.env.num_continue
        num_discrete = self.env.num_discrete
        self.env.save_performance([])

        GAMMA = gamma
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        epochs = self.env.epochs
        iterations = self.env.iterations

        output_dir = self.env.output_dir+"_"+str(lr)+"_"+str(gamma)
        if not os.path.exists("out/"+output_dir):
            os.makedirs("out/"+output_dir)

        self.model.build((1,self.env.width,self.env.height,self.env.channels))
        self.model.summary()
        if self.env.preload_model_weights:
            model.load_weights(self.env.preload_model_weights)
        if self.env.preload_losses:
            self.losses = np.load(self.env.preload_losses+".npy").tolist()
            self.losses.append(0)
        if self.env.preload_performance:
            performance_indexes = np.load(self.env.preload_performance+".npy", allow_pickle=True)
            self.env.load_performance(performance_indexes)
        if self.env.preload_data_to_save:
            self.env.load_data_to_save(np.load(self.env.preload_data_to_save+".npy", allow_pickle=True).item())

        print("GO inside: ")

        #for j in range (0,epochs):
        for j in range (starting_epoch+1,epochs):

            out_path = pipe.recv()

            elapsed_time = time.time()
            self.epoch = j
            with tf.GradientTape() as tape:
                done = False

                rewards = []
                values = []
                log_probs = []
                discrete_log_probs = []

                old_conf_path = self.env.configuration
                self.env.configuration = out_path
                observation = self.env.reset()
                self.env.configuration = old_conf_path
                observation = observation/255
                observation = tf.convert_to_tensor(observation)
                for iter in range(iterations):
                    
                    '''
                    obtain actions and generate log probs
                    '''
                    discrete_actions, continue_actions, normals, value = self.model(observation)
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
                            temp_cont.append(tf.clip_by_value(continue_actions[i],self.env.range_continue[i][0],self.env.range_continue[i][1]))
                        
                        continue_actions = tf.convert_to_tensor(temp_cont)

                    '''
                    act on the environment
                    '''
                    observation, reward, done, info = self.env.step(self.env.adapt_actions(d_acts, continue_actions))
                    rewards.append(reward)
                    observation = observation/255
                    observation = tf.convert_to_tensor(observation)

                    if done:
                        pipe.send("output/"+self.env.output_file+"_final_cell")
                        break

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
                grads = tape.gradient(ac_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                print("inner: epoch: ", j, ", loss: ", ac_loss.numpy(), " lr: ", lr, " gamma: ", gamma)
                self.get_infos()
                #print("Qval: ", Qvals[0].numpy())
                #print("Last reward: ", Qvals[-1].numpy())
                print("Elapsed epoch time: ",str(time.time()-elapsed_time))
                self.losses.append(ac_loss)
                #if len(self.scores)==0 or Qval>self.scores[-1]:
                if self.env.check_performance([Qvals[0].numpy(),time.time()-elapsed_time,ac_loss,j]):
                    #self.scores.append(Qval)
                    self.model.save_weights("out/"+output_dir+"/model_at_epoch_"+str(j)+"(best).h5")
                    np.save("out/"+output_dir+"/data_to_save_at_epoch_"+str(j)+"(best)",self.env.data_to_save())
                self.env.save_performance([Qvals[0].numpy(),time.time()-elapsed_time,ac_loss,j])
                if verbose:
                    print(len([i for i in grads if i==None]))

                
            if j%save_every==0:
                self.model.save_weights("out/"+output_dir+"/model_at_epoch_"+str(j)+".h5")
                np.save("out/"+output_dir+"/data_to_save_at_epoch_"+str(j),self.env.data_to_save())
                np.save("out/"+output_dir+"/performance_at_epoch_"+str(j),self.env.get_performance())
                #np.save("out/"+output_dir+"/scores_at_epoch_"+str(j),self.scores)
                np.save("out/"+output_dir+"/losses_at_epoch_"+str(j),self.losses)

        self.model.save_weights("out/"+output_dir+"/last_model.h5")
        #np.save("out/"+output_dir+"/full_scores",self.scores)
        np.save("out/"+output_dir+"/full_losses",self.losses)
        np.save("out/"+output_dir+"/last_data_to_save",self.env.data_to_save())
        np.save("out/"+output_dir+"/last_performance",self.env.get_performance())
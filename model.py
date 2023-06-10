'''
Credits: Alberto Castrignanò, s281689, Politecnico di Torino
'''

import tensorflow as tf
import tensorflow_probability as tfp

'''
Basic residual block.
'filters' is the number of filters applied in the convolutional layer.
'conv1x1' allow to add a 1x1 convolutional block instead of a simple skip connection
'''
class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, conv1x1=False):
      super().__init__()
      self.conv1 = tf.keras.layers.Conv2D(filters, padding="same", kernel_size=kernel_size, strides=strides, kernel_initializer = "glorot_uniform")
      self.conv2 = tf.keras.layers.Conv2D(filters, padding="same", kernel_size=kernel_size, kernel_initializer = "glorot_uniform")
      if conv1x1:
        self.conv1x1 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=strides, kernel_initializer = "glorot_uniform")
      else:
        self.conv1x1 = None
      self.bn1 = tf.keras.layers.BatchNormalization()
      self.bn2 = tf.keras.layers.BatchNormalization()
    
    def call(self, x):
      out = self.conv1(x)
      out = self.bn1(out)
      out = tf.keras.activations.relu(out)
      out = self.conv2(out)
      out = self.bn2(out)
      if self.conv1x1:
        out = tf.keras.layers.Add()([out,self.conv1x1(x)])
        #out = out+self.conv1x1(x)
      else:
        out = tf.keras.layers.Add()([out,x])
        #out = out+x
      return tf.keras.activations.relu(out)

'''
The Resnet block: each is composed of an arbitrary number of residual blocks
"filters" is the number of filters of convolutional layers in each residual block
(in this implementation, you can't obtain a resnet block made of residual blocks with a different number of filters)
"blocks_number" is the number of residual blocks composing the resnet block.
'''
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, blocks_number, downsample=True, **kwargs):
      super(ResnetBlock, self).__init__(**kwargs)
      self.residual_blocks = []
      for i in range(blocks_number):
        if i == 0 and downsample:
          self.residual_blocks.append(ResidualBlock(filters, strides=2, conv1x1=True))
        else:
          self.residual_blocks.append(ResidualBlock(filters))

    def call(self, x):
      for l in self.residual_blocks.layers:
        x = l(x)
      return x

'''
ResNet18: accepts an arbitrary-sized image, outputs a 'latent'-sized fully connected layer with relu activations.
Traditionally, it should output a number of softmax activations corresponding to the number of binary classes, given a classification task.
Here, the last layer is a dense so that we can get a latent representation of the input image, and elaborate it further, thus using the ResNet18 as a backbone.
'''
class ResNet18(tf.keras.Model):
    def __init__(self, latent=1000): #latent can be an hyperparameter
      super(ResNet18, self).__init__()
      self.block_a = tf.keras.Sequential([tf.keras.layers.Conv2D(32,kernel_size=7,strides=2,padding="same",kernel_initializer = "glorot_uniform"),
                                          tf.keras.layers.BatchNormalization(),
                                          tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")])
      self.block_b = ResnetBlock(32, 2, downsample=False)
      self.block_c = ResnetBlock(64, 2)
      self.block_d = ResnetBlock(128, 2)
      self.block_e = ResnetBlock(256, 2)
      self.dense = tf.keras.layers.Dense(latent,activation='relu',kernel_initializer = "glorot_uniform")
    
    def call(self, x):
      out = self.block_a(x)
      out = self.block_b(out)
      out = self.block_c(out)
      out = self.block_d(out)
      out = self.block_e(out)
      out = tf.keras.layers.GlobalAveragePooling2D()(out)
      out = self.dense(out)
      return out

'''
The encoder part of an autoencoder.
Should be trained with the decoder part by using the input images that will be used in the RL training, but should be sufficient
as long as it learns the necessary representations.
'''
class Encoder(tf.keras.Model):
  def __init__(self, latent=1000): #input_dim must be multiple of 16, latent is an hyperparameter
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(32,3,activation="relu",padding="same") # -> input/2, input/2, 32
    self.conv2 = tf.keras.layers.Conv2D(64,3,activation="relu",padding="same") # -> input/4, input/4, 64
    self.conv3 = tf.keras.layers.Conv2D(128,3,activation="relu",padding="same") # -> input/8, input/8, 128
    self.conv4 = tf.keras.layers.Conv2D(256,3,activation="relu",padding="same") # -> input/16, input/16, 256
    self.dense = tf.keras.layers.Dense(latent,activation="relu")
  
  def call(self, x):
    out = self.conv1(x)
    out = tf.keras.layers.AveragePooling2D(padding="same")(out)
    out = self.conv2(out)
    out = tf.keras.layers.AveragePooling2D(padding="same")(out)
    out = self.conv3(out)
    out = tf.keras.layers.AveragePooling2D(padding="same")(out)
    out = self.conv4(out)
    out = tf.keras.layers.AveragePooling2D(padding="same")(out)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = self.dense(out)
    return out

'''
The Actor part: it can be composed of multiple discrete or continue action layers. Each share a fully connected layer.
Discrete actions: represented with a fully connected layer with softmax activations, representing the probability of doing an action. Its output size is the number of actions that can be made.
Continue actions: represented with two fully connected layer, which output dimensions are the number of continuous actions that can be made.
                  One layer represents the mean, the other the variance, that will be used to sample a value in the normal distribution.
                  The normal distribution itself will be used to compute the loss for that action.
                  This approach corresponds to using a single multivariate normal distribution with covariances=0 (diagonal matrix)
'''
class PolicyHead(tf.keras.Model):
  def __init__(self, num_continue=0, num_discrete=0, range_continue=None, dim_discrete=None, hidden=256): #hidden is an hyperparameter
    super().__init__()

    if not isinstance(num_continue,int):
      raise TypeError("Number of continue actions must be integer")
    if not isinstance(num_discrete,int):
      raise TypeError("Number of discrete actions must be integer")
    if not isinstance(hidden,int):
      raise TypeError("Number of hidden dimensions must be integer")
    if num_continue == 0 and num_discrete == 0:
      raise ValueError("You need at least one kind of action!")

    self.dense1 = tf.keras.layers.Dense(hidden,activation="relu",kernel_initializer = "glorot_uniform")

    if num_continue>0:
      if range_continue == None:
        raise ValueError("You need to specify a list(tuple) in range_continue when having continue actions")
      if not isinstance(range_continue,list):
        raise TypeError("ranges of continue actions must be a list of tuples")
      if len(range_continue)!=num_continue:
        raise ValueError("range_continue must have a number of tuples equal to the number of continue actions")
      if not isinstance(range_continue[0],tuple):
        raise TypeError("ranges of continue actions must be a list of tuples")
      self.num_continue = num_continue
      self.range_continue = range_continue
      self.mu_dense = tf.keras.layers.Dense(num_continue, None, kernel_initializer = "glorot_uniform")
      self.sigma_dense = tf.keras.layers.Dense(num_continue, None, kernel_initializer = "glorot_uniform")
    else:
      self.mu_dense = None
      self.sigma = None
    
    if num_discrete>0:
      self.discrete_actions = []
      if dim_discrete==None:
        raise ValueError("You need to specify a list(int) in dim_discrete when having discrete actions")
      if not isinstance(dim_discrete,list):
        raise TypeError("discrete dimensions must be a list of integers")
      if len(dim_discrete)!=num_discrete:
        raise ValueError("dim_discrete must have a number of int equal to the number of discrete actions")
      if not isinstance(dim_discrete[0],int):
        raise TypeError("discrete dimensions must be a list of integers")
      for i in range(num_discrete):
        self.discrete_actions.append(tf.keras.layers.Dense(dim_discrete[i], activation="softmax", kernel_initializer = "glorot_uniform"))
    else:
      self.discrete_actions = None

  def call(self, x):
    out = self.dense1(x)

    if self.mu_dense:
      mu = self.mu_dense(out)
      sigma = self.sigma_dense(out)
      sigma = tf.keras.activations.softplus(sigma)+1e-5
      sigma = tf.clip_by_value(sigma, 1e-2, 1e3)
      continue_actions = []
      norm_functions = []
      for i in range(self.num_continue):
        norm = tfp.distributions.Normal(mu[0][i], sigma[0][i])
        action = tf.squeeze(norm.sample(1), axis=0)
        #action = tf.clip_by_value(action, self.range_continue[i][0], self.range_continue[i][1])
        norm_functions.append(norm)
        continue_actions.append(action)
    else:
      norm_functions = None
      continue_actions = None

    if self.discrete_actions:
      discrete_actions = []
      for dense in self.discrete_actions:
        out = tf.clip_by_value(dense(out),1e-2,1e3)
        discrete_actions.append(out)
    else:
      discrete_actions = None

    return discrete_actions, continue_actions, norm_functions

'''
The Critic part: it is composed of two fully connected layers, the first with relu activations.
The second layer is taken as it is, in order to output a value representing the Q-value of the state.
'''
class ValueHead(tf.keras.Model):
  def __init__(self, hidden=256): #hidden is an hyperparameter
    super().__init__()
    if not isinstance(hidden,int):
      raise TypeError("Number of hidden dimensions must be integer")
    self.dense1 = tf.keras.layers.Dense(hidden,activation="relu", kernel_initializer = "glorot_uniform")
    self.dense2 = tf.keras.layers.Dense(1,None, kernel_initializer = "glorot_uniform")

  def call(self,x):
    out = self.dense1(x)
    out = self.dense2(out)
    return out


'''
The ActorCrit network: backbone can be chosen (resnet, encoder, or a single fully connected layer).
'''
class ActorCritic(tf.keras.Model):
  def __init__(self, latent=1000, backbone="resnet", num_continue=0, num_discrete=0, range_continue=None, dim_discrete=None, hidden_actor=256, hidden_critic=256):
    super().__init__()

    if backbone=="resnet":
      self.backbone = ResNet18(latent)
    elif backbone=="encoder":
      self.backbone = Encoder(latent)
    else:
      raise ValueError("You need to specify either 'resnet' or 'encoder' as backbone")
    
    self.actor = PolicyHead(num_continue, num_discrete, range_continue, dim_discrete, hidden_actor)
    self.critic = ValueHead(hidden_critic)

  def call(self,x):
    repr = self.backbone(x)
    discrete_actions, continue_actions, norm_functions = self.actor(repr)
    value = self.critic(repr)
    return discrete_actions, continue_actions, norm_functions, value
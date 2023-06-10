#from model import ActorCritic, ValueHead
import numpy as np
import tensorflow as tf

from model import ActorCritic
from model import ResNet18

'''
model = ActorCritic(num_continue=1,num_discrete=1,range_continue=[(0.03,0.05)],dim_discrete=[2])
model.build((1,400,400,1))
model.summary()

model2 = ActorCritic(num_continue=6,num_discrete=2,range_continue=[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)],dim_discrete=[4,8])
model2.build((1,400,400,1))
model2.summary()

model3 = ActorCritic(backbone="encoder", num_continue=1,num_discrete=1,range_continue=[(0.03,0.05)],dim_discrete=[2])
model3.build((1,400,400,1))
model3.summary()

model4 = ActorCritic(backbone="none", num_continue=1,num_discrete=1,range_continue=[(0.03,0.05)],dim_discrete=[2])
model4.build((1,400,400,1))
model4.summary()
'''


model = ActorCritic(num_continue=6,num_discrete=2,range_continue=[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)],dim_discrete=[4,8])
model.build((1,400,400,3))
model.summary()

fake_input = np.zeros((1,400,400,3))
fake_input[0][0][0][0] = 1
'''
fake_input2 = np.zeros((1,400,400,3))
fake_input2[0][399][399][0] = 1
#print(fake_input2)
'''
discrete_actions, continue_actions, norm_functions, value = model(fake_input)
#result2 = model(fake_input2)

#print(result)
#print(result2)

'''
print(discrete_actions)
print("best action 0: " + str(discrete_actions[0].numpy().argmax()))
print("best action 1: " + str(discrete_actions[1].numpy().argmax()))
print("continuous action 0: " + str(continue_actions[0].numpy()))
print("continuous action 1: " + str(continue_actions[1].numpy()))
print("continuous action 2: " + str(continue_actions[2].numpy()))
print("continuous action 3: " + str(continue_actions[3].numpy()))
print("continuous action 4: " + str(continue_actions[4].numpy()))
print("continuous action 5: " + str(continue_actions[5].numpy()))
#print(norm_functions[0])
#print(norm_functions[1])
#print(norm_functions[2])
#print(norm_functions[3])
#print(norm_functions[4])
#print(norm_functions[5])
print("state value: " + str(value[0][0].numpy()))
print(value)
'''

#print(continue_actions)

'''
model = ActorCritic(num_continue=1,num_discrete=1,range_continue=[(0.03,0.05)],dim_discrete=[2])
model.build((1,400,400,3))
model.summary()

fake_input = np.zeros((1,400,400,3))
discrete_actions, continue_actions, normals, value = model(fake_input)
print(discrete_actions)
print(continue_actions)
print(normals)
print(value)

for _ in range(1000):
    result = model(fake_input)
'''
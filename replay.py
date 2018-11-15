

from collections import deque, Counter
import random
import numpy as np

class Replay: 
    
    def __init__(self, field, memory_size=1000, gamma = .90):
        self.field = field
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        
    def replay(self, model, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        minibatch = random.sample(self.memory, batch_size)
        
        # Number of actions
        nb_actions = model.get_output_shape_at(0)[-1]
        
        state_size = (1, self.field[0], self.field[1])
        env = np.zeros((batch_size, state_size[0], state_size[1], state_size[2]))
        env_next = np.zeros((batch_size, state_size[0], state_size[1], state_size[2]))
        actions, rewards, status = np.zeros((batch_size)), np.zeros((batch_size)), np.zeros((batch_size))
        
        
        count = 0
        for state, action, reward, next_state, done in minibatch:
            env[count] = state
            env_next[count] = next_state
            actions[count] = int(action)
            rewards[count] = reward
            status[count] = done
            count += 1
            
        # One-Hots
        rewards = rewards.repeat(nb_actions).reshape((batch_size, nb_actions))
        actions = self.convert_onehot(actions, batch_size, nb_actions)
        status = status.repeat(nb_actions).reshape((batch_size, nb_actions))
        
        # Normal prediction from the current env
        env_y = model.predict(env)
        
        # prediction from the next env
        env_next_y = model.predict(env_next)
        Qsa = np.max(env_next_y, axis=1).repeat(nb_actions).reshape((batch_size, nb_actions))
        
        # weired formular
        targets = (1 - actions) * env_y + actions * (rewards + self.gamma * (1 - status) * Qsa)
        return float(model.train_on_batch(env, targets))
    
    def convert_onehot(self, data, batch_size, nb_actions):
        data = np.cast['int'](data)
        result = np.zeros((batch_size, nb_actions))
        result[np.arange(batch_size), data] = 1
        return result
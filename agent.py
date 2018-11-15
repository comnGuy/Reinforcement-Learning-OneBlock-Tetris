# -*- coding: utf-8 -*-

from tetris import Tetris
from memory import ExperienceReplay
from replay import Replay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import matplotlib.pyplot as plt

class Agent:

    def __init__(self, game, model, action_range, field, memory=None, memory_size=1000, nb_frames=None, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon_range=[1., .01], epsilon_rate=0.99, reset_memory=False, observe=0, checkpoint=None):
        self.model = model
        self.game = game
        self.field = field
        self.memory_size = memory_size
        self.nb_frames = nb_frames
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_range = epsilon_range
        self.epsilon_rate = epsilon_rate
        self.reset_memory = reset_memory
        self.observe = observe
        self.checkpoint = checkpoint
        self.action_range = action_range
        self.loss = 0
        self.score_last_games = []
        self.ma_score_list = []
        
        self.replay = Replay(self.field, self.memory_size, gamma = self.gamma)
        
    def train(self):
        if type(self.epsilon_range) in {tuple, list}:
            delta =  ((self.epsilon_range[0] - self.epsilon_range[1]) / (self.nb_epoch * self.epsilon_rate))
            final_epsilon = self.epsilon_range[1]
            epsilon = self.epsilon_range[0]
        else:
            final_epsilon = self.epsilon_range
        
        
        for i in range(self.nb_epoch):
            # Play a whole game
            self.game.init_game()
            while not self.game.status():
                self.loss = 0
                old_env = self.game.get_env()
                if np.random.random() < epsilon:
                    action = int(np.random.randint(self.action_range))
                else:
                    action = int(np.argmax(self.model.predict(old_env)[0]))
                
                # Play
                self.game.act(action)

                # Save the SARSSARS
                self.replay.remember(old_env, action, self.game.reward(), self.game.get_env(), self.game.status()) # own
                
                # Learn and get the replay
                self.loss += self.replay.replay(self.model, self.batch_size)
            epsilon -= delta
            ma_score = self.moving_average(self.game.score)
            self.ma_score_list.append(ma_score)
            print("Epoch {:03d}/{:03d} | Epsilon {:.2f} | Loss {:.2f} | MA-Score {} | Score {} | Lines {}".format(i + 1, self.nb_epoch, epsilon, self.loss, ma_score, self.game.score, self.game.sum_lines))
            self.game.init_game()
        self.model.save_weights('weights/weights.dat')
            #print(self.game.get_env())
        
    def moving_average(self, score, moving_range = 100):
        if len(self.score_last_games) >= moving_range:
            self.score_last_games.pop(0)
        self.score_last_games.append(score)
        return int(sum(self.score_last_games)/len(self.score_last_games))
    
    def play(self, game, nb_epoch=10, epsilon=0., visualize=True):
        #self.check_game_compatibility(game)
        for epoch in range(nb_epoch):
            game.init_game()
            while not self.game.status():
                S = self.game.get_env()
                print(S)
                action = int(np.argmax(self.model.predict(S)[0]))
                self.game.act(action)
            print("Score: " + str(self.game.score))
                
    

    def plot_ma_score(self):
        plt.figure(figsize=(16,10))
        plt.title('Score Process', fontsize=22)
        plt.xlabel('#Games', fontsize=18)
        plt.ylabel('Mean Sliding Window of 100', fontsize=18)
        plt.plot(self.ma_score_list)
        plt.savefig('tetris_score_process.png')
        
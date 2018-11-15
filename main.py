# -*- coding: utf-8 -*-
# TODO evtl. mehr reinklatsch, Beispiel waren 4

from tetris import Tetris
from agent import Agent
from model import Models

# Height, Width
field = (11,6)
# Frames which we propagate
nb_frames = 1 
# Number of possible actions
nb_actions = 3


# Batch size
batch_size = 64
# Discounter
gamma = .95
# Number of games
number_games = 250


# Initiate Tetris
t = Tetris(field = field)
# Get model
models = Models(nb_frames, field[0], field[1], nb_actions)
model = models.get_model()


# Agent things
agent = Agent(t, model, nb_actions, field, nb_epoch = number_games, gamma=gamma, batch_size=batch_size)
agent.train()
agent.plot_ma_score()
agent.play(t, nb_epoch = 1)

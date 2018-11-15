# Reinforcement Learning - OneBlock-Tetris

My first successful Reinforcement Learning project is with a One-Block-Tetris game. This game has a field of 6 columns and 10 rows and the One-Block has a size of 2x2. 

![Score Process](tetris_score_process.png?raw=true "Score Process")

# Starting Code
```python
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
```

# Inspired
@farizrahman4u
@silvasur
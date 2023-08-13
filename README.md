# Jedi AI
A neural network that plays Jedi Knight: Jedi Academy but could easily be applied to other FPS games.

The AI works by pure imitation learning. Specifically it learns the stimulus -> response association
where the stimulus = game images and the response = game inputs.

In order to be performant a small neural network is used similar to that of the original DQN paper,
although reinforcement learning is not employed due to sample inefficiency. 

Instead there is a classification head for each key:

```
w, a, s, d, f, e, r, space, ctrl, mouse_left, mouse_middle, mouse_right
```

corresponding to:

```
forward, strafe left, backward, strafe right, duel challenge, grapple, release grapple,
switch saber style, alternate attack, attack, crouch, jump
```

along with a regression head for each mouse direction:

```
mouse_deltaX, mouse_deltaY
```

corresponding naturally to:

```
aim left/right, aim up/down
```

# Dependencies

Windows, Python, an FPS game, along with the following Python packages:

```
python -m pip install torch torchvision win32 mouse keyboard
```

# Recording Data

Run the game at 800x600 resolution and join a multiplayer server. Then:

```
mkdir data
mkdir data/train
python jka_recorder.py
```

Play the game as you normally would.

Press and hold 'c' to stop recording.

Note: since jka_recorder.py automatically brings the game to the foreground you
won't be able to alt+tab out effectively until you stop recording.

There should be a data.csv in the data folder with the header:

```
image_name,w,a,s,d,f,e,r,space,ctrl,mouse_left,mouse_middle,mouse_right,mouse_deltaX,mouse_deltaY
```

and data/train should contain screenshots of the game screen.

# Training the Neural Network Model

Simply do:

```
python jka_model.py
```

And a model will be trained on all the recorded data. It will appear as

```
jka_model.pt
```

# Running the Neural Network Model

Run the game at 800x600 and join a multiplayer server. Then,

```
python jka_controller.py
```

to run the Jedi AI. Press ESC to stop.

# CDA0
Initial prototype AI agent for cooperative driving automation (CDA)
This project is the first in what is expected to be a series of projects of increasing complexity to train AI agents that help solve some problems in Cooperative Driving Automation (CDA).
We use the [Ray/RLlib platform](https://docs.ray.io/en/latest/rllib/index.html) for training agents using Reinforcement Learning (RL).

## Repo Summary
This particular repo contains code that represents a simple highway driving scenario and trains a single RL agent to safely drive in that (simulated) environment.
Its primary purpose was to gain experience with the Ray training platform and with RL techniques and issues related to automated driving.
The environment model is built to the Gymnasium standard, which is supported by Ray.
The driving problem is formulated as an episodic problem (there is a definite end), where the agent drives from its starting point on the simulated track until it either reaches the end of the track (success) or drives off-road, crashes into another vehicle or comes to a stop in the roadway (all failures).
The RL agent takes in observations of the world - for this project, they are idealized and readily available - and its neural network (NN) maps those observations into an appropriate set of actions for a given small time step.
The environment model then moves the world model forward by one time step, by calculating the new state of the world and the agent using the programmed dynamics.
It takes in the agent's current actions, determines the new state of the world, and returns a new set of observations that the agent senses.

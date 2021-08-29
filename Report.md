[//]: # (Image References)

[image1]: ./img/m1.png "Model 1"
[image2]: ./img/m2.png "Model 2"
[image3]: ./img/m3.png "Model 3"
[image4]: ./img/m4.png "Model 4"
[image5]: ./img/m5.png "Model 5"
[image6]: ./img/dqn.png "DQN algorithm"
# Project 1: Navigation

## Introduction


For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  


### Project Structure

The code is written in PyTorch and Python3, executed in Jupyter Notebook
- Navigation.ipynb	: Training and evaluation of different agents
- dqnagent.py	: Agent and ReplayBuffer Class
- model.py	: Build QNetwork and train function
- model1.pth : Saved Model Weights


### Environment

In this task the agent should maximize the reward by collecting 
yellow bananas and avoiding blue bananas. The environment is implementedwith Unity. 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
THe agent can take four actions:
- Walk forward 
- Walk backward
- turn left
- turn right

In order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Learning Algorithm

Q learning is a value-based Reinforcement Learning algorithm to learn the value of an action in a particular state.
The goal of Q learning is to find an optimal policy which maximizes the reward over all successive steps. 

<img src="https://latex.codecogs.com/gif.latex?\mathit{Q}^{*}(s,&space;a)&space;=&space;\underset{\pi}{\mathrm{max}}\&space;\mathbb{E}&space;\left&space;[&space;r_{t}&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma&space;^{2}&space;r_{t&plus;2}&plus;...|s_{t}=s,&space;a_{t}=a,&space;\pi&space;\right&space;]" title="\mathit{Q}^{*}(s, a) = \underset{\pi}{\mathrm{max}}\ \mathbb{E} \left [ r_{t} + \gamma r_{t+1} + \gamma ^{2} r_{t+2}+...|s_{t}=s, a_{t}=a, \pi \right ]" />

To approximate the optimal action-value function a neural network was used. The usage of neural networks can lead that the reinforcement learning is unstable or
diverge. To overcome this issue, a nature inspired mechanism, namely experience replay was used in 'Human-level control through deep reinforcement
learning'. Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). The agent then samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive Q-learning algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agent has multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment. 

The second concept to address these issues was to update the target action-value function periodically to reduce correlations.

To find the optimal action-value function during the training following loss function is used:

<a href="https://www.codecogs.com/eqnedit.php?latex=L_i{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'};,\theta^-_i)&space;-&space;Q(s,a;\theta_i)\right&space;)^{2}&space;\right&space;]&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_i{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'};,\theta^-_i)&space;-&space;Q(s,a;\theta_i)\right&space;)^{2}&space;\right&space;]&space;}" title="L_i{(\theta) = \mathrm{E_{\mathit{s,a,r,s^{'} D}}}\left [ \left (r + \gamma \underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'};,\theta^-_i) - Q(s,a;\theta_i)\right )^{2} \right ] }" /></a>

- <a href="https://www.codecogs.com/eqnedit.php?latex=\theta^-_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta^-_i" title="\theta^-_i" /></a> parameters of the target network. updated every C steps
- <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_i" title="\theta_i" /></a> network parameters at iteration i
- <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a> discount factor

![alt text][image6]


Instead of applying a hard update of weights of the target network as proposed in the paper, a soft update is used:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\theta^-&space;\leftarrow&space;(1-\tau)\theta^-&space;&plus;&space;\tau&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\theta^-&space;\leftarrow&space;(1-\tau)\theta^-&space;&plus;&space;\tau&space;\theta" title="\theta^- \leftarrow (1-\tau)\theta^- + \tau \theta" /></a>

The weights of these target networks are then updated by having them slowly track the learned networks: θ⁻←τθ+(1−τ)θ⁻ with τ<<1. This means that the target values are constrained to change slowly, greatly improving the stability of learning[2].

Epsilon Greedy Algorithm

The agent implements the Epsilon greedy algorithm to handle the exploration vs exploitation dilemma. 
The challenge with the Q-function is to choosing the action to take while the agent is still learning the policy. 
One challenge with the Q-function above is choosing which action to take while the agent is still learning the optimal policy. Should the agent choose an action based on the Q-values observed thus far? Or, should the agent try a new action in hopes of earning a higher reward? 

To address this, in the agent an 𝛆-greedy algorithm is implemented. This algorithm allows the agent to systematically manage the exploration vs. exploitation trade-off. 
The agent pick the action in follow manner:
* picking a random action with probability  𝛜 -> exploration
* choose action based on the policy with probability (1- 𝛜) -> exploit its knowledge of the environment


Furthermore, the value of epsilon is purposely decayed over time, so that the agent favors exploration during its initial interactions with the environment, but increasingly favors exploitation as it gains more experience. 

### Implementation

In the file dqnagent.py the agent is implemented. It is based on the Agent from course.
The model.py implements the QNetwork class which is utilized by the agent. This class
is modified to support any number of hidden layers.

### Hyperparameters

In this section the hyperparameters of the agent and training are described. 

Agent hyperparameters

* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 64         # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR = 5e-4               # learning rate 
* UPDATE_EVERY = 4        # how often to update the network

Training Hyperparameters

* n_episodes = 2000   # maximum number of iterations
* max_t = 1000        # maximum number of time steps in one episode
* eps_start = 1.0     # epsilon value for e-greedy heuristic (exploration)
* eps_end = 0.025     # final value for epsilon
* eps_decay= 0.9975   # decay factor for epsilon

### Models

There are 5 models with different number of hidden layers and Ɛ tested.
Each architecture uses fully connected layers as hidden layers. After each hidden
layer the ReLu function is applied. The input to the neural network is the state vector 

| Model | Architecture         		|     Ɛ	        					| 
|:---------------------:|:---------------------:|:---------------------------------------------:| 
| 1| 37->64->32->4 | 0.9975 |
| 2| 37->128->64->16->4| 0.9975 |
| 3| 37->256->128->32->4 | 0.9975 |
| 4| 37->128->64->16->4 | 0.95 |
| 5| 37->64->32->4 | 0.95 |

## Results


For each Agent, the score over the episodes is recorded. Each Agent can solve the task in less than 1000 episodes. 
The number of episodes for the models are not far apart for the same Ɛ. From the first 3 models model 1 and 2 with
a Ɛ of 0.95 trained. By reducing Ɛ from 0.9975 to 0.95 the two agents learn a lot faster and can solve the task in 315 or
233 episodes. By reducing Ɛ the agents explore the environment less. 

The following table shows the episodes when an agent has solved the task:

| Model | Episodes         		|    
|:---------------------:|:---------------------:|
| 1| 794 | 
| 2| 743 | 
| 3| 761 | 
| 4| 315  | 
| 5| 233 | 


Deep Reinforcement Learning nature is very volatile, as the results show, the ups                         
and downs are very present during the whole training. 



### Model 1

![alt text][image1]

### Model 2

![alt text][image2]

### Model 3

![alt text][image3]

### Model 4

![alt text][image4]

### Model 5

![alt text][image5]

## Ideas for Future Work

To improve the stability of the reinforcement learning algorithm methods                          
like Experience Replay, target update in fixed intervals, soft update of weights was used. 

There are follow ideas:
* more training runs with different seeds and epsilon
* Implement the remaining improvements from rainbow, namely Dueling DQN, Noisy DQN, A3C, Distributional DQN
* Test the agent in the unity environment
* Replace conventional exploration heuristics with Noisy DQN — This approach is explained here in this research paper. The key takeaway is that parametric noise is added  to the weights to induce stochasticity to the agent's policy, yielding more efficient exploration[3].


## References

* [1] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
* [2] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra. Continuous control with deep reinforcement learning. https://arxiv.org/abs/1509.02971
* [3] Meire Fortunato and Mohammad Gheshlaghi Azar and Bilal Piot and Jacob Menick and Ian Osband and Alex Graves and Vlad Mnih and Remi Munos and Demis Hassabis and Olivier Pietquin and Charles Blundell and Shane Legg. Noisy Networks for Exploration. https://arxiv.org/abs/1706.10295

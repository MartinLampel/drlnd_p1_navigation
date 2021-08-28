[//]: # (Image References)

[image1]: ./img/m_64_32_085.png "Model 1"
[image2]: ./img/m_126_64_32.png "Model 2"
[image3]: ./img/m_256_32.png "Model 3"
[image4]: ./img/m64_32_09.png "Model 4"
[image5]: ./img/m64_32_095.png "Model 5"
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
learning'. This approach randomizes over data and remove correlations. The second concept to address these issues was to update the target action-value function periodically to reduce correlations.

To find the optimal action-value function during the training following loss function is used:

<a href="https://www.codecogs.com/eqnedit.php?latex=L_i{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'};,\theta^-_i)&space;-&space;Q(s,a;\theta_i)\right&space;)^{2}&space;\right&space;]&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_i{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'};,\theta^-_i)&space;-&space;Q(s,a;\theta_i)\right&space;)^{2}&space;\right&space;]&space;}" title="L_i{(\theta) = \mathrm{E_{\mathit{s,a,r,s^{'} D}}}\left [ \left (r + \gamma \underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'};,\theta^-_i) - Q(s,a;\theta_i)\right )^{2} \right ] }" /></a>

- <a href="https://www.codecogs.com/eqnedit.php?latex=\theta^-_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta^-_i" title="\theta^-_i" /></a> parameters of the target network. updated every C steps
- <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_i" title="\theta_i" /></a> network parameters at iteration i
- <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a> discount factor

![alt text][image6]

### Implementation

In the file dqnagent.py the agent is implemented. It is based on the Agent from course.
The model.py implements the QNetwork class which is utilized by the agent. This class
is modified to support any number of hidden layers.



### Models

There are 5 models with different number of hidden layers and Ɛ tested.
Each architecture uses fully connected layers as hidden layers. After each hidden
layer the ReLu function is applied.

| Model | Architecture         		|     Ɛ	        					| 
|:---------------------:|:---------------------:|:---------------------------------------------:| 
| 1| 37->64->32->4 | 0.85 |
| 2| 37->128->64->32->4| 0.85 |
| 3| 37->256->32->4 | 0.85 |
| 4| 37->64->32->4 | 0.95 |
| 5| 37->64->32->4 | 0.9 |

## Results

### Model 1

| Episode         		|     Average Score	        					|
|:---------------------:|:---------------------:|
| 100	| 1.83 |
| 200	| 7.00 |
| 300	| 12.13 |
| 329	| 13.05 |


![alt text][image1]

### Model 2


| Episode         		|     Average Score	        					|
|:---------------------:|:---------------------:|
| 100	| 0.87|
| 200	| 6.63|
| 300	| 9.16 |
| 384		| 13.01 |

![alt text][image2]

### Model 3


| Episode         		|     Average Score	        					|
|:---------------------:|:---------------------:|
| 100	| 2.23 |
| 200	| 6.67 |
| 300	| 10.9 |
| 382		| 13.09|

![alt text][image3]

### Model 4


| Episode         		|     Average Score	        					|
|:---------------------:|:---------------------:|
| 100	| 2.32 |
| 200	| 9.02 |
| 300	| 12.69 |
| 305		| 13.02|

![alt text][image4]

### Model 5

| Episode         		|     Average Score	        					|
|:---------------------:|:---------------------:|
| 100	| 2.38 |
| 200	| 8.35 |
| 300	| 10.86 |
| 355		| 13.08|

![alt text][image5]


## References

* [1] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236

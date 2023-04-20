## Environment Descriptions

### Discrete Environments

- **Blackjack-v1**: A simplified version of the classic card game Blackjack.
- **FrozenLake-v1**: Navigate through a grid of slippery ice and holes to reach the goal.
- **CliffWalking-v0**: Navigate along a cliff without falling off to reach the goal.
- **Taxi-v3**: Pick up and drop off passengers at their destinations in a grid-based environment.
- **CartPole-v1**: Balance a pole on a cart by applying forces to the cart.
- **Acrobot-v1**: Swing a two-link robot up to a given height.
- **MountainCar-v0**: Drive a car up a hill by gaining momentum from going back and forth.
- **LunarLander-v2**: Safely land a lunar module on the surface of the moon.
- **BipedalWalker-v3**: Control a 2D bipedal robot to walk through challenging terrain.
- **BipedalWalkerHardcore-v3**: Control a 2D bipedal robot to walk through even more challenging terrain.
- **Reacher-v4**: Move a robotic arm to reach a target.
- **Pusher-v4**: Control a robot arm to push objects to a target location.
- **Swimmer-v4**: Control a snake-like robot to swim through water.
- **Ant-v4**: Control a 4-legged ant-like robot to walk.
- **Hopper-v4**: Control a one-legged robot to hop and move forward.
- **Walker2d-v4**: Control a 2D bipedal robot to walk.
- **Humanoid-v4**: Control a 3D humanoid robot to walk and run.
- **HumanoidStandup-v4**: Control a 3D humanoid robot to stand up from a resting position.

### Continuous Environments

- **MountainCarContinuous-v0**: A continuous version of MountainCar, where the car's acceleration is controlled directly.
- **Pendulum-v1**: Control the torque on a pendulum to keep it upright.
- **LunarLanderContinuous-v2**: A continuous version of LunarLander, where the lander's thrusters are controlled directly.
- **CarRacing-v0**: Drive a car around a randomly generated racetrack.
- **InvertedPendulum-v4**: Balance an inverted pendulum on a cart by applying forces to the cart.
- **InvertedDoublePendulum-v4**: Balance a double inverted pendulum on a cart by applying forces to the cart.

### Custom Environments

- **GridWorld**: A custom environment where an agent navigates a grid-based world to reach a goal.

## Agent Descriptions

- **RandomAgent**: An agent that selects actions randomly without learning from the environment.
- **QLAgent**: A Q-Learning agent that learns to select actions based on estimates of state-action values.
- **DQNAgent**: A Deep Q-Network agent that uses a neural network to approximate state-action values.
- **PPOAgent**: A Proximal Policy Optimization agent that optimizes a policy using a surrogate objective function.
- **PGAgent**: A Policy Gradient agent that learns a parameterized policy by estimating gradients of expected rewards.
- **DDPGAgent**: A Deep Deterministic Policy Gradient agent that combines deep learning and policy gradient methods for continuous control.
- **A2CAgent**: An Advantage Actor-Critic agent that uses separate neural networks for policy and value functions.
- **SACAgent**: A Soft Actor-Critic agent that learns a stochastic policy for continuous control tasks using a maximum entropy framework
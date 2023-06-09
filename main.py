import os
import sys
import gymnasium as gym

def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample() 
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    return

if __name__ == "__main__":
    main()
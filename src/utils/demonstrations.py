import gymnasium as gym
import subprocess
import inspect
import sys

def selector():
    # Lets the user select from available demos, and runs the selected demo.
    # Make sure the selector can handle other demo_xxx() functions

    # Get a list of all the demo functions in the current module
    members = inspect.getmembers(sys.modules[__name__])
    demo_funcs = [func for name, func in members if callable(func) and name.startswith("demo_")]
    demo_names = [func.__name__[5:].capitalize() for func in demo_funcs]

    print("Available demos:")
    for i, name in enumerate(demo_names, start=1):
        print(f"{i}. {name}")

    demo_choice = input("Enter the demo number or name you want to run: ")
    selected_func = None

    # Try to parse input as integer first
    try:
        demo_choice_int = int(demo_choice)
        if demo_choice_int in range(1, len(demo_funcs)+1):
            selected_func = demo_funcs[demo_choice_int-1]
    except ValueError:
        pass

    # If input was not an integer or not a valid demo number, try to match input as demo name
    if not selected_func:
        for func, name in zip(demo_funcs, demo_names):
            if name.lower() == demo_choice.lower():
                selected_func = func
                break

    if selected_func:
        selected_func()
    else:
        print("Invalid demo number or name. Please try again.")
        selector()
        
def demo_tests():
    
    # Demonstrates the functionality and commands assosiated with the tests
    # Defaults to: pytest test_agent_random.py
    
    subprocess.run(["pytest", "test_agent_random.py"])
    
def demo_gym():
        
    env = gym.make("LunarLander-v2", render_mode='human')
    env.reset()
    total_reward = 0

    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, info = env.reset(seed=123, options={})
        done = False

        while not done:
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        env.close()
        
        total_reward += reward

        if done:
            print(f"Episode finished after {_ + 1} timesteps")
            print(f"Total reward: {total_reward}")
            break

    env.close()
    
if __name__ == "__main__":
    selector()
import os
import sys
import gymnasium as gym

# add src directory to path

from src.utils import demonstrations

def main():
    print("Starting")
    demonstrations.selector()
    return

if __name__ == "__main__":
    main()
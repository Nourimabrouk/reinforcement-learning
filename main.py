import os
import sys
import gymnasium as gym

# add src directory to path
sys.path.append(os.path.abspath("src"))

from src.utils import demonstrations

def main():
    demonstrations.selector()
    return

if __name__ == "__main__":
    main()

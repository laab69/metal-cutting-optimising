"""
Training Executable Runner (`train_policy.py`)

Run this module to start REINFORCE policy gradient training on the sheet nesting environment.
"""

from train.trainer import train_reinforce

if __name__ == "__main__":
    train_reinforce()

#!/usr/bin/env python3

import sys
from envs import ReduceGraphEnv

def main():
    env = ReduceGraphEnv(max_graph_size=20)
    env.reset()
    for _ in range(1000):
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        print('.', end='')
        sys.stdout.flush()
    env.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()

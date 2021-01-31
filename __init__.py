from gym.envs.registration import register

register(
    id='reduce-graph-v1',
    entry_point='__main__.envs:ReduceGraphEnv',
)

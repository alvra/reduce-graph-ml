from typing import Optional, Iterable, Tuple
from dataclasses import dataclass
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np


def get_graph_index(env, state, start_block: int, end_block: int) -> int:
    assert 0 <= start_block < env.max_graph_size
    assert 0 <= end_block < env.max_graph_size
    return start_block + env.max_graph_size * end_block

def get_graph_link(env, state, start_block: int, end_block: int) -> bool:
    index = get_graph_index(env, state, start_block, end_block)
    return bool(state['graph'][index])

def set_graph_link(env, state, start_block: int, end_block: int, link: bool):
    index = get_graph_index(env, state, start_block, end_block)
    state['graph'][index] = int(link)

def get_links_from(env, state, start_block: int) -> Iterable[int]:
    for end_block in range(env.max_graph_size):
        if get_graph_link(env, state, start_block, end_block):
            yield end_block

def get_links_into(env, state, end_block: int) -> Iterable[int]:
    for start_block in range(env.max_graph_size):
        if get_graph_link(env, state, start_block, end_block):
            yield start_block

def update_links_into(env, state, old_end_block: int, new_end_block: int):
    for previous_block in get_links_into(env, state, old_end_block):
        set_graph_link(env, state, previous_block, old_end_block, False)
        set_graph_link(env, state, previous_block, new_end_block, True)

def update_links_from(env, state, old_start_block: int, new_start_block: int):
    for next_block in get_links_from(env, state, old_start_block):
        set_graph_link(env, state, old_start_block, next_block, False)
        set_graph_link(env, state, new_start_block, next_block, True)

def block_is_linked(env, state, block: int) -> bool:
    "Return True if a block has any links, None if it is unlinked to anything."
    return any(
        get_graph_link(env, state, block, other_block)
        or
        get_graph_link(env, state, other_block, block)
        for other_block in range(env.max_graph_size))

def is_block_empty(env, state, block: int) -> bool:
    return bool(state['is_empty'][block])

def set_is_block_empty(env, state, block: int, value: bool) -> bool:
    state['is_empty'][block] = int(value)

def count_links(env, state) -> int:
    return np.sum(state['graph'])

def count_backjumps(env, state) -> int:
    return sum(
        np.sum(state['graph'][block + 1:])
        for block in range(1, env.max_graph_size))

def all_backjumps(env, state) -> Iterable[Tuple[int, int]]:
    for end_block in range(env.max_graph_size):
        for start_block in get_links_into(env, state, end_block):
            if end_block < start_block:
                yield start_block, end_block

def count_irriducibles(env, state) -> int:
    result = 0
    for start_block, end_block in all_backjumps(env, state):
        loop_blocks = range(end_block, start_block + 1)
        for loop_block in loop_blocks:
            for outside_block in get_links_into(env, state, loop_block):
                if outside_block not in loop_blocks:
                    result += 1
    return result


class Action:
    def apply(self, env, state):
        raise NotImplementedError


class InvalidAction(Exception):
    pass


@dataclass
class SwapBlocks(Action):
    block_1: int
    block_2: int

    def apply(self, env, state):
        if self.block_1 == self.block_2:
            # don't allow the no-op
            raise InvalidAction
        # TODO optimize this using numpy array manipulations
        # (the graph then needs to become a proper numpy matrix)
        is_empty_1 = is_block_empty(env, state, self.block_1)
        is_empty_2 = is_block_empty(env, state, self.block_2)
        set_is_block_empty(env, state, self.block_1, is_empty_2)
        set_is_block_empty(env, state, self.block_2, is_empty_1)
        block_1_links_in = [
            get_graph_link(env, state, block, self.block_1)
            for block in range(env.max_graph_size)]
        block_2_links_in = [
            get_graph_link(env, state, block, self.block_2)
            for block in range(env.max_graph_size)]
        for block, value in enumerate(block_1_links_in):
            set_graph_link(env, state, block, self.block_2, value)
        for block, value in enumerate(block_2_links_in):
            set_graph_link(env, state, block, self.block_1, value)
        # Note we have to swap the dimensions in order
        # else we mess up the intersecting element.
        block_1_links_out = [
            get_graph_link(env, state, self.block_1, block)
            for block in range(env.max_graph_size)]
        block_2_links_out = [
            get_graph_link(env, state, self.block_2, block)
            for block in range(env.max_graph_size)]
        for block, value in enumerate(block_1_links_out):
            set_graph_link(env, state, self.block_2, block, value)
        for block, value in enumerate(block_2_links_out):
            set_graph_link(env, state, self.block_1, block, value)


@dataclass
class IncludeEmptyBlockBefore(Action):
    block: int
    empty_block: int

    def apply(self, env, state):
        if is_block_empty(env, state, self.block):
            raise InvalidAction
        if not is_block_empty(env, state, self.empty_block):
            raise InvalidAction
        update_links_into(env, state, self.block, self.empty_block)
        set_graph_link(env, state, self.empty_block, self.block, True)


@dataclass
class IncludeEmptyBlockAfter(Action):
    block: int
    empty_block: int

    def apply(self, env, state):
        if is_block_empty(env, state, self.block):
            raise InvalidAction
        if not is_block_empty(env, state, self.empty_block):
            raise InvalidAction
        update_links_from(env, state, self.block, self.empty_block)
        set_graph_link(env, state, self.block, self.empty_block, True)


# TODO copy block action


action_types = [
    SwapBlocks,
    IncludeEmptyBlockBefore,
    IncludeEmptyBlockAfter,
]


class ReduceGraphEnv(gym.Env):
    reward_range = (-float('inf'), float('inf'))  # TODO

    rewards = dict(
        invalid_action=-.10,
        link_added=-1,
        backjump_added=-2,
        irriducible_added=-10,
    )
    graph_probabilities = dict(
        empty_ratio=.333,
        second_link=.333,
        self_link=.05,
        backward_links=.2,
    )

    viewer_size = 200

    def __init__(self, max_graph_size):
        assert (self.graph_probabilities['backward_links']
                + self.graph_probabilities['self_link']) < 1

        self.max_graph_size = max_graph_size

        self.action_space = spaces.Dict(dict(
            action_type=spaces.Discrete(len(action_types)),
            block_1=spaces.Discrete(max_graph_size),
            block_2=spaces.Discrete(max_graph_size),
        ))
        self.observation_space = spaces.Dict(dict(
            # the graph itself
            graph=spaces.MultiBinary(max_graph_size * max_graph_size),
            # which blocks are empty
            is_empty=spaces.MultiBinary(max_graph_size),
        ))

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate_link(self, state, block, non_empty_blocks):
        link_type = self.np_random.random()
        if link_type < self.graph_probabilities['backward_links']:
            self.generate_backward_link(state, block)
            return
        link_type -= self.graph_probabilities['backward_links']
        if link_type < self.graph_probabilities['self_link']:
            self.generate_self_link(state, block)
        else:
            self.generate_forward_link(state, block, non_empty_blocks)

    def generate_forward_link(self, state, start_block, non_empty_blocks):
        assert 0 <= start_block <= non_empty_blocks, start_block
        if non_empty_blocks - 1 == start_block:
            # Don't generate links from the end block.
            return
        assert non_empty_blocks <= self.max_graph_size
        end_block = self.np_random.randint(start_block + 1, non_empty_blocks)
        set_graph_link(self, state, start_block, end_block, True)

    def generate_backward_link(self, state, start_block):
        assert 0 <= self.max_graph_size, start_block
        if start_block <= 1:
            # Don't generate backward links from or into the start block.
            return
        end_block = self.np_random.randint(1, start_block)
        set_graph_link(self, state, start_block, end_block, True)

    def generate_self_link(self, state, block):
        set_graph_link(self, state, block, block, True)

    def reset(self):
        empty_blocks = self.np_random.binomial(
            self.max_graph_size - 1,
            self.graph_probabilities['empty_ratio'])
        assert 0 <= empty_blocks < self.max_graph_size, empty_blocks
        non_empty_blocks = self.max_graph_size - empty_blocks
        is_empty = np.array(
            [0] * non_empty_blocks + [1] * empty_blocks,
            dtype=self.observation_space.spaces['is_empty'].dtype)
        graph = np.zeros(self.max_graph_size * self.max_graph_size)
        state = dict(
            graph=graph,
            is_empty=is_empty,
        )
        # Note we don't link from the end block.
        for block in range(non_empty_blocks - 1):
            self.generate_link(state, block, non_empty_blocks)
            if self.np_random.random() < self.graph_probabilities['second_link']:
                # Note this may create the same link as the first one,
                # but we don't care as it's very rare.
                self.generate_link(state, block, non_empty_blocks)
        self.state = state
        return np.array(self.state)

    def step(self, action):
        link_count_pre = count_links(self, self.state)
        backjump_count_pre = count_backjumps(self, self.state)
        irriducible_count_pre = count_irriducibles(self, self.state)
        action_type = action_types[action['action_type']]
        action_obj = action_type(action['block_1'], action['block_2'])
        try:
            action_obj.apply(self, self.state)
        except InvalidAction:
            reward = self.rewards['invalid_action']
        else:
            link_count_post = count_links(self, self.state)
            links_created = link_count_post - link_count_pre
            backjump_count_post = count_backjumps(self, self.state)
            backjumps_created = backjump_count_post - backjump_count_pre
            irriducible_count_post = count_irriducibles(self, self.state)
            irriducibles_added = irriducible_count_post - irriducible_count_pre
            reward = sum([
                self.rewards['link_added'] * links_created,
                self.rewards['backjump_added'] * backjumps_created,
                self.rewards['irriducible_added'] * irriducibles_added,
            ])
        done = False  # TODO
        info = {}
        return np.array(self.state), reward, done, info

    def render_rect(self, viewer, x, y, w, h, color):
        # invert y so it runs downward
        y = viewer.height - y
        h = - h
        rect = rendering.FilledPolygon([
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h),
        ])
        rect.set_color(*color)
        viewer.add_geom(rect)

    def render_graph(self, viewer, state):
        dx = viewer.width / (self.max_graph_size + 1)
        dy = viewer.height / self.max_graph_size
        self.render_rect(viewer, 0, 0, viewer.width, viewer.height, (0, 0, 0))
        for start_block in range(self.max_graph_size):
            for end_block in range(self.max_graph_size):
                if get_graph_link(self, state, start_block, end_block):
                    color = (1, 1, 1)
                    if start_block == end_block:
                        color = (1, 0, 0)
                    self.render_rect(
                        viewer,
                        end_block * dx, start_block * dy,
                        dx, dy,
                        color)
                else:
                    if start_block == end_block:
                        self.render_rect(
                            viewer,
                            end_block * dx, start_block * dy,
                            dx, dy,
                            (.1, .1, .1))
        for block in range(self.max_graph_size):
            if is_block_empty(self, state, block):
                self.render_rect(
                    viewer,
                    self.max_graph_size * dx, block * dy,
                    dx, dy,
                    (0, 0, 1))

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(
                int(self.viewer_size * (1.1 + 1 / self.max_graph_size)),
                self.viewer_size)
        if self.state is None:
            return None
        self.render_graph(self.viewer, self.state)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

import numpy as np

def parse_action(raw_action) -> dict:
    """
    We assume that an action is of the shape (20, 20, 3) but flattened
    Now we pull it apart into the real action
        {node_1: [0-19]
         node_2: [0-19]
         action_type: [0-2]}
    """
    # node_1, remainder = divmod(raw_action, (20*20))
    remainder, action_type = divmod(raw_action, 3)
    node_1, node_2 = divmod(remainder, 20)
    return (node_1, node_2, action_type)


def inverse(node_1, node_2, action_type):
    return ((node_1 * 20) + node_2) * 3 + action_type


def test(n, x, y, z):
    s = f'{n}: {parse_action(np.arange(1200).reshape((20, 20, 3))[x, y, z])} == {(x, y, z)}'
    assert parse_action(np.arange(1200).reshape((20, 20, 3))[x, y, z]) == (x, y, z), s

test(1, 0, 0, 1)
test(2, 1, 0, 0)
test(3, 0, 1, 0)


assert parse_action(np.arange(1200).reshape((20, 20, 3))[19, 19, 1]) == (19, 19, 1)
assert parse_action(np.arange(1200).reshape((20, 20, 3))[19, 15, 1]) == (19, 15, 1)
assert parse_action(np.arange(1200).reshape((20, 20, 3))[1, 3, 2]) == (1, 3, 2)

for x in range(20):
    for y in range(20):
        for z in range(3):
            test(f'{x}:{y}:{z}', x, y, z)
            assert parse_action(inverse(x, y, z)) == (x, y, z)

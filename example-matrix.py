#!/usr/bin/env python3

# Block indices
#
# blocks are given in the order in which they appear in the source
blocks = {
       0: "start",
       1: "0x0000(01)",
       2: "0x0007(05)",
       3: "0x000A(00)",
       4: "0x000A(02)",
       5: "0x0027(01)",
       6: "0x0032(04)",
       7: "0x0032(14)",
       8: "0x0039(01)",
       9: "0x003E(00)",
      10: "0x003E(13)",
      11: "0x0048(02)",
      12: "0x004A(08)",
      13: "0x004B(02)",
      14: "0x004B(06)",
      15: "0x004F(00)",
      16: "0x004F(09)",
      17: "0x0060(03)",
      18: "0x0062(06)",
      19: "0x0064(09)",
      20: "0x0068(11)",
      21: "0x0068(15)",
      22: "0x006B(06)",
      23: "0x006B(13)",
      24: "0x006E(08)",
      25: "0x006E(14)",
      26: "0x0078(00)",
      27: "0x0078(03)",
      28: "0x007E(10)",
      29: "0x007E(14)",
      30: "0x0080(06)",
      31: "0x0086(00)",
      32: "0x008C(01)",
      33: "0x008C(05)",
      34: "0x008F(09)",
      35: "0x008F(15)",
      36: "0x00E6(10)",
      37: "0x00E9(06)",
      38: "0x00E9(09)",
      39: "0x00E9(13)",
      40: "0x00EF(08)",
      41: "0x00EF(14)",
      42: "0x00F4(03)",
      43: "0x00F7(04)",
      44: "0x00F7(12)",
      45: "0x00FA(01)",
      46: "0x00FA(08)",
      47: "0x00FA(09)",
      48: "0x0100(03)",
      49: "end",
}

# Sparse matrix of jumps
#
# the first index is the start of the jump
# the second index is the end of the jump
# the final object is the jump id
# so: sparse_matrix[start_block_index][end_block_index] = jump_id
sparse_matrix = {
     0: {
         1: "start -> 0x0000(01) unconditional",
    },
     1: {
         2: "0x0000(01) -> 0x0007(05) unconditional",
    },
     2: {
         4: "0x0007(05) -> 0x000A(02) unconditional",
         3: "0x0007(05) -> 0x000A(00) conditional",
    },
     3: {
         2: "0x000A(00) -> 0x0007(05) unconditional",
    },
     4: {
         5: "0x000A(02) -> 0x0027(01) unconditional",
    },
     5: {
         6: "0x0027(01) -> 0x0032(04) unconditional",
         7: "0x0027(01) -> 0x0032(14) conditional",
    },
     6: {
         8: "0x0032(04) -> 0x0039(01) unconditional",
    },
     7: {
         5: "0x0032(14) -> 0x0027(01) unconditional",
    },
     8: {
        10: "0x0039(01) -> 0x003E(13) conditional",
         9: "0x0039(01) -> 0x003E(00) unconditional",
    },
     9: {
        11: "0x003E(00) -> 0x0048(02) unconditional",
    },
    10: {
         8: "0x003E(13) -> 0x0039(01) unconditional",
    },
    11: {
        12: "0x0048(02) -> 0x004A(08) unconditional",
    },
    12: {
        14: "0x004A(08) -> 0x004B(06) unconditional",
        13: "0x004A(08) -> 0x004B(02) conditional",
    },
    13: {
        17: "0x004B(02) -> 0x0060(03) unconditional",
    },
    14: {
        15: "0x004B(06) -> 0x004F(00) unconditional",
        16: "0x004B(06) -> 0x004F(09) conditional",
    },
    15: {
        11: "0x004F(00) -> 0x0048(02) unconditional",
    },
    16: {
        12: "0x004F(09) -> 0x004A(08) unconditional",
    },
    17: {
        18: "0x0060(03) -> 0x0062(06) unconditional",
    },
    18: {
        19: "0x0062(06) -> 0x0064(09) unconditional",
    },
    19: {
        21: "0x0064(09) -> 0x0068(15) conditional",
        20: "0x0064(09) -> 0x0068(11) unconditional",
    },
    20: {
        23: "0x0068(11) -> 0x006B(13) conditional",
        22: "0x0068(11) -> 0x006B(06) unconditional",
    },
    21: {
        19: "0x0068(15) -> 0x0064(09) unconditional",
    },
    22: {
        24: "0x006B(06) -> 0x006E(08) unconditional",
        25: "0x006B(06) -> 0x006E(14) conditional",
    },
    23: {
        19: "0x006B(13) -> 0x0064(09) unconditional",
    },
    24: {
        26: "0x006E(08) -> 0x0078(00) unconditional",
        27: "0x006E(08) -> 0x0078(03) conditional",
    },
    25: {
        18: "0x006E(14) -> 0x0062(06) unconditional",
    },
    26: {
        29: "0x0078(00) -> 0x007E(14) conditional",
        28: "0x0078(00) -> 0x007E(10) unconditional",
    },
    27: {
        30: "0x0078(03) -> 0x0080(06) unconditional",
    },
    28: {
        30: "0x007E(10) -> 0x0080(06) unconditional",
    },
    29: {
        31: "0x007E(14) -> 0x0086(00) unconditional",
    },
    30: {
        31: "0x0080(06) -> 0x0086(00) unconditional",
    },
    31: {
        32: "0x0086(00) -> 0x008C(01) unconditional",
        33: "0x0086(00) -> 0x008C(05) conditional",
    },
    32: {
        35: "0x008C(01) -> 0x008F(15) conditional",
        34: "0x008C(01) -> 0x008F(09) unconditional",
    },
    33: {
        17: "0x008C(05) -> 0x0060(03) unconditional",
    },
    34: {
        17: "0x008F(09) -> 0x0060(03) unconditional",
    },
    35: {
        36: "0x008F(15) -> 0x00E6(10) unconditional",
    },
    36: {
        38: "0x00E6(10) -> 0x00E9(09) unconditional",
    },
    37: {
        38: "0x00E9(06) -> 0x00E9(09) unconditional",
    },
    38: {
        39: "0x00E9(09) -> 0x00E9(13) unconditional",
        37: "0x00E9(09) -> 0x00E9(06) conditional",
    },
    39: {
        40: "0x00E9(13) -> 0x00EF(08) conditional",
        41: "0x00E9(13) -> 0x00EF(14) unconditional",
    },
    40: {
        36: "0x00EF(08) -> 0x00E6(10) unconditional",
    },
    41: {
        42: "0x00EF(14) -> 0x00F4(03) unconditional",
    },
    42: {
        43: "0x00F4(03) -> 0x00F7(04) conditional",
        44: "0x00F4(03) -> 0x00F7(12) unconditional",
    },
    43: {
        42: "0x00F7(04) -> 0x00F4(03) unconditional",
    },
    44: {
        46: "0x00F7(12) -> 0x00FA(08) unconditional",
    },
    45: {
        46: "0x00FA(01) -> 0x00FA(08) unconditional",
    },
    46: {
        47: "0x00FA(08) -> 0x00FA(09) unconditional",
        45: "0x00FA(08) -> 0x00FA(01) conditional",
    },
    47: {
        48: "0x00FA(09) -> 0x0100(03) unconditional",
    },
    48: {
        48: "0x0100(03) -> 0x0100(03) unconditional",
    },
    49: {
    },
}

# Complete matrix of jumps
#
# the matrix contains a True if there exist a link between the blocks
# so: matrix[start_block_index][end_block_index] = bool
matrix_size = len(blocks)
matrix = [
    [
        end_id in sparse_matrix.get(start_id, set())
        for end_id in range(matrix_size)
    ]
    for start_id in range(matrix_size)
]

def print_matrix(matrix,
                 size=matrix_size,
                 link_char='x',
                 no_link_char='.',
                 diagonal_style='\033[91m{}\033[0m',
                 ):
   """
   Print a jump matrix as a matrix of text
   where each line (row) shows the jumps from a block
   and each column the jumps into a block.
   """
   for start_id in range(size):
       for end_id in range(size):
           try:
               is_linked = matrix[start_id][end_id]
           except IndexError:
               is_linked = False
           char = link_char if is_linked else no_link_char
           if start_id == end_id:
               char = diagonal_style.format(char)
           print(char, end='')
       print()

if __name__ == '__main__':
    print_matrix(matrix)

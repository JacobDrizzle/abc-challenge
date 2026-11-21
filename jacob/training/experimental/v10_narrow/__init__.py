"""v10_narrow: does a block need every channel of its input? block_inputs lets a block read only the first few
channels of each stream, which shrinks that block's streaming state, its weights and its MACs. Built on v10_speed
(and through it on best/); README.md has the question and the results.
"""

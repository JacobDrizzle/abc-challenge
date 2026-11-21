"""v10_summary: can the last block's 329-sample history buffer be replaced by a few running averages per channel?
summary_taus swaps the last block for one that reads exponential moving averages, which shrinks the model's streaming
state from 25,088 to 4,480 floats at the same parameter count. Built on best/; README.md has the question and results.
"""

"""Self-evolving training: choose the training signal per unit of data.

See ``DESIGN_signal_routing.md`` for the derivation. The short version: for group-based
RL there is an exact condition under which the gradient is identically zero, it is
measurable from quantities GRPO already computes, and the two ways it can be zero need
opposite responses.
"""

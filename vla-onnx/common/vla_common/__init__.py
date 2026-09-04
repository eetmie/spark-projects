"""Shared plumbing for the LeRobot VLA -> split ONNX -> Orin Nano pipeline.

What lives here is what all three model playbooks (smolvla, xvla, evo1) do
*identically*: hash a bundle, validate a traced graph, record provenance, convert
weights to FP16, read a checkpoint header, reshape a LeRobot dataset.

What does NOT live here is how a model is cut into graphs. `_build_wrappers()` is
~200 lines in each playbook and 5% similar between them, because the cut follows the
architecture. That is the real content of an exporter and it stays model-side.
"""

"""SmolVLA inference runtime for Jetson Orin Nano.

RealSense D435i RGB in -> SmolVLA (pure TensorRT engine) -> action chunk out.
No robot control here — this is the model pipeline only.
"""

__version__ = "0.1.0"

"""FLYNN: a connectome-constrained (Drosophila FAFB v783) navigation agent trained by DAgger / BC / PPO in MuJoCo."""
import os
import sys

# Headless rendering default for servers/notebooks without a display. Override by exporting MUJOCO_GL
# yourself (e.g. MUJOCO_GL=glfw) *before* importing flynn, e.g. to use render_mode="human".
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "0.1.0"

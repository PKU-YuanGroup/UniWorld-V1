import subprocess
import sys
sys.path.append(".")
from multi_nodes.config import *

for node in nodes:
    print(node)
    subprocess.run(["ssh", f"{node}", "mkdir -p /home/node-user/.cache/torch/hub/checkpoints/"], capture_output=True, text=True)
    print(subprocess.run(["scp", "/home/node-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth", f"{node}:/home/node-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth"], capture_output=True, text=True))
    subprocess.run(["scp", "/home/node-user/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth", f"{node}:/home/node-user/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"], capture_output=True, text=True)
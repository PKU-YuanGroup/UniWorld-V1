import subprocess
import sys
import os
sys.path.append(".")
from multi_nodes.config import *

for node in nodes:
    print(node)
    result = subprocess.run(["ssh", f"{node}", f"pkill -f train_ddp.py"], capture_output=True, text=True)
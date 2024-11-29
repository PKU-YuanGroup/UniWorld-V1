import subprocess
import sys
import os
sys.path.append(".")
from multi_nodes.config import *

idx = 1
for node in nodes[1:]:
    print(node)
    output_log_path = os.path.join("/storage/lcm/WF-VAE_paper/multi_nodes/logs", f"output_{idx}.log")
    result = subprocess.run(["ssh", f"{node}", f"nohup bash /storage/lcm/WF-VAE_paper/examples/train_ddp_onlydecoder.sh {idx} > {str(output_log_path)} 2>&1 &"], capture_output=True, text=True)
    idx += 1
import os
import numpy as np
import pandas as pd
import json
from glob import glob
import argparse

VLMEvalKit = [
    'POPE_score.csv',  
    'HallusionBench_score.csv', 
    'MMBench_DEV_EN_acc.csv', 
    'MMBench_DEV_CN_acc.csv', 
    'SEEDBench_IMG_acc.csv', 
    'MMMU_DEV_VAL_acc.csv', 
    'AI2D_TEST_acc.csv', 
    'OCRBench_score.json', 
    'RealWorldQA_acc.csv', 
    'ChartQA_TEST_acc.csv', 
    'DocVQA_VAL_acc.csv', 
    'InfoVQA_VAL_acc.csv', 
    'TextVQA_VAL_acc.csv', 
    'GQA_TestDev_Balanced_acc.csv', 
    ]

def convert_01_to_100(res1, res2):
    if 0 < res1 < 1:
        res1 = res1 * 100
    if res2 is not None and 0 < res2 < 1:
        res2 = res2 * 100
    return round(res1, 1), round(res2, 1) if res2 is not None else res2

def print_VLMEvalKit_results(args):
    def get_results(result_path):
        res1, res2 = None, None
        if result_path.endswith('.csv'):
            data = pd.read_csv(result_path)
            if 'Overall' in data.columns:
                res1 = data['Overall'][0]
                if 'MMMU_DEV_VAL_acc' in result_path:
                    res2 = data['Overall'][1]
            else:
                data = pd.read_csv(result_path, index_col=[0])
                res1 = data.loc["Overall"].iloc[0]
        elif result_path.endswith('.json'):
            with open(result_path, 'r') as f:
                data = json.load(f)
                res1 = data["Final Score"]
        return convert_01_to_100(res1, res2)

    print(f"|{'-'*25}|{'-'*11}|{'-'*11}|")
    print(f"|{'VLMEvalKit':^47}|")
    print(f"|{'-'*25}|{'-'*11}|{'-'*11}|")
    for bench_name in VLMEvalKit:
        result_path = os.path.join(args.vlmevalkit_root, bench_name)
        if not os.path.exists(result_path):
            continue
        try:
            res1, res2 = get_results(result_path)
            if res2 is None:
                print(f"|{bench_name.split('.')[0]:^25}|{res1:^11}|{str(res2):^11}|")
            else:
                print(f"|{bench_name.split('.')[0]:^25}|{str(res2)+'(val)':^11}|{str(res2)+'(dev)':^11}|")
            print(f"|{'-'*25}|{'-'*11}|{'-'*11}|")
        except Exception as e:
            print(e)
    print()

parser = argparse.ArgumentParser(description="print results")
parser.add_argument("--vlmevalkit_root", type=str, default="", help="")

args = parser.parse_args()

if os.path.exists(args.vlmevalkit_root):
    print_VLMEvalKit_results(args)


'''
vlmevalkit="/storage/lb/logs/ross/ross-clip-qwen2-0p5b-pt558k-sft737k/eval/VLMEvalKit/T20250322_G8d5ba4ee"
python tools/print_results.py --vlmevalkit_root ${vlmevalkit} 
'''
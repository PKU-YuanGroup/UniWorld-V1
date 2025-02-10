pred_npz=$1

python guided-diffusion/evaluations/evaluator.py \
    /path/to/reference.npz \
    ${pred_npz}
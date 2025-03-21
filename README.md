

```
conda create -n ross_lb python=3.10 -y
conda activate ross_lb

pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
MAX_JOBS=64 pip install flash-attn==2.1.2.post3 --no-build-isolation
```

```
conda create -n ross_lb_eval python=3.10 -y

conda activate ross_lb_eval

cd evaluation/VLMEvalKit
pip install -r requirements.txt
cd ../../

MAX_JOBS=64 pip install flash-attn==2.1.2.post3 --no-build-isolation

cd evaluation/lmms-eval
pip install -e .
cd ../../
```
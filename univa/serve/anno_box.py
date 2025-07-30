import os
import json
import gradio as gr
from gradio_image_annotation import image_annotator
import torch
from transformers import MarianTokenizer, MarianMTModel

# 全局状态
DATA = []
INDEX = 0
save_path = ""
model_name = "/mnt/data/checkpoints/Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device=device, dtype=torch.bfloat16)
model = torch.compile(model)

def translate_hf(texts):
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    translated = model.generate(**batch)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 加载输入的 JSON 数据列表，并尝试加载已有注释

def load_data(json_path, img_root, save_path_arg):
    global DATA, INDEX, save_path
    save_path = save_path_arg
    with open(json_path, 'r', encoding='utf-8') as f:
        DATA = json.load(f)
    # 如果已有输出文件，加载已有注释
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            for i, item in enumerate(DATA):
                item['boxes'] = existing[i].get('boxes', []) if i < len(existing) else []
        except Exception:
            for item in DATA:
                item['boxes'] = []
    else:
        for item in DATA:
            item['boxes'] = []
    INDEX = 0
    for item in DATA:
        item['full_image_path'] = os.path.join(img_root, item['image'])
    return update_sample('next_unmarked')

# 切换样本

def update_sample(step):
    global INDEX
    if not DATA:
        return None, "请先加载 JSON 数据并指定图片根路径。", gr.update(value=None)
    if step == 'next_unmarked':
        for i in range(INDEX, len(DATA)):
            if not DATA[i].get('boxes'):
                INDEX = i
                break
        else:
            return None, "所有数据都已标注完毕！", gr.update(value=None)
    elif step == 'prev':
        INDEX = max(0, INDEX - 1)
    elif step == 'next':
        INDEX = min(len(DATA) - 1, INDEX + 1)
    # step 'jump' 或其他，直接显示当前 INDEX
    item = DATA[INDEX]
    init_ann = {'image': item['full_image_path'], 'boxes': item.get('boxes', [])}
    status = (f"样本 {INDEX+1}/{len(DATA)} - Prompt: "
              f"{translate_hf(item['prompt'])} ({item['prompt']}) - "
              f"已标注: {'是' if item.get('boxes') else '否'}")
    return init_ann, status, gr.update(value=None)

# 点击获取框，保存到输出 JSON（合并已有数据），并跳到下一个未标记

def on_annotate(ann):
    global INDEX
    DATA[INDEX]['boxes'] = ann['boxes']
    merged = []
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                merged = json.load(f)
        except Exception:
            merged = []
    if len(merged) < len(DATA):
        merged = DATA.copy()
    merged[INDEX] = DATA[INDEX]
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return update_sample('next')

# 跳转到指定样本

def jump_to_index(idx_str):
    global INDEX
    if not DATA:
        return None, "请先加载 JSON 数据并指定图片根路径。", gr.update(value=None)
    try:
        idx = int(idx_str) - 1
    except Exception:
        return None, f"请输入1~{len(DATA)}之间的整数示例序号。", gr.update(value=None)
    if idx < 0 or idx >= len(DATA):
        return None, f"索引超出范围: 1~{len(DATA)}。", gr.update(value=None)
    INDEX = idx
    return update_sample('jump')

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 在线标注 Bound Box 应用")
    with gr.Row():
        json_input = gr.Textbox(label="JSON 路径", placeholder="输入 JSON 文件路径，例如：/mnt/data/input.json")
        imgroot_input = gr.Textbox(label="图片根路径", placeholder="输入图片文件夹路径，例如：/mnt/data/images")
        save_input = gr.Textbox(label="输出 JSON 路径", placeholder="输入保存标注结果的 JSON 路径，例如：/mnt/data/annotations.json")
        load_btn = gr.Button("加载数据并恢复注释")
    status_box = gr.Textbox(label="标注状态", interactive=False)
    annotator = image_annotator(
        label_list=None,
        height=500, width=700,
        show_remove_button=True,
        enable_keyboard_shortcuts=True
    )
    with gr.Row():
        prev_btn = gr.Button("← 上一个")
        jump_input = gr.Textbox(label="跳转到样本编号", placeholder="输入样本序号，如1")
        jump_btn = gr.Button("跳转")
        annotate_btn = gr.Button("保存并下一个")
        next_btn = gr.Button("下一个 →")

    # 绑定事件
    load_btn.click(load_data, inputs=[json_input, imgroot_input, save_input], outputs=[annotator, status_box, gr.JSON()])
    prev_btn.click(lambda: update_sample('prev'), outputs=[annotator, status_box, gr.JSON()])
    next_btn.click(lambda: update_sample('next'), outputs=[annotator, status_box, gr.JSON()])
    annotate_btn.click(on_annotate, inputs=annotator, outputs=[annotator, status_box, gr.JSON()])
    jump_btn.click(jump_to_index, inputs=jump_input, outputs=[annotator, status_box, gr.JSON()])

if __name__ == "__main__":
    server_port = os.getenv('PORT', None)
    server_name = None
    if server_port is not None:
        server_port = int(server_port)
        server_name = '0.0.0.0'
    demo.launch(allowed_paths=['/'], server_port=server_port, server_name=server_name)
'''
/mnt/data/datasets/stepfun-ai/gedit_bench_anno.json
/mnt/data/datasets/stepfun-ai/gedit_bench_images
/mnt/data/datasets/stepfun-ai/gedit_bench_anno_box.json
PORT=10002 python univa/serve/anno_box.py


/mnt/data/datasets/black-forest-labs/kontext-bench/kontext_bench_anno.json
/mnt/data/datasets/black-forest-labs/kontext_bench_images
/mnt/data/datasets/black-forest-labs/kontext-bench/kontext_bench_anno_box.json
PORT=10001 python univa/serve/anno_box.py


/mnt/data/datasets/imgedit/Benchmark/imgedit_bench_anno.json
/mnt/data/datasets/imgedit/Benchmark/imgedit_bench_images
/mnt/data/datasets/imgedit/Benchmark/imgedit_bench_anno_box.json
PORT=10003 python univa/serve/anno_box.py
'''
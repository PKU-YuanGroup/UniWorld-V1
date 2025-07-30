import os
import json
import torch
import gradio as gr
import numpy as np
import random
from PIL import Image
from transformers import MarianTokenizer, MarianMTModel

# 全局状态
DATA = []
INDEX = 0
IMG_ROOT = ""
RED_BOX_IMG_ROOT = ""
WO_RED_BOX_IMG_ROOT = ""
OUTPUT_DIR = ""
DISPLAY_ORDER = []  # 用于记录每个样本的图片展示顺序

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

# 获取并展示样本
def get_sample(idx):
    sample = DATA[idx]
    # 原始路径
    src_path = os.path.join(IMG_ROOT, sample['kontext_input_image'])
    path_with = os.path.join(RED_BOX_IMG_ROOT, sample['with_redbox_image'])
    path_without = os.path.join(WO_RED_BOX_IMG_ROOT, sample['without_redbox_image'])
    # 根据 DISPLAY_ORDER 决定顺序
    swap = DISPLAY_ORDER[idx]
    if swap:
        gallery_images = [path_without, path_with]
        order_info = ['without_redbox', 'with_redbox']
    else:
        gallery_images = [path_with, path_without]
        order_info = ['with_redbox', 'without_redbox']

    # 保存上一轮 order 信息，用于提交时写入文件
    sample['_last_order'] = order_info

    en = sample['kontext_prompt_no_redbox']
    cn = translate_hf(en)
    text = f"{cn} ({en})"
    # text = ""
    # for i, turn in enumerate(sample.get("conversations", [])):
    #     prefix = "🧑 User: " if turn.get("from") == "human" else "🤖 AI: "
    #     if i == 0 and sample.get("kontext_prompt", None) is not None:
    #         en = sample.get("kontext_prompt").replace('\n', '').strip()
    #         cn = translate_hf(en)
    #         text += f"{prefix}{cn}({en})\n"
    #     elif i % 2 == 0:
    #         en = turn.get('value', '').replace('\n', '').strip()
    #         cn = translate_hf(en)
    #         text += f"{prefix}{cn}({en})\n"
    #     else:
    #         en = turn.get('value', '').replace('\n', '').strip()
    #         text += f"{prefix}{en}\n"

    status = f"当前样本: {idx + 1} / {len(DATA)}\n"
    ann_path = os.path.join(OUTPUT_DIR, f"annotation_{idx}.json")
    status += "标记状态: 已标记 ✅" if os.path.exists(ann_path) else "标记状态: 未标记 ❌"
    print(gallery_images)
    return (
        [src_path],
        gallery_images,
        gr.update(value=text),
        gr.update(value=status),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None)
    )

# 导航函数
def show_first():
    global INDEX
    INDEX = 0
    return get_sample(INDEX)

def show_next():
    global INDEX
    if INDEX < len(DATA) - 1:
        INDEX += 1
    return get_sample(INDEX)

def show_prev():
    global INDEX
    if INDEX > 0:
        INDEX -= 1
    return get_sample(INDEX)

def jump_to_index(target_idx):
    global INDEX
    try:
        target_idx = int(target_idx)
        if 0 <= target_idx < len(DATA):
            INDEX = target_idx
            return get_sample(INDEX)
        else:
            return (
                gr.update(),
                gr.update(),
                gr.update(value=f"❌ 索引超出范围 (0 - {len(DATA)-1})"),
                gr.update(value=f"❌ 无效索引: {target_idx}"),
                gr.update(),
                gr.update(),
                gr.update()
            )
    except ValueError:
        return (
            gr.update(),
            gr.update(),
            gr.update(value="❌ 索引必须是整数"),
            gr.update(value="❌ 请输入有效的数字索引"),
            gr.update(),
            gr.update(),
            gr.update()
        )

def show_first_unmarked():
    global INDEX
    for idx in range(len(DATA)):
        ann_path = os.path.join(OUTPUT_DIR, f"annotation_{idx}.json")
        if not os.path.exists(ann_path):
            INDEX = idx
            return get_sample(INDEX)
    return (
        gr.update(),
        gr.update(),
        gr.update(value="✅ 全部已经完成"),
        gr.update(value="✅ 无未标记样本"),
        gr.update(),
        gr.update(),
        gr.update()
    )

# 标注并保存结果并跳转到第一个未标记样本
def submit_annotation(region_completion, target_accuracy, stability):
    global INDEX, DATA, OUTPUT_DIR
    if region_completion is None or target_accuracy is None or stability is None:
        return (
            "❌ 请完成所有标注后再提交",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update()
        )

    sample = DATA[INDEX]
    order_info = sample['_last_order']
    result = {
        "index": INDEX,
        "first_image": order_info[0],
        "second_image": order_info[1],
        "编辑区域指令完成度": region_completion,
        "目标区域变化准确度": target_accuracy,
        "非编辑区域稳定性": stability
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"annotation_{INDEX}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    msg = f"保存成功: {out_path}"
    # 跳转到第一个未标记
    for idx in range(len(DATA)):
        if not os.path.exists(os.path.join(OUTPUT_DIR, f"annotation_{idx}.json")):
            INDEX = idx
            return (msg,) + get_sample(INDEX)
    return (msg,) + (
        gr.update(),
        gr.update(),
        gr.update(value="✅ 全部已经完成"),
        gr.update(value="✅ 无未标记样本"),
        gr.update(),
        gr.update(),
        gr.update()
    )

# 加载 JSON 数据
def load_json(json_path, image_root, red_box_image_root, wo_red_box_image_root, output_dir):
    global DATA, INDEX, IMG_ROOT, RED_BOX_IMG_ROOT, WO_RED_BOX_IMG_ROOT, OUTPUT_DIR, DISPLAY_ORDER
    try:
        with open(json_path.strip(), 'r', encoding='utf-8') as f:
            DATA = json.load(f)
    except Exception as e:
        return f"❌ Failed to load JSON: {e}", None, None, "", "", None, None, None
    IMG_ROOT = image_root.strip()
    RED_BOX_IMG_ROOT = red_box_image_root.strip()
    WO_RED_BOX_IMG_ROOT = wo_red_box_image_root.strip()
    OUTPUT_DIR = output_dir.strip()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 为每个样本随机决定展示顺序
    DISPLAY_ORDER = [random.choice([False, True]) for _ in range(len(DATA))]
    # DISPLAY_ORDER = [False for _ in range(len(DATA))]
    INDEX = 0
    return (f"✅ Loaded {len(DATA)} samples.",) + get_sample(INDEX)

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("##  Data Online comparison (数据在线筛选器)")
    with gr.Row():
        json_path    = gr.Textbox(label="JSON file path (JSON文件路径)")
        image_root   = gr.Textbox(label="Image root directory (图片根目录)")
        red_box_image_root   = gr.Textbox(label="Red Box Image root directory (Red Box图片根目录)")
        wo_red_box_image_root   = gr.Textbox(label="Without Red Box Image root directory (非Red Box图片根目录)")
        approved_dir = gr.Textbox(label="Output dir (输出目录)")
        load_btn     = gr.Button("Load JSON (加载 JSON)")
        load_status  = gr.Textbox(label="Loading status (加载状态)", interactive=False)

    with gr.Row():
        with gr.Column(scale=0.7):
            src_gallery  = gr.Gallery(label="Source Image (源图)", columns=1, allow_preview=True, show_label=True, height="auto", object_fit="contain")
            text_box     = gr.Textbox(label="Conversation Content (对话内容)", lines=2, interactive=False)
        with gr.Column():
            gallery      = gr.Gallery(label="Image Preview (图片预览/点击可放大对比)", columns=2, allow_preview=True, show_label=True, height="auto", object_fit="contain")
            sample_status = gr.Textbox(label="样本状态", interactive=False)

    with gr.Row():
        prev_btn     = gr.Button("⟵ Previous (向前)")
        next_btn     = gr.Button("Next ⟶ (向后)")
        jump_index   = gr.Number(label="索引跳转", precision=0)
        jump_btn     = gr.Button("🔎 Jump (跳转)")
        first_unmarked_btn = gr.Button("⏩ First Unmarked (跳到未标记样本)")

    with gr.Row():
        region_completion = gr.Radio(["第一张", "第二张", "平局"], label="编辑区域指令完成度", type="value")
        target_accuracy   = gr.Radio(["第一张", "第二张", "平局"], label="目标区域变化准确度", type="value")
        stability         = gr.Radio(["第一张", "第二张", "平局"], label="非编辑区域稳定性", type="value")
    submit_btn        = gr.Button("提交标注自动下一个")
    submit_status     = gr.Textbox(label="提交状态", interactive=False)

    load_btn.click(
        load_json,
        inputs=[json_path, image_root, red_box_image_root, wo_red_box_image_root, approved_dir],
        outputs=[load_status, src_gallery, gallery, text_box, sample_status, region_completion, target_accuracy, stability]
    )

    prev_btn.click(show_prev, outputs=[src_gallery, gallery, text_box, sample_status, region_completion, target_accuracy, stability])
    next_btn.click(show_next, outputs=[src_gallery, gallery, text_box, sample_status, region_completion, target_accuracy, stability])
    jump_btn.click(jump_to_index, inputs=[jump_index], outputs=[src_gallery, gallery, text_box, sample_status, region_completion, target_accuracy, stability])
    first_unmarked_btn.click(show_first_unmarked, outputs=[src_gallery, gallery, text_box, sample_status, region_completion, target_accuracy, stability])

    submit_btn.click(
        submit_annotation,
        inputs=[region_completion, target_accuracy, stability],
        outputs=[submit_status, src_gallery, gallery, text_box, sample_status, region_completion, target_accuracy, stability]
    )

if __name__ == "__main__":
    server_port = os.getenv('PORT', None)
    server_name = None
    if server_port is not None:
        server_port = int(server_port)
        server_name = '0.0.0.0'
    demo.launch(allowed_paths=['/'], server_port=server_port, server_name=server_name)



'''
/mnt/data/lb/6.5/UniWorld-V1/comfyui/notebooks/compare_json_test_result_dir_laion_remove_part0_100.json
/mnt/data/lzj/codes/shitedit_comfyui/results_remove
/mnt/data/lb/6.5/UniWorld-V1/comfyui/red_box_test_result_dir_laion_remove_part0
/mnt/data/lb/6.5/UniWorld-V1/comfyui/wo_red_box_test_result_dir_laion_remove_part0
/mnt/data/lb/6.5/UniWorld-V1/comfyui/compare_red_box_test_result_dir_laion_remove_part0

PORT=10000 python univa/serve/compare_data.py
'''
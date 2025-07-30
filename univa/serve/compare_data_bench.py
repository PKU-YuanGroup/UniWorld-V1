import os
import json
import torch
import gradio as gr
import numpy as np
import random
from transformers import MarianTokenizer, MarianMTModel
import json
import os
from PIL import Image, ImageDraw, ImageColor
from typing import Union, Sequence, Tuple


# 全局状态
DATA = []
INDEX = 0
IMG_ROOT = ""
RED_BOX_PREFIX = ""
OUTPUT_DIR = ""
DISPLAY_ORDER = []  # 用于记录每个样本的图片展示顺序

model_name = "/mnt/data/checkpoints/Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device=device, dtype=torch.bfloat16)
model = torch.compile(model)



def draw_boxes(
    img_input: Union[str, Image.Image],
    boxes: Sequence[Tuple[int, int, int, int]],
    width: int = 50,
    color: Union[str, Tuple[int, int, int]] = "red",
    border_opacity: float = 0.6
) -> Image.Image:
    """
    在图片上画多个半透明边框（无填充）。

    Args:
        img_input: 图片路径或 PIL.Image 对象。
        boxes: List of boxes，每个 box 是 (xmin, ymin, xmax, ymax)。
        width: 边框线宽，默认为 50 像素。
        color: 边框颜色，可为字符串（如 "red"）或 RGB 三元组。
        border_opacity: 边框透明度，0 完全透明，1 完全不透明，默认 0.5。

    Returns:
        带有半透明边框的 PIL.Image 对象（RGBA 模式）。
    """
    # 1. 打开并转换为 RGBA
    if isinstance(img_input, str):
        base = Image.open(img_input).convert("RGBA")
    else:
        base = img_input.convert("RGBA")

    # 2. 新建透明叠加层
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # 3. 解析 color 并计算带透明度的 RGBA
    if isinstance(color, str):
        rgb = ImageColor.getrgb(color)
    else:
        rgb = color
    alpha = int(255 * border_opacity)
    rgba = (rgb[0], rgb[1], rgb[2], alpha)

    # 4. 在 overlay 上画只有 outline 的半透明框
    for xmin, ymin, xmax, ymax in boxes:
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            outline=rgba,
            width=width
        )

    # 5. 合成并返回
    return Image.alpha_composite(base, overlay)


def translate_hf(texts):
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    translated = model.generate(**batch)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 获取并展示样本
def get_sample(idx):
    sample = DATA[idx]
    # 原始路径
    basename = sample['image']
    src_path = os.path.join(IMG_ROOT, basename)
    boxes = [[box['xmin'], box['ymin'], box['xmax'], box['ymax']] for box in sample['boxes']]
    # image = draw_boxes(src_path, boxes, width=10, color="red")
    image = src_path
    path_with = os.path.join(IMG_ROOT, f'{RED_BOX_PREFIX}{basename}')
    path_without = os.path.join(IMG_ROOT, f'kontext_out_{basename}')
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

    en = sample['prompt']
    cn = translate_hf(en)
    text = f"{cn} ({en})"

    status = f"当前样本: {idx + 1} / {len(DATA)}\n"
    ann_path = os.path.join(OUTPUT_DIR, f"annotation_{idx}.json")
    status += "标记状态: 已标记 ✅" if os.path.exists(ann_path) else "标记状态: 未标记 ❌"
    print(gallery_images)
    return (
        [image],
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
def load_json(json_path, image_root, red_box_prefix, output_dir):
    global DATA, INDEX, IMG_ROOT, RED_BOX_PREFIX, OUTPUT_DIR, DISPLAY_ORDER
    try:
        with open(json_path.strip(), 'r', encoding='utf-8') as f:
            DATA = json.load(f)
    except Exception as e:
        return f"❌ Failed to load JSON: {e}", None, None, "", "", None, None, None
    IMG_ROOT = image_root.strip()
    RED_BOX_PREFIX = red_box_prefix.strip()
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
    gr.Markdown(
        """
        **指令完成度**：只需要关注指令完成度，比如指令：让一个人举起左手。第一张图举起左手同时右手也举起了，第二张图举起右手。仍然判断平局
        
        **目标区域变化准确度**：是不是只在改变的地方变，不改变的地方不变。比如让右边加个人，第一张图加成了狗，第二张图加了人，但依然平局
        
        **参考图一致性**：和参考图的一致性。比如都编辑成功了，但是第一个图比第二个图更像参考图的人，那就第一张图赢。但如果第一张图少了车，第二张图多了车，这应当在目标区域变化准确度进行判断而不是编辑一致性
        """
    )
    gr.Markdown(
        """
        **特别备注**：如果一个图片中出现了红框，那么请忽略这个红框，因为基模型未经微调
        """
    )
    with gr.Row():
        json_path    = gr.Textbox(label="JSON file path (JSON文件路径)")
        image_root   = gr.Textbox(label="Image root directory (图片根目录)")
        red_box_prefix   = gr.Textbox(label="Red Box Image Prefix (Red Box图片前缀)")
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
        region_completion = gr.Radio(["第一张", "第二张", "平局"], label="指令完成度", type="value")
        target_accuracy   = gr.Radio(["第一张", "第二张", "平局"], label="目标区域变化准确度", type="value")
        stability         = gr.Radio(["第一张", "第二张", "平局"], label="参考图一致性", type="value")
    submit_btn        = gr.Button("提交标注自动下一个")
    submit_status     = gr.Textbox(label="提交状态", interactive=False)

    load_btn.click(
        load_json,
        inputs=[json_path, image_root, red_box_prefix, approved_dir],
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
/mnt/data/datasets/imgedit/Benchmark/imgedit_bench_anno_box.json
/mnt/data/datasets/imgedit/Benchmark/imgedit_bench_images
kontext_out_red_box_w10_
/mnt/data/lb/6.5/UniWorld-V1/comfyui/compare_imgedit_bench_w10
PORT=10000 python univa/serve/compare_data_bench.py

/mnt/data/datasets/black-forest-labs/kontext-bench/kontext_bench_anno_box.json
/mnt/data/datasets/black-forest-labs/kontext_bench_images
kontext_out_red_box_w30_
/mnt/data/lb/6.5/UniWorld-V1/comfyui/compare_kontext_bench_w30
PORT=10001 python univa/serve/compare_data_bench.py

/mnt/data/datasets/stepfun-ai/gedit_bench_anno_box.json
/mnt/data/datasets/stepfun-ai/gedit_bench_images
kontext_out_red_box_w10_
/mnt/data/lb/6.5/UniWorld-V1/comfyui/compare_gedit_bench_w10
PORT=10002 python univa/serve/compare_data_bench.py
'''
import os
import json
import torch
import gradio as gr
from PIL import Image, ImageFilter, ImageChops
import cv2
import numpy as np
from PIL import Image
from transformers import MarianTokenizer, MarianMTModel
# 全局状态
DATA = []
INDEX = 0
IMG_ROOT = ""
APPROVED_DIR = ""
REJECTED_DIR = ""


model_name = "/mnt/data/checkpoints/Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
model = model.to(device=device, dtype=torch.bfloat16)
model = torch.compile(model)
def translate_hf(texts):
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    translated = model.generate(**batch)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def check_index():
    global DATA, INDEX
    assert 0 <= INDEX < len(DATA), f'0 <= INDEX ({INDEX}) < len(DATA) {len(DATA)}'

# 加载 JSON 数据
def load_json(json_path, image_root, approved_dir, rejected_dir):
    global DATA, INDEX, IMG_ROOT, APPROVED_DIR, REJECTED_DIR
    try:
        with open(json_path.strip(), 'r', encoding='utf-8') as f:
            DATA = json.load(f)
    except Exception as e:
        return f"Failed to load JSON: {e}", [], "", ""
    IMG_ROOT = image_root.strip()
    APPROVED_DIR = approved_dir.strip()
    REJECTED_DIR = rejected_dir.strip()
    os.makedirs(APPROVED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)
    INDEX = 0
    gallery_imgs, text, nav, mark_status = get_sample(INDEX)
    check_index()
    return f"Loaded {len(DATA)} samples.", gallery_imgs, text, nav, mark_status




def overlay_mask_border_red_cv(
    image: Image.Image,
    mask: Image.Image,
    border_width: int = 5,
) -> Image.Image:
    """
    Overlay only the outermost 'border_width'-pixel ring of a binary mask as red on the image.
    
    Args:
        image: PIL.Image in RGB or RGBA.
        mask: PIL.Image in any mode; white (255) means foreground.
        border_width: thickness of the border in pixels.
        
    Returns:
        PIL.Image (RGBA) with red border overlaid.
    """
    # 1) Make sure mask matches image size:
    rgba = image.convert("RGBA")
    if mask.size != rgba.size:
        mask = mask.resize(rgba.size, Image.NEAREST)
    
    # 2) PIL → OpenCV arrays
    base_np = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)
    mask_np  = np.array(mask.convert("L"))
    
    # 3) Binarize in C
    _, bin_mask = cv2.threshold(mask_np, 128, 255, cv2.THRESH_BINARY)
    
    # 4) Morphological gradient: outer border
    k = 2 * border_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    border = cv2.morphologyEx(bin_mask, cv2.MORPH_GRADIENT, kernel)
    
    # 5) Make a solid-red BGRA overlay
    red_layer = np.zeros_like(base_np)
    red_layer[..., 2] = 255  # R channel
    red_layer[..., 3] = 255  # Alpha
    
    # 6) Blend via np.where
    #   border>0 is shape (H,W).  Broadcast to (H,W,4) automatically.
    mask4 = border[..., None] > 0
    result_np = np.where(mask4, red_layer, base_np)
    
    check_index()
    # 7) Back to PIL
    return Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGRA2RGBA))

# def overlay_mask_border_red(
#     image: str,
#     mask: str,
#     border_width: int = 10
# ):
#     """
#     只保留 mask 的最外边一圈，将其叠加为红色到原图上。

#     Args:
#         image_path: 原图路径。
#         mask_path: 二值 mask 路径（白色255表示目标区域）。
#         output_path: 合成后图像保存路径。
#         border_width: 边缘宽度，单位像素，默认为1。
#     """
#     # 1. 读图
#     base = image.convert("RGBA")
#     mask = mask.convert("L")
#     if mask.size != base.size:
#         mask = mask.resize(base.size, Image.NEAREST)

#     # 2. 二值化（阈值128）
#     bin_mask = mask.point(lambda p: 255 if p > 128 else 0)

#     # 3. 腐蚀：使用 MinFilter(kernel_size = 2*border_width+1)
#     kernel = 2 * border_width + 1
#     eroded = bin_mask.filter(ImageFilter.MinFilter(kernel))

#     # 4. 最外圈 = 原 mask – 腐蚀后 mask
#     border = ImageChops.subtract(bin_mask, eroded)

#     # 5. 创建红色层（全不透明）
#     red_overlay = Image.new("RGBA", base.size, (255, 0, 0, 255))

#     # 6. 在原图上只粘贴边缘部分
#     result = base.copy()
#     result.paste(red_overlay, mask=border)
#     return result

# 判断当前样本是否已标记
def get_sample_mark_status(idx):
    approved_path = os.path.join(APPROVED_DIR, f"sample_{idx}.json")
    rejected_path = os.path.join(REJECTED_DIR, f"sample_{idx}.json")
    if os.path.exists(approved_path):
        return f"✅ This sample is already APPROVED."
    elif os.path.exists(rejected_path):
        return f"❌ This sample is already REJECTED."
    else:
        return "📝 Unmarked."
    check_index()

# 获取并展示样本
def get_sample(idx):
    sample = DATA[idx]
    img_val = sample.get("image", [])
    img_list = [img_val] if isinstance(img_val, str) else (img_val or [])
    full_paths = [Image.open(os.path.join(IMG_ROOT, p)) for p in img_list if os.path.exists(os.path.join(IMG_ROOT, p))]
    text = ""
    for i, turn in enumerate(sample.get("conversations", [])):
        prefix = "🧑 User: " if turn.get("from") == "human" else "🤖 AI: "
        if i == 0 and sample.get("kontext_prompt", None) is not None:
            en = sample.get("kontext_prompt").replace('\n', '').strip()
            cn = translate_hf(en)
            text += f"{prefix}{cn}({en})\n"
        elif i % 2 == 0:
            en = turn.get('value', '').replace('\n', '').strip()
            cn = translate_hf(en)
            text += f"{prefix}{cn}({en})\n"
        else:
            en = turn.get('value', '').replace('\n', '').strip()
            text += f"{prefix}{en}\n"
    nav = f"Sample {idx+1}/{len(DATA)}"
    mark_status = get_sample_mark_status(idx)

    kontext_input_image = sample.get("kontext_input_image", None)
    if kontext_input_image is not None:
        full_paths = [Image.open(os.path.join(IMG_ROOT, kontext_input_image)), full_paths[1]]
    assert len(full_paths) == 2 or len(full_paths) == 3
    if len(full_paths) == 3:
        full_paths = [overlay_mask_border_red_cv(full_paths[0], full_paths[-1]), full_paths[1]]

    check_index()
    return full_paths, text, nav, mark_status

# 导航函数
def show_first():
    global INDEX
    INDEX = 0
    check_index()
    return get_sample(INDEX)

def show_next():
    global INDEX
    if INDEX < len(DATA) - 1:
        INDEX += 1
    check_index()
    return get_sample(INDEX)

def show_prev():
    global INDEX
    if INDEX > 0:
        INDEX -= 1
    check_index()
    return get_sample(INDEX)

def show_first_unmarked():
    global INDEX
    approved_indices = []
    if os.path.exists(APPROVED_DIR):
        approved_indices = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(APPROVED_DIR) if f.startswith("sample_") and f.endswith(".json")]
    rejected_indices = []
    if os.path.exists(REJECTED_DIR):
        rejected_indices = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(REJECTED_DIR) if f.startswith("sample_") and f.endswith(".json")]
    marked_indices = set(approved_indices + rejected_indices)
    for idx in range(len(DATA)):
        if idx not in marked_indices:
            INDEX = idx
            result_text = f"Jumped to first unmarked sample at index {INDEX}."
            gallery_imgs, text, nav, mark_status = get_sample(INDEX)
            return result_text, gallery_imgs, text, nav, mark_status
    result_text = "✅ All samples have been marked."
    gallery_imgs, text, nav, mark_status = get_sample(INDEX)
    check_index()
    return result_text, gallery_imgs, text, nav, mark_status

def jump_to_index(target_idx):
    global INDEX
    if 0 <= target_idx < len(DATA):
        INDEX = target_idx
        check_index()
        return f"Jumped to sample {INDEX}.", *get_sample(INDEX)
    else:
        check_index()
        return f"Invalid index: {target_idx}.", *get_sample(INDEX)

# 标记统计
def get_marked_status():
    approved_files = os.listdir(APPROVED_DIR) if os.path.exists(APPROVED_DIR) else []
    rejected_files = os.listdir(REJECTED_DIR) if os.path.exists(REJECTED_DIR) else []
    marked = len(approved_files) + len(rejected_files)
    unmarked = len(DATA) - marked
    check_index()
    return f"✅ Marked: {marked}, ❌ Unmarked: {unmarked}"

# 标记并保存样本
def mark_approve():
    global INDEX
    sample = DATA[INDEX]
    out_path = os.path.join(APPROVED_DIR, f"sample_{INDEX}.json")
    reject_path = os.path.join(REJECTED_DIR, f"sample_{INDEX}.json")
    if os.path.exists(reject_path):
        os.remove(reject_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    check_index()
    return f"✅ Approved sample {INDEX} saved to {APPROVED_DIR}.\n" + get_marked_status()

def mark_reject():
    global INDEX
    sample = DATA[INDEX]
    out_path = os.path.join(REJECTED_DIR, f"sample_{INDEX}.json")
    approve_path = os.path.join(APPROVED_DIR, f"sample_{INDEX}.json")
    if os.path.exists(approve_path):
        os.remove(approve_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    check_index()
    return f"❌ Rejected sample {INDEX} saved to {REJECTED_DIR}.\n" + get_marked_status()

# Approve/Reject 并跳转到下一个未标记的样本
def approve_and_next():
    status = mark_approve()
    result_text, gallery_imgs, text, nav, mark_status = show_first_unmarked()
    check_index()
    return (
        status + "\n" + result_text,
        gr.update(value=gallery_imgs),
        gr.update(value=text),
        gr.update(value=nav),
        gr.update(value=mark_status),
    )

def reject_and_next():
    status = mark_reject()
    result_text, gallery_imgs, text, nav, mark_status = show_first_unmarked()
    check_index()
    return (
        status + "\n" + result_text,
        gr.update(value=gallery_imgs),
        gr.update(value=text),
        gr.update(value=nav),
        gr.update(value=mark_status),
    )

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("""
    ## 数据审核规则说明

    Gallery中每组展示两个或三个图像：

    1. **原图**：第一个图像。
    2. **编辑后图**：第二个图像。
    3. **差分图**：仅在非风格迁移任务中出现，显示原图与编辑图之间的像素差异（白色区域代表变化）。

    符合要求点击approve，不符合要求点击reject，按照高标准来筛选
    ### 一、文本编辑任务

    -  首先查看差分图：
      - 若差分图中**大面积出现白色**区域（表示图像修改幅度大或位置错误），→ **Reject**。
    -  检查编辑后图：
      - 若图中出现**红框**，→ **Reject**。
    -  判断是否根据 prompt 对图中文字进行了修改：
      - 若未修改或修改不符合 prompt 描述，→ **Reject**。

    ### 二、Add / Remove / Compose / Alter 类任务

    -  首先查看差分图：
      - 若差分区域**偏离 prompt 描述位置**，即错误位置发生了大变化，→ **Reject**。
    -  检查编辑后图：
      - 若图中出现**红框**，→ **Reject**。
    -  判断是否满足 prompt：
      - 若内容不准确或完成度低（如2个步骤只完成了1个），→ **Reject**。

    ### 三、风格迁移任务

    -  检查编辑图的一致性：
      - 若整体风格改变导致**内容丢失或不一致**，→ **Reject**。
      - 若黑白图被错误转换为彩色图，反正同理，→ **Reject**。
      - 若不满足指定风格（如油画风、水彩风等，标注前先上网搜索了解该风格），→ **Reject**。
                
    """)
    with gr.Row():
        json_path    = gr.Textbox(label="JSON file path (JSON文件路径)")
        image_root   = gr.Textbox(label="Image root directory (图片根目录)")
        approved_dir = gr.Textbox(label="Approved output dir (选择输出目录)")
        rejected_dir = gr.Textbox(label="Rejected output dir (拒绝输出目录)")
        load_btn     = gr.Button("Load JSON (加载 JSON)")
        load_status  = gr.Textbox(label="Loading status (加载状态)", interactive=False)

    with gr.Row():
        gallery      = gr.Gallery(label="Image Preview (图片预览/点击图片可放大对比)", columns=2, allow_preview=True, show_label=True, height="auto", object_fit="contain")
        with gr.Column():
            text_box     = gr.Textbox(label="Conversation Content (对话内容)", lines=2, interactive=False)
            result_status = gr.Textbox(label="Result Status (结果状态)", interactive=False)
            with gr.Row():
                nav_status   = gr.Textbox(label="Navigation (导航索引)", interactive=False)
                mark_status  = gr.Textbox(label="Mark Status (标注状态)", interactive=False)

            with gr.Row():
                first_btn           = gr.Button("⏪ Jump to First (跳转第一个)")
                first_unmarked_btn  = gr.Button("🔎 Jump to First Unmarked (跳转第一个未标记)")
                prev_btn            = gr.Button("⟵ Previous (向前)")
                next_btn            = gr.Button("Next ⟶ (向后)")
            with gr.Row():
                index_input = gr.Number(label="Jump to index (跳转至)", precision=0)
                jump_btn    = gr.Button("🔎 Jump (跳转)")
                approve_btn = gr.Button("✅ Approve (选择)")
                reject_btn  = gr.Button("❌ Reject (拒绝)")

    # 绑定交互
    load_btn.click(load_json,
                   inputs=[json_path, image_root, approved_dir, rejected_dir],
                   outputs=[load_status, gallery, text_box, nav_status, mark_status])
    first_btn.click(show_first, outputs=[gallery, text_box, nav_status, mark_status])
    first_unmarked_btn.click(show_first_unmarked, outputs=[result_status, gallery, text_box, nav_status, mark_status])
    prev_btn.click(show_prev, outputs=[gallery, text_box, nav_status, mark_status])
    next_btn.click(show_next, outputs=[gallery, text_box, nav_status, mark_status])
    jump_btn.click(jump_to_index, inputs=[index_input], outputs=[result_status, gallery, text_box, nav_status, mark_status])
    approve_btn.click(approve_and_next,
                      inputs=[],
                      outputs=[result_status, gallery, text_box, nav_status, mark_status])
    reject_btn.click(reject_and_next,
                     inputs=[],
                     outputs=[result_status, gallery, text_box, nav_status, mark_status])

if __name__ == "__main__":
    server_port = os.getenv('PORT', None)
    server_name = None
    if server_port is not None:
        server_port = int(server_port)
        server_name = '0.0.0.0'
    demo.launch(allowed_paths=['/'], server_port=server_port, server_name=server_name)


'''
CUDA_VISIBLE_DEVICES=0 PORT=10000 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=1 PORT=10001 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=2 PORT=10002 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10003 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10004 python univa/serve/pick_data.py

CUDA_VISIBLE_DEVICES=0 PORT=10005 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=1 PORT=10006 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=2 PORT=10007 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10008 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10009 python univa/serve/pick_data.py

CUDA_VISIBLE_DEVICES=0 PORT=10010 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=1 PORT=10011 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=2 PORT=10012 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10013 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10014 python univa/serve/pick_data.py

CUDA_VISIBLE_DEVICES=0 PORT=10015 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=1 PORT=10016 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=2 PORT=10017 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10018 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10019 python univa/serve/pick_data.py

CUDA_VISIBLE_DEVICES=0 PORT=10020 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=1 PORT=10021 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=2 PORT=10022 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10023 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10024 python univa/serve/pick_data.py

CUDA_VISIBLE_DEVICES=0 PORT=10025 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=1 PORT=10026 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=2 PORT=10027 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10028 python univa/serve/pick_data.py
CUDA_VISIBLE_DEVICES=3 PORT=10029 python univa/serve/pick_data.py
'''
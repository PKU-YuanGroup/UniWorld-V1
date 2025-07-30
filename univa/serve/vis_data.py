import os
import json
import torch
import gradio as gr
from PIL import Image
from transformers import MarianTokenizer, MarianMTModel

# 全局状态
DATA = []
INDEX = 0
IMG_ROOT = ""
FILTER_MODE = "All"  # All, Correct Only, Incorrect Only
FILTERED_INDICES = []

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

# 生成过滤后的索引列表
def update_filtered_indices():
    global FILTERED_INDICES
    if FILTER_MODE == "Correct Only":
        FILTERED_INDICES = [i for i, s in enumerate(DATA) if s.get('correct', False)]
    elif FILTER_MODE == "Incorrect Only":
        FILTERED_INDICES = [i for i, s in enumerate(DATA) if not s.get('correct', False)]
    else:
        FILTERED_INDICES = list(range(len(DATA)))

# 获取并展示样本
def get_sample(idx):
    sample = DATA[idx]
    gallery_images = [os.path.join(IMG_ROOT, i) for i in sample['image']]
    gallery_images.append(os.path.join(IMG_ROOT, sample['kontext_input_image']))

    # 翻译 prompts
    en_no = sample.get('kontext_prompt_no_redbox', 'None')
    cn_no = translate_hf(en_no)
    kontext_prompt_no_redbox = f"{cn_no} ({en_no})\n\n"

    en = sample['kontext_prompt']
    cn = translate_hf(en)
    kontext_prompt = f"{cn} ({en})\n\n"

    # 对话内容
    text = ""
    for turn in sample.get("conversations", []):
        prefix = "🧑 User: " if turn.get("from") == "human" else "🤖 AI: "
        content = turn.get('value', '').replace('\n', '').strip()
        if turn.get('from') == 'human':
            cn = translate_hf(content)
            text += f"{prefix}{cn} ({content})\n"
        else:
            text += f"{prefix}{content}\n"

    correct_status = f"Correct: {sample.get('correct', False)}"
    status = f"样本 {FILTER_MODE} (共 {len(FILTERED_INDICES)}) | 当前: {FILTERED_INDICES.index(idx)+1} / {len(FILTERED_INDICES)}"

    return (
        gallery_images,
        gr.update(value=text),
        gr.update(value=kontext_prompt),
        gr.update(value=kontext_prompt_no_redbox),
        gr.update(value=correct_status),
        gr.update(value=status)
    )

# 导航函数

def show_next():
    global INDEX
    pos = FILTERED_INDICES.index(INDEX)
    if pos < len(FILTERED_INDICES) - 1:
        INDEX = FILTERED_INDICES[pos + 1]
    return get_sample(INDEX)


def show_prev():
    global INDEX
    pos = FILTERED_INDICES.index(INDEX)
    if pos > 0:
        INDEX = FILTERED_INDICES[pos - 1]
    return get_sample(INDEX)


def jump_to_index(target_idx):
    global INDEX
    try:
        t = int(target_idx)
        if t < 0 or t >= len(DATA):
            raise IndexError
        if t not in FILTERED_INDICES:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(value=f"❌ 样本 {t} 不符合 {FILTER_MODE} 过滤条件")
            )
        INDEX = t
        return get_sample(INDEX)
    except ValueError:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(value="❌ 索引必须是整数")
        )
    except IndexError:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(value=f"❌ 索引超出范围 (0 - {len(DATA)-1})")
        )

# 更改过滤模式回调

def update_filter(mode):
    global FILTER_MODE, INDEX
    FILTER_MODE = mode
    update_filtered_indices()
    if not FILTERED_INDICES:
        return ([], gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value="❌ 无符合条件的样本"))
    INDEX = FILTERED_INDICES[0]
    return get_sample(INDEX)

# 加载 JSON 数据

def load_json(json_path, image_root):
    global DATA, INDEX, IMG_ROOT
    try:
        with open(json_path.strip(), 'r', encoding='utf-8') as f:
            DATA = json.load(f)
    except Exception as e:
        return f"❌ 无法加载 JSON: {e}", None, None, None, None, None
    IMG_ROOT = image_root.strip()
    INDEX = 0
    update_filtered_indices()
    if not FILTERED_INDICES:
        return f"✅ Loaded {len(DATA)} 样本", [], None, None, None, "❌ 无符合条件样本"
    INDEX = FILTERED_INDICES[0]
    return (f"✅ Loaded {len(DATA)} 样本",) + get_sample(INDEX)

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## Data Online Vis (数据在线可视化)")
    with gr.Row():
        json_path    = gr.Textbox(label="JSON file path (JSON文件路径)")
        image_root   = gr.Textbox(label="Image root directory (图片根目录)")
        load_btn     = gr.Button("Load JSON (加载 JSON)")
        load_status  = gr.Textbox(label="Loading status (加载状态)", interactive=False)

    # 过滤开关
    filter_dropdown = gr.Dropdown(
        choices=["All", "Correct Only", "Incorrect Only"],
        value="All",
        label="Filter by Correct Flag (过滤条件)"
    )

    gallery      = gr.Gallery(label="Image Preview", columns=3, allow_preview=True, interactive=False)
    with gr.Row():
        text_box     = gr.Textbox(label="对话内容", lines=4, interactive=False)
        kontext_prompt     = gr.Textbox(label="kontext_prompt", lines=4, interactive=False)
        kontext_prompt_no_redbox     = gr.Textbox(label="kontext_prompt_no_redbox", lines=4, interactive=False)

    with gr.Row():
        correct_display = gr.Textbox(label="Correct Flag", interactive=False)
        sample_status = gr.Textbox(label="样本状态", interactive=False)
        prev_btn     = gr.Button("⟵ Previous")
        next_btn     = gr.Button("Next ⟶")
        jump_index   = gr.Number(label="跳转到样本索引", precision=0)
        jump_btn     = gr.Button("🔎 Jump")

    # 事件绑定
    load_btn.click(
        load_json,
        inputs=[json_path, image_root],
        outputs=[load_status, gallery, text_box, kontext_prompt, kontext_prompt_no_redbox, correct_display, sample_status]
    )
    filter_dropdown.change(
        update_filter,
        inputs=[filter_dropdown],
        outputs=[gallery, text_box, kontext_prompt, kontext_prompt_no_redbox, correct_display, sample_status]
    )
    prev_btn.click(show_prev, outputs=[gallery, text_box, kontext_prompt, kontext_prompt_no_redbox, correct_display, sample_status])
    next_btn.click(show_next, outputs=[gallery, text_box, kontext_prompt, kontext_prompt_no_redbox, correct_display, sample_status])
    jump_btn.click(jump_to_index, inputs=[jump_index], outputs=[gallery, text_box, kontext_prompt, kontext_prompt_no_redbox, correct_display, sample_status])

if __name__ == "__main__":
    server_port = os.getenv('PORT', None)
    server_name = None
    if server_port is not None:
        server_port = int(server_port)
        server_name = '0.0.0.0'
    demo.launch(allowed_paths=['/'], server_port=server_port, server_name=server_name)

'''
/mnt/data/lzj/codes/uniworld_data/laion_remove_part0_edit_mask_filtered_11856_fix.json
/mnt/data/lzj/codes/shitedit_comfyui/results_remove

PORT=10000 python univa/serve/vis_data.py
'''
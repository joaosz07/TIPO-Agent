import os
import glob
import re
import json
import base64
from io import BytesIO

import torch
import numpy as np
import gradio as gr
from llama_cpp import Llama
from objprint import objprint
from sdeval.corrupt import AICorruptMetrics
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.logging import logger
from kgen.formatter import seperate_tags, apply_format

from diff import load_model, generate_from_text
from config import *


tipo.BAN_TAGS = BAN_TAGS

sdxl_pipe = load_model(T2I_MODEL, device=T2I_DEVICE)
sdxl_pipe.to(T2I_DEVICE)
models.download_gguf(
    models.tipo_model_list[0][0],
    models.tipo_model_list[0][1][0],
)
models.load_model(
    "TIPO-500M-ft_TIPO-500M-ft-F16.gguf",
    gguf=True,
    device=TIPO_DEVICE,
    main_gpu=TIPO_MAIN_GPU,
    tensor_split=TIPO_SPLIT,
)

corrupt_score = AICorruptMetrics()
jina_emb = SentenceTransformer(
    "jinaai/jina-embeddings-v2-small-en", # switch to en/zh for English or Chinese
    trust_remote_code=True
)
jina_emb.max_seq_length = 512


def text_sim(text1, text2):
    return cos_sim(*jina_emb.encode([text1, text2]))


def task(tags, nl_prompt):
    width = 1024
    height = 1024
    meta, operations, general, nl_prompt = tipo.parse_tipo_request(
        seperate_tags(tags.split(",")),
        nl_prompt,
        tag_length_target="long",
        generate_extra_nl_prompt=True,
    )
    meta["aspect_ratio"] = f"{width / height:.1f}"
    result, timing = tipo.tipo_runner(meta, operations, general, nl_prompt)
    return apply_format(result, DEFAULT_FORMAT)


llm = None
current_model = None


# Function to get available models
def get_models(models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    model_files = glob.glob(f"{models_dir}/*.gguf")
    return [os.path.basename(model) for model in model_files]


# Initialize chat history
def init_history():
    return []


# Function to convert image to base64 string
def image_to_base64(img, format="JPEG", quality=95):
    buffered = BytesIO()
    img.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


# Mock image generation service - replace with your actual service
def generate_image_from_prompt(prompt_data, tipo_prompt_per_data=4, choosed_prompt_per_data=2, image_per_prompt=3):
    prompt_data = {k.lower(): v for k, v in prompt_data.items()}
    objprint(prompt_data)

    aspect_ratio = prompt_data.get("aspect ratio", None) or prompt_data.get(
        "Aspect ratio", None
    )
    if aspect_ratio is None:
        aspect_ratio = "1:1"
    aspect_ratio = re.search(r"(\d+):(\d+)", aspect_ratio)
    aspect_ratio = aspect_ratio.groups() if aspect_ratio else ("1", "1")
    try:
        width = int(aspect_ratio[0])
        height = int(aspect_ratio[1])
    except ValueError:
        width = 1
        height = 1
    base_resolution = BASE_SIZE
    unit_res = (base_resolution / (width * height)) ** 0.5
    width = int(unit_res * width) // 32 * 32
    height = int(unit_res * height) // 32 * 32

    nl_prompt = prompt_data.get("brief", None) or prompt_data.get("Brief", None)
    if nl_prompt is None:
        nl_prompt = max(prompt_data.values(), key=len)
    prompts = [
        task("masterpiece, newest, absurdres, safe", nl_prompt)
        for _ in range(tipo_prompt_per_data)
    ]
    sim_scores = [
        text_sim(nl_prompt, prompt) for prompt in prompts
    ]
    choosed_prompt = sorted(
        zip(prompts, sim_scores), key=lambda x: x[1], reverse=True
    )[:choosed_prompt_per_data]
    choosed_prompt = [x[0] for x in choosed_prompt]

    logger.info(
        f"generate image with prompt: {choosed_prompt}, width: {width}, height: {height}"
    )

    prompts = choosed_prompt*image_per_prompt
    results = generate_from_text(
        sdxl_pipe,
        prompts,
        prompt_data.get("negative prompt", "") + DEFAULT_NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=T2I_DIFF_STEP,
        guidance_scale=T2I_DIFF_CFG,
    )
    torch.cuda.empty_cache()
    scores = corrupt_score.score(results, mode="seq")
    # scores = np.random.randn(len(results))  # Mock scores, replace with actual scoring
    result = results[np.argmax(scores)]
    prompt = prompts[np.argmax(scores)]
    return result, prompt


def generate_img(art_piece):
    # Generate the image
    img, generated_prompt = generate_image_from_prompt(art_piece)
    img_b64 = image_to_base64(img)

    # Create alt string for the image
    # need to replace the special characters, " and \n
    generated_prompt = generated_prompt.replace('"', "&quot;").replace("\n", " ")

    # Create HTML with the image and caption
    html_images = f"""
    <img 
        src="{img_b64}" 
        alt="{generated_prompt}" 
        style="
            max-width: 75%; 
            max-height: 30vh; 
            width: auto; 
            height: auto; 
            border-radius: 8px; 
            margin: 10px 0;
        " 
    />
    <p style="font-size: 10px; color: #999999;">{generated_prompt}</p>
    """.strip().replace(
        "\n", " "
    )
    return html_images


# Function to generate model response
def generate_response(
    message,
    history,
    system_prompt,
    model_path,
    temperature,
    max_tokens,
    top_p,
    min_p,
    top_k,
    repeat_penalty,
    image_generation_enabled=True,
):
    global llm, current_model
    # Load model with the specified parameters
    model_full_path = os.path.join("models", model_path)
    if current_model != model_full_path:
        llm = Llama(
            model_path=model_full_path,
            n_ctx=32768,  # Context window
            n_gpu_layers=LLM_GPU_LAYERS,  # Auto-detect GPU layers
            chat_format=LLM_CHAT_FORMAT,
            verbose=True,
            main_gpu=LLM_MAIN_GPU,
            tensor_split=LLM_SPLIT,  # Split model across GPU0,1
        )  # put LLM in sub GPU
        current_model = model_full_path

    # Format history into messages for the chat completion API
    messages = []

    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})

        # Remove HTML tags from history when sending to the model
        if assistant_msg and isinstance(assistant_msg, str):
            # Remove HTML image tags
            while find_alt := re.search(r'<img.*?alt="(.*?)".*?>', assistant_msg):
                alt_text = find_alt.group(1)
                # Replace the image tag with the alt text
                assistant_msg = assistant_msg.replace(
                    find_alt.group(0),
                    f"\n|image-generated| prompt for this image: {alt_text} |image-generated|\n",
                )
            clean_msg = re.sub(r"<img.*?>", "", assistant_msg)
            # # Remove other HTML formatting
            clean_msg = re.sub(r"<.*?>.*?</.*?>", "", clean_msg)
            clean_msg = re.sub(r"<.*?>", "", clean_msg)
            messages.append({"role": "assistant", "content": clean_msg})
        else:
            messages.append({"role": "assistant", "content": assistant_msg})

    # Add the current message and insert system message
    messages.append({"role": "user", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    objprint(messages)

    # Generate response with the specified parameters
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        stream=True,
    )

    accumulated_text = ""
    for resp in response:
        if "content" not in resp["choices"][0]["delta"]:
            continue

        chunk = resp["choices"][0]["delta"]["content"]
        accumulated_text += chunk
        yield accumulated_text

        # Check for images in the text as it streams
        if not image_generation_enabled:
            continue

        # Look for complete draw commands
        pattern = r"\|start-draw\|(.*?)\|end-draw\|"
        for m in re.finditer(pattern, accumulated_text, re.DOTALL):
            draw_cmd = m.group(0)
            draw_content = m.group(1).strip()
            if draw_cmd not in accumulated_text:
                continue

            try:
                draw_data = json.loads(draw_content)
            except json.JSONDecodeError:
                # Not valid JSON yet, keep accumulating
                pass
            except Exception as e:
                # Log error but continue
                print(f"Error processing draw command: {e}")

            img_count = len(draw_data) if isinstance(draw_data, list) else 1

            yield accumulated_text.replace(
                draw_cmd, f"Generating {img_count} images..."
            )

            html_images = ""
            if isinstance(draw_data, list):
                for idx, art_piece in enumerate(draw_data):
                    html_images += generate_img(art_piece)
                    yield accumulated_text.replace(
                        draw_cmd,
                        html_images + f"\n\nGenerating {img_count-idx-1} images...",
                    )
            else:
                html_images += generate_img(draw_data)

            # Replace the draw command with the HTML in the accumulated text
            accumulated_text = accumulated_text.replace(draw_cmd, html_images)
            yield accumulated_text


# Event handlers
def process_message(
    message,
    history,
    system_prompt,
    model,
    enable_imgs,
    temp,
    tokens,
    p,
    min_p,
    k,
    penalty,
):
    if not model:
        yield "", history, {"error": "Please select a model first"}
        return

    try:
        for response in generate_response(
            message,
            history,
            system_prompt,
            model,
            temp,
            tokens,
            p,
            min_p,
            k,
            penalty,
            enable_imgs,
        ):
            new_history = history + [[message, response]]

            # Debug info
            msg_count = len(new_history)
            debug = {
                "messages_count": msg_count,
                "model": model,
                "chat_format": LLM_CHAT_FORMAT,
                "settings": {
                    "temperature": temp,
                    "max_tokens": tokens,
                    "top_p": p,
                    "min_p": min_p,
                    "top_k": k,
                    "repeat_penalty": penalty,
                },
            }

            yield "", new_history, debug
    except Exception as e:
        yield "", history, {"error": str(e)}


# Main UI function
def create_ui():
    with gr.Blocks(title="TIPO-Agent") as demo:

        with gr.Row():
            # Left column: Chat interface
            with gr.Column(scale=7):
                gr.Markdown("# TIPO-Agent: Chat and Draw !")
                chatbot = gr.Chatbot(height=1680, render=True)

                msg = gr.Textbox(
                    placeholder="Type your message here...", show_label=False
                )

                clear = gr.Button("Clear Chat")

            # Right column: Settings
            with gr.Column(scale=3):
                gr.Markdown("## Model Settings")

                model_dropdown = gr.Dropdown(
                    choices=get_models()[::-1],
                    label="Select Model",
                    info="Place your GGUF models in the 'models' directory",
                )

                system_prompt = gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT,
                    label="System Prompt",
                    lines=3,
                    info="Instructions for the assistant's behavior",
                )

                enable_images = gr.Checkbox(
                    value=True,
                    label="Enable Image Generation",
                    info="Process |start-draw| commands to generate images",
                )

                with gr.Accordion("Generation Parameters", open=True):
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                    )
                    max_tokens = gr.Slider(
                        minimum=16,
                        maximum=4096,
                        value=2048,
                        step=16,
                        label="Max Tokens",
                    )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Top-p"
                    )
                    min_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.1, step=0.05, label="Min-p"
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=512, value=128, step=1, label="Top-k"
                    )
                    repeat_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.00,
                        step=0.05,
                        label="Repetition Penalty",
                    )

                refresh_models = gr.Button("Refresh Model List")

                with gr.Accordion("Debug Info", open=True):
                    debug_info = gr.JSON(value={})

        msg.submit(
            fn=process_message,
            inputs=[
                msg,
                chatbot,
                system_prompt,
                model_dropdown,
                enable_images,
                temperature,
                max_tokens,
                top_p,
                min_p,
                top_k,
                repeat_penalty,
            ],
            outputs=[msg, chatbot, debug_info],
        )

        clear.click(lambda: [], outputs=chatbot)
        refresh_models.click(fn=get_models, outputs=model_dropdown)

    return demo


# Run the app
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name=SERVER_HOST, server_port=SERVER_PORT)

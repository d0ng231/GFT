import gradio as gr
import torch
import time
from PIL import Image
from unsloth import FastVisionModel

def load_local_model(model_path):
    """
    Load the vision model and its tokenizer from a local checkpoint.
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

def run_inference(model, tokenizer, image: Image.Image, prompt: str, max_retries: int = 5):
    """
    Run inference using both image and text prompt.
    Attempts up to max_retries times and returns the raw output (full string).
    """
    for attempt in range(max_retries):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        }]
        inputs = tokenizer(
            image,
            tokenizer.apply_chat_template(messages, add_generation_prompt=True),
            add_special_tokens=False,
            return_tensors="pt"
        ).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.5,
            do_sample=True
        )
        raw_output = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        return raw_output
    return ""

def run_text_inference(model, tokenizer, prompt: str, max_retries: int = 5):
    """
    Run inference using text prompt only.
    Attempts up to max_retries times and returns the raw output (full string).
    """
    for attempt in range(max_retries):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
        }]
        inputs = tokenizer(
            prompt,
            tokenizer.apply_chat_template(messages, add_generation_prompt=True),
            add_special_tokens=False,
            return_tensors="pt"
        ).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.3,
            do_sample=True
        )
        raw_output = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        return raw_output
    return ""

# Load the model once at startup
MODEL_PATH = "/home/cl2733/unsloth/checkpoints/llama_3.2_11b_split_2_128_stage2"
model, tokenizer = load_local_model(MODEL_PATH)

def inference_interface(image, prompt, history):
    """
    This function now returns a generator:
      1) Immediately shows user message
      2) Generates response text from model
      3) Streams out the response character-by-character.
    """
    # 1) Show user's message immediately
    user_message = str(prompt)
    history.append({"role": "user", "content": user_message})
    # Yield so the user's message appears
    yield history, history

    # 2) Generate the full assistant response (no partial model streaming here)
    if image is not None:
        full_message = run_inference(model, tokenizer, image, prompt)
    else:
        full_message = run_text_inference(model, tokenizer, prompt)

    # 3) Append an empty assistant message
    history.append({"role": "assistant", "content": ""})
    current_text = ""

    # 4) Typewriter effect: yield partial response character-by-character
    for char in full_message:
        current_text += char
        history[-1]["content"] = current_text
        # Each yield updates the Chatbot live in the UI
        yield history, history
        time.sleep(0.02)  # adjust delay as desired (e.g., 0.01 or 0.05)

with gr.Blocks() as demo:
    gr.Markdown("# GFT-Llama 3.2 11B")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="OCTA image", type="pil")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat Log", type="messages")
    prompt_input = gr.Textbox(label="Enter your prompt")
    send_button = gr.Button("Send")
    state = gr.State([])

    # Note: Must enable queue for streaming yields!
    send_button.click(
        fn=inference_interface,
        inputs=[image_input, prompt_input, state],
        outputs=[chatbot, state],
        queue=True  # <-- enabling streaming
    )

# Also call queue() on the demo itself if needed
demo.queue()
demo.launch(share=True)

import argparse, gradio as gr
from unsloth import FastVisionModel
from unsloth import FastVisionModel as FVM
from PIL import Image

def load_model(p):
    m, t = FastVisionModel.from_pretrained(p, load_in_4bit=False)
    FVM.for_inference(m)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    m = m.to(device)
    return m, t

def reply(model, tok, image, prompt):
    if image is None or not prompt: return ""
    msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
    device = next(model.parameters()).device
    inputs = tok(
        image,
        tok.apply_chat_template(msgs, add_generation_prompt=True),
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    out = model.generate(**inputs, max_new_tokens=512, temperature=0.3)
    return tok.decode(out[0], skip_special_tokens=True)

def main(args):
    m, t = load_model(args.model)
    with gr.Blocks() as demo:
        gr.Markdown("# GFT-VLM Demo")
        img = gr.Image(type="pil", label="OCTA")
        txt = gr.Textbox(label="Prompt", value="Stage this image and explain briefly.")
        chat = gr.Chatbot()
        btn = gr.Button("Generate")
        def _run(i, p, history):
            resp = reply(m, t, i, p)
            history = history + [(p, resp)]
            return history
        btn.click(_run, [img, txt, chat], [chat])
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    main(ap.parse_args())

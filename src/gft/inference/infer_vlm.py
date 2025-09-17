import argparse
from unsloth import FastVisionModel
from unsloth import FastVisionModel as FVM
from PIL import Image

def run(args):
    m, t = FastVisionModel.from_pretrained(args.model, load_in_4bit=False)
    FVM.for_inference(m)
    img = Image.open(args.image).convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":args.prompt}]}]
    inputs = t(img, t.apply_chat_template(msgs, add_generation_prompt=True), add_special_tokens=False, return_tensors="pt").to("cuda")
    out = m.generate(**inputs, max_new_tokens=512, temperature=0.2)
    print(t.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Stage this image and explain briefly.")
    run(ap.parse_args())

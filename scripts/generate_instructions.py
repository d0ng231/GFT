
import argparse, json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

def make_pairs(img_path, stage_label):
    img = Image.open(img_path).convert("L").resize((304,304))
    arr = np.array(img)/255.0
    q1 = "What is the DR stage? Keep it to one line then a short reason."
    stages = ["Healthy","PDR","NPDR"]
    s = stages[stage_label] if 0 <= stage_label < len(stages) else "Healthy"
    r = f"{s}. Mean={arr.mean():.3f}, Var={arr.var():.3f}."
    qa1 = {
        "messages":[
            {"role":"user","content":[{"type":"image"},{"type":"text","text":q1}]},
            {"role":"assistant","content":[{"type":"text","text":r}]}
        ]
    }
    q2 = "Which quadrant looks most abnormal? Explain in one sentence."
    quad_means = [arr[:152,:152].mean(), arr[:152,152:].mean(), arr[152:,:152].mean(), arr[152:,152:].mean()]
    k = int(np.argmax(np.abs(np.array(quad_means)-arr.mean())))
    quad = ["UL","UR","LL","LR"][k]
    qa2 = {
        "messages":[
            {"role":"user","content":[{"type":"image"},{"type":"text","text":q2}]},
            {"role":"assistant","content":[{"type":"text","text":f"{quad}. Contrast vs global mean."}]}
        ]
    }
    return [qa1, qa2]

def main(args):
    root = Path(args.data_dir)
    csv = root / "train.csv"
    items = []
    if csv.exists():
        df = pd.read_csv(csv)
        items = [(Path(p), int(l)) for p,l in zip(df["path"], df["label"])]
    else:
        items = [(p, 0) for p in root.glob("*.png")]
    out = []
    for p, y in items:
        out.extend(make_pairs(p, y))
    Path(args.out).write_text("\\n".join(json.dumps(x) for x in out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="examples/sample_data")
    ap.add_argument("--out", type=str, default="instructions.jsonl")
    main(ap.parse_args())

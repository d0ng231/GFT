from pathlib import Path
from PIL import Image
import torch, pandas as pd, numpy as np

class OCTADataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train"):
        root = Path(root)
        csv = root / f"{split}.csv"
        if not csv.exists():
            imgs = sorted((root).glob("*.png"))
            if not imgs:
                (root / "toy.png").parent.mkdir(parents=True, exist_ok=True)
            if not imgs:
                img = Image.new("L", (304, 304), 0)
                for y in range(0, 304, 16):
                    for x in range(0, 304, 16):
                        val = int((x+y) % 255)
                        img.putpixel((x, y), val)
                img.save(root / "toy.png")
                imgs = [root / "toy.png"]
            import pandas as pd
            df = pd.DataFrame({ "path": [str(p) for p in imgs], "label": [0]*len(imgs) })
            df.to_csv(csv, index=False)
        self.df = pd.read_csv(csv)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        p = Path(self.df.loc[idx, "path"])
        y = int(self.df.loc[idx, "label"])
        x = Image.open(p).convert("L").resize((304, 304))
        x = torch.from_numpy(np.array(x)).float().unsqueeze(0) / 255.0
        return x, y, str(p)

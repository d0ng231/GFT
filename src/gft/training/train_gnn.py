import argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from gft.data.dataset import OCTADataset
from gft.graphs.build_graph import image_to_tiles
from gft.models.gnn import SAGE
from gft.utils.seed import fix
from gft.utils.metrics import classify_metrics
from tqdm import tqdm

def run(args):
    fix(0)
    ds = OCTADataset(args.data_dir, split="train")
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    dl_eval = DataLoader(ds, batch_size=1, shuffle=False)
    model = SAGE(in_ch=3, hid=64, classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        for x, y, _ in tqdm(dl, disable=not args.verbose):
            x = x[0]
            gx, pos, edge_index = image_to_tiles(x, tiles=8)
            logit = model(gx, edge_index)
            loss = F.cross_entropy(logit.unsqueeze(0), torch.tensor([y]))
            opt.zero_grad(); loss.backward(); opt.step()
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _ in dl_eval:
            x = x[0]
            gx, pos, edge_index = image_to_tiles(x, tiles=8)
            p = model(gx, edge_index).argmax().item()
            ys.append(int(y)); ps.append(p)
    print(classify_metrics(ys, ps))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="examples/sample_data")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args)

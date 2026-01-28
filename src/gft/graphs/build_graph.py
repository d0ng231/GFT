import torch
import numpy as np

def image_to_tiles(x, tiles=8):
    # x: (1, H, W) tensor in [0,1]
    _, H, W = x.shape
    th, tw = H // tiles, W // tiles
    feats, coords, edges = [], [], []
    nid = 0
    ids = np.zeros((tiles, tiles), dtype=int)
    for i in range(tiles):
        for j in range(tiles):
            tile = x[:, i*th:(i+1)*th, j*tw:(j+1)*tw]
            m = tile.mean().item()
            v = tile.var(unbiased=False).item()
            a = float(tile.numel())/ (H*W)
            feats.append([m, v, a])
            coords.append([i+0.5, j+0.5])
            ids[i, j] = nid
            nid += 1
    for i in range(tiles):
        for j in range(tiles):
            u = ids[i,j]
            if i+1<tiles: edges.append((u, ids[i+1,j]))
            if j+1<tiles: edges.append((u, ids[i,j+1]))
    if edges:
        edges = edges + [(v, u) for (u, v) in edges]
    for i in range(tiles * tiles):
        edges.append((i, i))
    x = torch.tensor(feats).float()            
    pos = torch.tensor(coords).float()         
    edge_index = torch.tensor(edges).long().t().contiguous() if edges else torch.zeros(2,0).long()
    return x, pos, edge_index

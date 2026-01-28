import torch, torch.nn as nn, torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self, in_ch, hid, classes):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, hid)
        self.lin2 = nn.Linear(hid, hid)
        self.out = nn.Linear(hid, classes)
    def agg(self, x, edge_index):
        if edge_index.numel()==0: return torch.zeros_like(x)
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device).index_add_(
            0, dst, torch.ones(dst.size(0), device=x.device, dtype=x.dtype)
        )
        deg = deg.clamp_min_(1).unsqueeze(1)
        return agg/deg
    def forward(self, x, edge_index):
        h = F.relu(self.lin1(x) + self.agg(x, edge_index))
        h = F.relu(self.lin2(h) + self.agg(h, edge_index))
        g = torch.cat([h.mean(0, keepdim=True), h.max(0, keepdim=True).values], dim=1)
        g = self.out(g)
        return g.squeeze(0)

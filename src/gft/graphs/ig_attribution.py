import torch

def integrated_gradients(model, x, edge_index, baseline=None, target=0, steps=20):
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
    scaled = [baseline + (float(i)/steps)*(x-baseline) for i in range(steps+1)]
    grads = []
    for s in scaled:
        s = s.requires_grad_(True)
        out = model(s, edge_index)
        logit = out[target]
        g, = torch.autograd.grad(logit, s, retain_graph=False, create_graph=False)
        grads.append(g)
    grads = torch.stack(grads, 0)         
    avg = (grads[:-1] + grads[1:]) / 2.0  
    ig = (x - baseline) * avg.mean(0)
    return ig.detach()

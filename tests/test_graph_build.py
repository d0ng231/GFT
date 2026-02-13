from gft.graphs.build_graph import image_to_tiles
import pytest
import torch


def test_tiles():
    x = torch.rand(1, 304, 304)
    gx, pos, edge_index = image_to_tiles(x, tiles=8)
    assert gx.shape[0] == 64 and gx.shape[1] == 3
    assert pos.shape == (64, 2)
    assert edge_index.shape[0] == 2


def test_tiles_requires_divisible_shape():
    x = torch.rand(1, 305, 304)
    with pytest.raises(ValueError, match="divisible"):
        image_to_tiles(x, tiles=8)


def test_tiles_requires_positive_count():
    x = torch.rand(1, 304, 304)
    with pytest.raises(ValueError, match="positive"):
        image_to_tiles(x, tiles=0)

from types import SimpleNamespace

import pytest
import torch

from utils.data_sampler import DataSampler


def test_sample_pin_uses_per_point_origins_for_rays_and_weights(monkeypatch):
    config = SimpleNamespace(
        device="cpu",
        surface_sample_range_m=0.25,
        surface_sample_n=1,
        free_behind_n=0,
        free_front_n=0,
        free_sample_begin_ratio=0.5,
        free_sample_end_dist_m=1.0,
        dist_weight_on=True,
        dist_weight_scale=0.8,
        max_range=10.0,
        behind_dropoff_on=False,
    )
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: torch.ones(*args))
    points = torch.tensor([[10.0, 2.0, 0.0], [1.0, 12.0, 0.0]])
    origins = torch.tensor([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]])

    coordinates, _, _, _, _, weights = DataSampler(config).sample_pin(
        points, origins, None, None, None
    )

    torch.testing.assert_close(coordinates[0], points[0])
    torch.testing.assert_close(coordinates[2], points[1])
    assert coordinates[1, 1].item() == pytest.approx(2.0)
    assert coordinates[3, 0].item() == pytest.approx(1.0)
    torch.testing.assert_close(weights, torch.full((4,), 0.6))

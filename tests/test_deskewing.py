import torch

from utils.tools import deskewing


def test_deskewing_returns_unchanged_points_for_zero_timestamp_span():
    points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    timestamps = torch.full((2, 1), 0.25)
    pose = torch.eye(4)
    pose[0, 3] = 2.0

    result = deskewing(points, timestamps, pose)

    torch.testing.assert_close(result, points)
    assert torch.isfinite(result).all()


def test_valid_timestamp_subsets_keep_original_scan_phase():
    points = torch.zeros((4, 3))
    timestamps = torch.tensor([[0.0], [0.25], [0.75], [1.0]])
    pose = torch.eye(4)
    pose[0, 3] = 2.0

    full = deskewing(points.clone(), timestamps, pose, normalize_ts=False)
    subset = deskewing(
        torch.zeros((2, 3)), timestamps[1:3], pose, normalize_ts=False
    )

    torch.testing.assert_close(subset, full[1:3])

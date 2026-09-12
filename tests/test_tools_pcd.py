import numpy as np
import pytest

from utils.tools import write_binary_xyz_pcd


def test_write_binary_xyz_pcd(tmp_path):
    points = np.array([[1.0, 2.0, 3.0], [-4.5, 0.25, 6.0]], dtype=np.float64)
    path = tmp_path / "nested" / "cloud.pcd"

    write_binary_xyz_pcd(path, points)

    contents = path.read_bytes()
    header, payload = contents.split(b"DATA binary\n", 1)
    assert b"FIELDS x y z\n" in header
    assert b"WIDTH 2\n" in header
    assert b"POINTS 2\n" in header
    np.testing.assert_array_equal(
        np.frombuffer(payload, dtype="<f4").reshape(-1, 3), points.astype(np.float32)
    )


def test_write_binary_xyz_pcd_rejects_non_xyz(tmp_path):
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        write_binary_xyz_pcd(tmp_path / "cloud.pcd", np.zeros((2, 4)))

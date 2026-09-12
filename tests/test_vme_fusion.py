from types import SimpleNamespace

import torch

from utils.error_state_iekf import (
    IEKFOM,
    InputIkfom,
    STATE_DIM,
    StateIkfom,
    _shape_lidar_normal_equations,
    boxplus,
)
from utils.so3_math import so3Exp, vec2skew


def make_filter():
    tracker = IEKFOM.__new__(IEKFOM)
    tracker.tran_dtype = torch.float64
    tracker.config = SimpleNamespace(
        vme_vel_noise=0.05,
        vme_nhc_noise=0.05,
        vme_yawrate_on=False,
        vme_yawrate_noise=0.05,
        vme_scale_on=True,
        vme_scale_random_walk=1e-4,
        vme_scale_min=0.95,
        vme_scale_max=1.05,
        vme_scale_min_speed_mps=0.1,
        vme_scale_min_lidar_weight=0.5,
        vme_scale_settle_frames=3,
        vme_innovation_inflate_frames=3,
        vme_velocity_recovery_sigma_mps=0.15,
        imu_noise_density_on=False,
        vel_process_noise_mps_per_sqrt_s=0.15,
    )
    tracker.x = StateIkfom(torch.float64)
    tracker.P = torch.eye(STATE_DIM, dtype=torch.float64)
    tracker.latest_gyro = None
    tracker.latest_vme_speed = None
    tracker.previous_lidar_longitudinal_weight = 0.0
    tracker.vme_consistent_frames = 0
    tracker.vme_inconsistent_frames = 0
    tracker.vme_scale_adaptation_allowed = False
    tracker.vme_diagnostics = tracker._empty_vme_diagnostics()
    tracker.Q = torch.zeros((12, 12), dtype=torch.float64)
    return tracker


def test_body_velocity_rotation_jacobian_matches_finite_difference():
    state = StateIkfom(torch.float64)
    state.rot = so3Exp(torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64))
    state.vel = torch.tensor([3.0, -0.4, 0.7], dtype=torch.float64)
    expected = vec2skew(state.rot.T @ state.vel)
    actual = torch.empty((3, 3), dtype=torch.float64)
    epsilon = 1e-7
    for axis in range(3):
        delta = torch.zeros(STATE_DIM, dtype=torch.float64)
        delta[axis] = epsilon
        plus = boxplus(state, delta).rot.T @ state.vel
        delta[axis] = -epsilon
        minus = boxplus(state, delta).rot.T @ state.vel
        actual[:, axis] = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-7)


def test_vme_update_converges_body_velocity_and_preserves_position():
    tracker = make_filter()
    tracker.x.vel = torch.tensor([4.0, 1.0, -0.5], dtype=torch.float64)
    original_position = tracker.x.pos.clone()
    original_error = torch.linalg.vector_norm(
        tracker.x.vel - torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
    )
    tracker.update_vme(2.0)
    body_velocity = tracker.x.rot.T @ tracker.x.vel
    new_error = torch.linalg.vector_norm(
        body_velocity - torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert new_error < original_error
    torch.testing.assert_close(tracker.x.pos, original_position)
    torch.testing.assert_close(tracker.P, tracker.P.T)
    assert torch.linalg.eigvalsh(tracker.P).min() >= -1e-12


def test_vme_scale_freezes_for_inconsistent_nis_then_adapts_after_settling():
    tracker = make_filter()
    tracker.previous_lidar_longitudinal_weight = 1.0
    tracker.P[6:9, 6:9] *= 1e-4
    tracker.P[18, 18] = 0.02**2
    tracker.P[6, 18] = tracker.P[18, 6] = 1e-4
    tracker.x.vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

    tracker.update_vme(2.0)
    assert tracker.vme_diagnostics["vme_scale_frozen"]
    torch.testing.assert_close(tracker.x.vme_scale, torch.tensor(1.0, dtype=torch.float64))

    tracker.x.vel[0] = 1.0
    for _ in range(tracker.config.vme_scale_settle_frames - 1):
        tracker.update_vme(1.01)
        assert tracker.vme_diagnostics["vme_scale_frozen"]
    tracker.update_vme(1.01)

    assert tracker.x.vme_scale < 1.0


def test_information_cap_bounds_large_eigenvalues_and_shapes_gradient():
    H = torch.zeros((2, STATE_DIM), dtype=torch.float64)
    H[0, 3] = 0.1
    H[1, 4] = 1e6
    residual = torch.tensor([1.0, 0.0], dtype=torch.float64)
    shaped_S, shaped_b, eigenvalues, _, weights = _shape_lidar_normal_equations(
        H, torch.ones(2, dtype=torch.float64), residual, 10.0, 0.02
    )
    raw_S, raw_b, _, _, _ = _shape_lidar_normal_equations(
        H, torch.ones(2, dtype=torch.float64), residual, 10.0, 0.02, False
    )
    prior_information = torch.eye(STATE_DIM, dtype=torch.float64)
    shaped_correction = torch.linalg.solve(
        shaped_S + prior_information, shaped_b
    )
    raw_correction = torch.linalg.solve(raw_S + prior_information, raw_b)

    lambda_ref = 1.0 / 0.02**2
    torch.testing.assert_close(weights, lambda_ref / (eigenvalues + lambda_ref))
    assert torch.linalg.eigvalsh(shaped_S[:6, :6]).max() <= lambda_ref + 1e-8
    assert eigenvalues[0] < eigenvalues[-1]
    assert abs(shaped_correction[3]) <= abs(raw_correction[3])


def test_information_cap_keeps_low_isotropic_information_nearly_unchanged():
    H = torch.zeros((6, STATE_DIM))
    H[:, :6] = torch.diag(torch.tensor([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))
    _, _, _, _, weights = _shape_lidar_normal_equations(
        H, torch.ones(6), torch.ones(6), 10.0, 0.02
    )

    assert torch.all(weights > 0.99)


def test_disabled_degeneracy_shaping_equals_raw_normal_equations():
    H = torch.randn(20, STATE_DIM, dtype=torch.float64)
    H[:, 6:] = 0.0
    R_inv = torch.rand(20, dtype=torch.float64)
    residual = torch.randn(20, dtype=torch.float64)

    information, gradient, _, _, weights = _shape_lidar_normal_equations(
        H, R_inv, residual, 10.0, 0.02, False
    )

    H_T_R_inv = H.T * R_inv
    torch.testing.assert_close(information, H_T_R_inv @ H)
    torch.testing.assert_close(gradient, H_T_R_inv @ residual)
    torch.testing.assert_close(weights, torch.ones(6, dtype=torch.float64))


def test_scale_disabled_leaves_scale_at_one():
    tracker = make_filter()
    tracker.config.vme_scale_on = False
    tracker.x.vel[0] = 1.0

    tracker.update_vme(1.01)

    torch.testing.assert_close(
        tracker.x.vme_scale, torch.tensor(1.0, dtype=torch.float64)
    )


def test_no_vme_does_not_change_filter_state():
    tracker = make_filter()
    state_before = tracker.x.vel.clone()
    covariance_before = tracker.P.clone()
    # The caller deliberately performs no update for an absent/invalid VME sample.
    torch.testing.assert_close(tracker.x.vel, state_before)
    torch.testing.assert_close(tracker.P, covariance_before)


def test_lidar_posterior_covariance_is_symmetric_positive():
    prior = torch.eye(STATE_DIM, dtype=torch.float64)
    H = torch.randn(40, STATE_DIM, dtype=torch.float64)
    H[:, 6:] = 0.0
    information, _, _, _, _ = _shape_lidar_normal_equations(
        H,
        torch.ones(40, dtype=torch.float64),
        torch.randn(40, dtype=torch.float64),
        10.0,
        0.02,
    )
    posterior = torch.linalg.inv(information + torch.linalg.inv(prior))
    posterior = 0.5 * (posterior + posterior.T)

    torch.testing.assert_close(posterior, posterior.T)
    assert torch.linalg.eigvalsh(posterior).min() > 0.0


def test_velocity_recovers_after_repeated_stationary_lidar_overconfidence():
    tracker = make_filter()
    tracker.config.vme_scale_on = False
    inflation_was_active = False
    for _ in range(8):
        # Emulate a stationary LiDAR posterior claiming enormous information.
        tracker.P[6, 6] = 1e-12
        tracker.update_vme(0.8)
        inflation_was_active |= tracker.vme_diagnostics[
            "covariance_inflation_active"
        ]

    assert tracker.x.vel[0] > 0.7
    assert inflation_was_active


def test_velocity_covariance_inflation_preserves_joint_psd():
    tracker = make_filter()
    tracker.config.vme_scale_on = False
    tracker.config.vme_innovation_inflate_frames = 1
    direction = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
    tracker.x.rot = so3Exp(direction)
    factor = torch.randn(STATE_DIM, 6, dtype=torch.float64)
    tracker.P = factor @ factor.T + 1e-9 * torch.eye(STATE_DIM, dtype=torch.float64)
    tracker.P *= 1e-5
    tracker.x.vel.zero_()

    tracker.update_vme(1.0)

    torch.testing.assert_close(tracker.P, tracker.P.T)
    assert torch.linalg.eigvalsh(tracker.P).min() >= -1e-12


def test_yaw_rate_model_and_jacobian_use_world_vertical():
    tracker = make_filter()
    tracker.config.vme_yawrate_on = True
    tracker.config.vme_scale_on = False
    tracker.x.rot = so3Exp(torch.tensor([0.4, -0.3, 0.2], dtype=torch.float64))
    tracker.x.bg = torch.tensor([0.03, -0.02, 0.01], dtype=torch.float64)
    tracker.latest_gyro = torch.tensor([0.7, -0.4, 1.2], dtype=torch.float64)
    tracker.P *= 1e-8
    prior = tracker.x
    omega = tracker.latest_gyro - prior.bg
    expected_rate = prior.rot[2] @ omega
    expected_H = torch.zeros(STATE_DIM, dtype=torch.float64)
    expected_H[:3] = -prior.rot[2] @ vec2skew(omega)
    expected_H[9:12] = -prior.rot[2]
    epsilon = 1e-7
    finite_difference = torch.zeros(STATE_DIM, dtype=torch.float64)
    for index in list(range(3)) + list(range(9, 12)):
        delta = torch.zeros(STATE_DIM, dtype=torch.float64)
        delta[index] = epsilon
        plus = boxplus(prior, delta)
        plus_rate = plus.rot[2] @ (tracker.latest_gyro - plus.bg)
        delta[index] = -epsilon
        minus = boxplus(prior, delta)
        minus_rate = minus.rot[2] @ (tracker.latest_gyro - minus.bg)
        finite_difference[index] = (plus_rate - minus_rate) / (2.0 * epsilon)

    torch.testing.assert_close(finite_difference, expected_H, atol=1e-9, rtol=1e-7)
    tracker.update_vme(0.0, float(expected_rate))
    torch.testing.assert_close(tracker.x.rot, prior.rot, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(tracker.x.bg, prior.bg, atol=1e-10, rtol=1e-10)


def test_process_noise_density_has_one_second_velocity_covariance_growth():
    tracker = make_filter()
    tracker.config.imu_noise_density_on = True
    tracker.P.zero_()
    imu = InputIkfom(
        torch.float64,
        [0.0, 0.0, 9.81],
        [0.0, 0.0, 0.0],
    )
    for _ in range(100):
        tracker.predict(imu, 0.01)

    expected = tracker.config.vel_process_noise_mps_per_sqrt_s**2
    torch.testing.assert_close(
        torch.diag(tracker.P[6:9, 6:9]),
        torch.full((3,), expected, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

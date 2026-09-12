#!/usr/bin/env python3
# @file      error_state_iekf.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved

import math
import torch
import numpy as np
from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.so3_math import vec2skew, so3Exp, SO3Log, batch_vec2skew
from utils.tools import get_gradient, transform_torch

G_m_s2 = 9.81  # 定义全局重力加速度
STATE_DIM = 19


def _shape_lidar_normal_equations(
    H, R_inv, residual, lever_arm_m, position_sigma_m, enabled=True
):
    """Build consistently weighted LiDAR information and gradient."""
    H_T_R_inv = H.T * R_inv
    information = H_T_R_inv @ H
    gradient = H_T_R_inv @ residual
    pose_scale = torch.ones(6, dtype=H.dtype, device=H.device)
    pose_scale[:3] = 1.0 / lever_arm_m
    normalized_information = (
        pose_scale.unsqueeze(1) * information[:6, :6] * pose_scale.unsqueeze(0)
    )
    normalized_information = 0.5 * (
        normalized_information + normalized_information.T
    )
    information_is_finite = torch.isfinite(normalized_information).all()
    if information_is_finite:
        eigenvalues, eigenvectors = torch.linalg.eigh(normalized_information)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
    else:
        eigenvalues = torch.zeros(6, dtype=H.dtype, device=H.device)
        eigenvectors = torch.eye(6, dtype=H.dtype, device=H.device)

    if enabled and eigenvalues.numel() and torch.isfinite(eigenvalues).all():
        lambda_ref = 1.0 / position_sigma_m**2
        weights = lambda_ref / (eigenvalues + lambda_ref)
    elif enabled:
        weights = torch.zeros_like(eigenvalues)
    else:
        weights = torch.ones_like(eigenvalues)

    if not enabled:
        return information, gradient, eigenvalues, eigenvectors, weights

    shaped_information = torch.zeros_like(information)
    shaped_gradient = torch.zeros_like(gradient)
    if information_is_finite and torch.isfinite(gradient[:6]).all():
        inverse_pose_scale = 1.0 / pose_scale
        normalized_gradient = pose_scale * gradient[:6]
        information_filter = (
            eigenvectors * torch.sqrt(weights).unsqueeze(0)
        ) @ eigenvectors.T
        gradient_filter = (eigenvectors * weights.unsqueeze(0)) @ eigenvectors.T
        shaped_information[:6, :6] = (
            inverse_pose_scale.unsqueeze(1)
            * (information_filter @ normalized_information @ information_filter)
            * inverse_pose_scale.unsqueeze(0)
        )
        shaped_gradient[:6] = inverse_pose_scale * (
            gradient_filter @ normalized_gradient
        )
    return (
        shaped_information,
        shaped_gradient,
        eigenvalues,
        eigenvectors,
        weights,
    )


class StateIkfom:
    """19D state: rotation, position, velocity, IMU biases, gravity, VME scale."""

    def __init__(
        self,
        dtype,
        pos=None,
        rot=None,
        vel=None,
        bg=None,
        ba=None,
        grav=None,
        vme_scale=None,
    ):
        self.dtype = dtype
        self.rot = torch.eye(3, dtype=self.dtype) if rot is None else rot
        self.pos = torch.zeros(3, dtype=self.dtype) if pos is None else pos
        self.vel = torch.zeros(3, dtype=self.dtype) if vel is None else vel
        self.bg = torch.zeros(3, dtype=self.dtype) if bg is None else bg
        self.ba = torch.zeros(3, dtype=self.dtype) if ba is None else ba
        self.grav = (
            torch.tensor([0.0, 0.0, -G_m_s2], dtype=self.dtype)
            if grav is None
            else grav
        )
        self.vme_scale = (
            torch.tensor(1.0, dtype=self.dtype) if vme_scale is None else vme_scale
        )

    def cpu(self):
        """将所有张量转移到CPU"""
        self.rot = self.rot.cpu()
        self.pos = self.pos.cpu()
        self.vel = self.vel.cpu()
        self.bg = self.bg.cpu()
        self.ba = self.ba.cpu()
        self.grav = self.grav.cpu()
        self.vme_scale = self.vme_scale.cpu()

    def cuda(self):
        """将所有张量转移到GPU"""
        self.rot = self.rot.cuda()
        self.pos = self.pos.cuda()
        self.vel = self.vel.cuda()
        self.bg = self.bg.cuda()
        self.ba = self.ba.cuda()
        self.grav = self.grav.cuda()
        self.vme_scale = self.vme_scale.cuda()


class InputIkfom:
    """输入向量类定义，用于表示陀螺仪和加速度计的测量值。"""

    def __init__(self, dtype, acc: np.array, gyro: np.array):
        self.dtype = dtype
        self.acc = torch.tensor(acc, dtype=self.dtype)
        self.gyro = torch.tensor(gyro, dtype=self.dtype)


def boxplus(state: StateIkfom, delta: torch.tensor):
    """广义加法操作"""
    new_state = StateIkfom(state.dtype)
    new_state.rot = state.rot @ so3Exp(delta[0:3])
    new_state.pos = state.pos + delta[3:6]
    new_state.vel = state.vel + delta[6:9]
    new_state.bg = state.bg + delta[9:12]
    new_state.ba = state.ba + delta[12:15]
    new_state.grav = state.grav + delta[15:18]
    new_state.vme_scale = state.vme_scale + delta[18]
    return new_state


def boxminus(x1: StateIkfom, x2: StateIkfom):
    """广义减法操作，计算两个状态之间的差"""
    delta_rot = SO3Log(x2.rot.T @ x1.rot)
    delta_pos = x1.pos - x2.pos
    delta_vel = x1.vel - x2.vel
    delta_bg = x1.bg - x2.bg
    delta_ba = x1.ba - x2.ba
    delta_grav = x1.grav - x2.grav
    delta_vme_scale = (x1.vme_scale - x2.vme_scale).reshape(1)
    delta = torch.concatenate(
        [
            delta_rot,
            delta_pos,
            delta_vel,
            delta_bg,
            delta_ba,
            delta_grav,
            delta_vme_scale,
        ]
    )
    return delta


class IEKFOM:
    """迭代扩展卡尔曼滤波器类"""

    def __init__(
        self,
        config: Config,
        neural_points: NeuralPoints,
        geo_decoder: Decoder,
    ):
        self.config = config
        self.silence = config.silence
        self.neural_points = neural_points
        self.geo_decoder = geo_decoder
        self.device = self.config.device
        self.dtype = config.dtype
        self.tran_dtype = config.tran_dtype

        self.x = StateIkfom(self.tran_dtype)  # 初始化状态
        self.P = torch.eye(STATE_DIM, dtype=self.tran_dtype)  # 初始化状态协方差矩阵
        self.P[9:12, 9:12] = self.P[9:12, 9:12] * 1e-4  # 初始陀螺仪偏置协方差
        self.P[12:15, 12:15] = self.P[12:15, 12:15] * 1e-3  # 初始加速度计协方差
        self.P[15:18, 15:18] = self.P[15:18, 15:18] * 1e-4  # 初始重力协方差
        self.P[18, 18] = self.config.vme_scale_init_sigma**2
        self.Q = self.process_noise_covariance()  # 前向传播白噪声协方差
        self.R_inv = None  # 测量噪声协方差
        self.eps = 0.001  # 收敛阈值
        self.max_iteration = self.config.reg_iter_n  # 最大迭代轮数
        self.latest_gyro = None
        self.latest_vme_speed = None
        self.previous_lidar_longitudinal_weight = 0.0
        self.vme_consistent_frames = 0
        self.vme_inconsistent_frames = 0
        self.vme_scale_adaptation_allowed = False
        self.vme_diagnostics = self._empty_vme_diagnostics()
        self.last_registration_diagnostics = {}

    def _empty_vme_diagnostics(self):
        return {
            "vme_innovation_mps": float("nan"),
            "vme_innovation_sigma_mps": float("nan"),
            "vme_nis": float("nan"),
            "velocity_sigma_long_mps": float("nan"),
            "vme_scale_frozen": True,
            "covariance_inflation_active": False,
            "vme_scale_saturated": False,
        }

    def process_noise_covariance(self):
        """噪声协方差Q的初始化"""
        Q = torch.zeros((12, 12), dtype=self.config.tran_dtype)
        Q[:3, :3] = self.config.measurement_noise_covariance * torch.eye(3)
        Q[3:6, 3:6] = self.config.measurement_noise_covariance * torch.eye(3)
        Q[6:9, 6:9] = self.config.bias_noise_covariance * torch.eye(3)
        Q[9:12, 9:12] = self.config.bias_noise_covariance * torch.eye(3)
        return Q

    def df_dx(self, s: StateIkfom, in_: InputIkfom, dt: float):
        """计算状态转移函数的雅可比矩阵"""
        # omega_ = in_.gyro - s.bg
        acc_ = in_.acc - s.ba
        df_dx = torch.eye(STATE_DIM, dtype=self.tran_dtype)
        I_dt = torch.eye(3, dtype=self.tran_dtype) * dt
        # df_dx[0:3, 0:3] = so3Exp(-omega_ * dt)
        # so3Exp(-omega_ * dt) 可以近似为I
        df_dx[0:3, 0:3] = torch.eye(3, dtype=self.tran_dtype)
        df_dx[0:3, 9:12] = -I_dt
        df_dx[3:6, 6:9] = I_dt
        df_dx[6:9, 0:3] = -s.rot @ vec2skew(acc_) * dt
        df_dx[6:9, 12:15] = -s.rot * dt
        df_dx[6:9, 15:18] = I_dt

        return df_dx

    def df_dw(self, s: StateIkfom, in_: InputIkfom, dt: float):
        """计算过程噪声的雅可比矩阵"""
        # omega_ = in_.gyro - s.bg
        I = torch.eye(3, dtype=self.tran_dtype)
        cov = torch.zeros((STATE_DIM, 12), dtype=self.tran_dtype)
        # cov[0:3, 0:3] = -A_T(omega_ * dt)
        # -A(w dt)可以简化为-I
        cov[0:3, 0:3] = -I
        cov[6:9, 3:6] = -s.rot  # -R
        cov[9:12, 6:9] = I
        cov[12:15, 9:12] = I
        cov = cov * dt

        return cov

    def predict(self, i_in: InputIkfom, dt: float):
        """前向传播，在cpu上执行前向传播效率高得多"""
        self.latest_gyro = i_in.gyro.clone()
        f = self.f_model(self.x, i_in)
        df_dx = self.df_dx(self.x, i_in, dt)
        df_dw = self.df_dw(self.x, i_in, dt)

        self.x = boxplus(self.x, f * dt)
        process_covariance = df_dw @ self.Q @ df_dw.T
        if self.config.imu_noise_density_on:
            if dt > 0.0:
                process_covariance = process_covariance / dt
            velocity_variance = (
                self.config.vel_process_noise_mps_per_sqrt_s**2 * max(dt, 0.0)
            )
            process_covariance[6:9, 6:9] += velocity_variance * torch.eye(
                3, dtype=self.tran_dtype, device=self.P.device
            )
        self.P = df_dx @ self.P @ df_dx.T + process_covariance
        if self.config.vme_scale_on and self.vme_scale_adaptation_allowed:
            self.P[18, 18] += self.config.vme_scale_random_walk**2 * dt

    def mark_vme_missing(self):
        """Reset adaptation/recovery state when a frame has no usable VME."""
        self.latest_vme_speed = None
        self.vme_consistent_frames = 0
        self.vme_inconsistent_frames = 0
        self.vme_scale_adaptation_allowed = False
        self.vme_diagnostics = self._empty_vme_diagnostics()

    def update_vme(self, v_long_mps: float, yaw_rate_rps=None):
        """Fuse longitudinal body velocity and lateral/vertical NHC constraints."""
        self.latest_vme_speed = abs(v_long_mps)
        body_vel = self.x.rot.T @ self.x.vel
        H = torch.zeros((3, STATE_DIM), dtype=self.tran_dtype, device=self.P.device)
        # R_new = R Exp(dtheta), so d(R_new^T v)/dtheta = skew(R^T v).
        H[:, 0:3] = vec2skew(body_vel)
        H[:, 6:9] = self.x.rot.T
        scale = self.x.vme_scale if self.config.vme_scale_on else 1.0
        measurement = torch.zeros(3, dtype=self.tran_dtype, device=self.P.device)
        measurement[0] = scale * v_long_mps
        noise_variance = torch.tensor(
            [
                self.config.vme_vel_noise**2,
                self.config.vme_nhc_noise**2,
                self.config.vme_nhc_noise**2,
            ],
            dtype=self.tran_dtype,
            device=self.P.device,
        )

        # Evaluate consistency without allowing scale to absorb a velocity error.
        longitudinal_innovation = measurement[0] - body_vel[0]
        longitudinal_H = H[0]
        longitudinal_variance = (
            longitudinal_H @ self.P @ longitudinal_H
            + noise_variance[0]
        )
        nis_variance = longitudinal_variance
        longitudinal_nis = longitudinal_innovation.square() / longitudinal_variance
        consistent = float(longitudinal_nis) <= 3.841
        if consistent:
            self.vme_inconsistent_frames = 0
        else:
            self.vme_inconsistent_frames += 1

        covariance_inflation_active = (
            self.vme_inconsistent_frames
            >= self.config.vme_innovation_inflate_frames
        )
        if covariance_inflation_active:
            world_longitudinal = self.x.rot[:, 0]
            current_velocity_variance = (
                world_longitudinal @ self.P[6:9, 6:9] @ world_longitudinal
            )
            recovery_variance = self.config.vme_velocity_recovery_sigma_mps**2
            variance_increase = torch.clamp(
                recovery_variance - current_velocity_variance, min=0.0
            )
            inflation_vector = torch.zeros(
                STATE_DIM, dtype=self.tran_dtype, device=self.P.device
            )
            inflation_vector[6:9] = (
                torch.sqrt(variance_increase) * world_longitudinal
            )
            self.P += torch.outer(inflation_vector, inflation_vector)
            longitudinal_variance = (
                longitudinal_H @ self.P @ longitudinal_H
                + noise_variance[0]
            )

        scale_conditions_met = (
            self.latest_vme_speed >= self.config.vme_scale_min_speed_mps
            and consistent
            and self.previous_lidar_longitudinal_weight
            >= self.config.vme_scale_min_lidar_weight
        )
        if scale_conditions_met:
            self.vme_consistent_frames += 1
        else:
            self.vme_consistent_frames = 0
        self.vme_scale_adaptation_allowed = (
            self.config.vme_scale_on
            and self.vme_consistent_frames >= self.config.vme_scale_settle_frames
        )
        if self.vme_scale_adaptation_allowed:
            # h(x) = body_speed - scale * raw_speed and innovation = -h(x).
            H[0, 18] = -v_long_mps

        if (
            self.config.vme_yawrate_on
            and yaw_rate_rps is not None
            and self.latest_gyro is not None
        ):
            yaw_H = torch.zeros(
                (1, STATE_DIM), dtype=self.tran_dtype, device=self.P.device
            )
            angular_velocity = self.latest_gyro.to(self.P.device) - self.x.bg
            world_vertical = self.x.rot[2, :]
            yaw_H[0, 0:3] = -world_vertical @ vec2skew(angular_velocity)
            yaw_H[0, 9:12] = -world_vertical
            H = torch.cat((H, yaw_H), dim=0)
            predicted_yaw_rate = world_vertical @ angular_velocity
            measurement = torch.cat(
                (
                    measurement,
                    torch.as_tensor(
                        [yaw_rate_rps], dtype=self.tran_dtype, device=self.P.device
                    ),
                )
            )
            body_vel = torch.cat((body_vel, predicted_yaw_rate.reshape(1)))
            noise_variance = torch.cat(
                (
                    noise_variance,
                    torch.as_tensor(
                        [self.config.vme_yawrate_noise**2],
                        dtype=self.tran_dtype,
                        device=self.P.device,
                    ),
                )
            )

        innovation = measurement - body_vel
        measurement_cov = torch.diag(noise_variance)
        innovation_cov = H @ self.P @ H.T + measurement_cov
        kalman_gain = torch.linalg.solve(innovation_cov, H @ self.P).T
        if not self.vme_scale_adaptation_allowed:
            kalman_gain[18, :] = 0.0
        self.x = boxplus(self.x, kalman_gain @ innovation)
        scale_saturated = False
        if self.config.vme_scale_on:
            unclamped_scale = self.x.vme_scale.clone()
            self.x.vme_scale.clamp_(
                self.config.vme_scale_min, self.config.vme_scale_max
            )
            scale_saturated = bool(self.x.vme_scale != unclamped_scale)

        identity = torch.eye(STATE_DIM, dtype=self.tran_dtype, device=self.P.device)
        covariance_factor = identity - kalman_gain @ H
        self.P = (
            covariance_factor @ self.P @ covariance_factor.T
            + kalman_gain @ measurement_cov @ kalman_gain.T
        )
        self.P = 0.5 * (self.P + self.P.T)
        velocity_direction = self.x.rot[:, 0]
        velocity_variance = velocity_direction @ self.P[6:9, 6:9] @ velocity_direction
        self.vme_diagnostics = {
            "vme_innovation_mps": float(longitudinal_innovation),
            "vme_innovation_sigma_mps": float(
                torch.sqrt(torch.clamp(nis_variance, min=0.0))
            ),
            "vme_nis": float(longitudinal_nis),
            "velocity_sigma_long_mps": float(torch.sqrt(torch.clamp(velocity_variance, min=0.0))),
            "vme_scale_frozen": not self.vme_scale_adaptation_allowed,
            "covariance_inflation_active": covariance_inflation_active,
            "vme_scale_saturated": scale_saturated,
        }

    def f_model(self, s: StateIkfom, in_: InputIkfom):
        """获取运动方程，用于描述状态如何随时间演变"""
        res = torch.zeros(STATE_DIM, dtype=self.tran_dtype)
        a_inertial = s.rot @ (in_.acc - s.ba) + s.grav
        res[:3] = in_.gyro - s.bg
        res[3:6] = s.vel
        res[6:9] = a_inertial
        return res

    def h_model(self, pc_imu: torch.tensor):
        bs = self.config.infer_bs
        mask_min_nn_count = self.config.track_mask_query_nn_k
        min_grad_norm = self.config.reg_min_grad_norm
        max_grad_norm = self.config.reg_max_grad_norm

        T = torch.eye(4)
        T[:3, :3] = self.x.rot
        T[:3, 3] = self.x.pos

        pc_map = transform_torch(pc_imu, T)
        sample_count = pc_map.shape[0]
        iter_n = math.ceil(sample_count / bs)

        sdf_pred = torch.zeros(sample_count, device=pc_map.device)
        sdf_std = torch.zeros(sample_count, device=pc_map.device)
        mc_mask = torch.zeros(sample_count, device=pc_map.device, dtype=torch.bool)
        sdf_grad = torch.zeros((sample_count, 3), device=pc_map.device)
        certainty = torch.zeros(sample_count, device=pc_map.device)

        # 分批处理，计算点云的SDF预测值
        for n in range(iter_n):
            head = n * bs
            tail = min((n + 1) * bs, sample_count)
            batch_coord = pc_map[head:tail, :]
            batch_coord.requires_grad_(True)

            (
                batch_geo_feature,
                _,
                weight_knn,
                nn_count,
                batch_certainty,
            ) = self.neural_points.query_feature(
                batch_coord,
                training_mode=False,
                query_locally=True,
                query_color_feature=False,
            )  # inference mode

            batch_sdf = self.geo_decoder.sdf(batch_geo_feature)
            if not self.config.weighted_first:
                batch_sdf_mean = torch.sum(batch_sdf * weight_knn, dim=1)
                batch_sdf_var = torch.sum(
                    (weight_knn * (batch_sdf - batch_sdf_mean.unsqueeze(-1)) ** 2),
                    dim=1,
                )
                batch_sdf_std = torch.sqrt(torch.clamp(batch_sdf_var, min=0.0)).squeeze(1)
                batch_sdf = batch_sdf_mean.squeeze(1)
                sdf_std[head:tail] = batch_sdf_std.detach()

            batch_sdf_grad = get_gradient(batch_coord, batch_sdf)
            sdf_grad[head:tail, :] = batch_sdf_grad.detach()
            sdf_pred[head:tail] = batch_sdf.detach()
            mc_mask[head:tail] = nn_count >= mask_min_nn_count
            certainty[head:tail] = batch_certainty.detach()

        # 剔除异常观测
        grad_norm = sdf_grad.norm(dim=-1, keepdim=True).squeeze()
        max_sdf_std = self.config.surface_sample_range_m * self.config.max_sdf_std_ratio
        valid_idx = (
            mc_mask
            & (grad_norm < max_grad_norm)
            & (grad_norm > min_grad_norm)
            & (sdf_std < max_sdf_std)
        )
        valid_points = pc_map[valid_idx]
        N = valid_points.shape[0]
        pc_imu = pc_imu[valid_idx]
        grad_norm = grad_norm[valid_idx]
        sdf_pred = sdf_pred[valid_idx]
        sdf_grad = sdf_grad[valid_idx]

        # 计算雅可比矩阵
        H = torch.zeros((N, STATE_DIM), device=self.device, dtype=self.tran_dtype)
        pc_imu_hat = batch_vec2skew(pc_imu)
        rotation = self.x.rot.to(dtype=self.dtype).unsqueeze(0)
        A = torch.bmm(rotation.repeat(N, 1, 1), pc_imu_hat)
        H[:, 0:3] = -torch.bmm(sdf_grad.unsqueeze(1), A).squeeze(1)
        H[:, 3:6] = sdf_grad

        # 计算不确定度（对精度有一个轻微的提升）
        sdf_residual = sdf_pred.to(dtype=self.tran_dtype)
        grad_anomaly = (grad_norm - 1.0).to(dtype=self.tran_dtype)
        w_grad = 1 / (1 + grad_anomaly**2)
        w_res = 0.4 / (0.4 + sdf_residual**2)
        self.R_inv = w_grad * w_res * 1000

        return sdf_residual, H, valid_points

    def update_iterated(self, pc_imu: torch.tensor):
        """
        使用迭代方法更新状态估计。

        Args:
        source_points (np.array): 测量点云，假定为 Nx3 矩阵。
        maximum_iter (int): 最大迭代次数。
        """
        # 将状态量和协方差矩阵转移到GPU
        self.x.cuda()
        self.P = self.P.cuda()
        valid_flag = True
        converged = False
        iteration_count = 0
        rejection_reason = ""
        final_linearized_residual = None
        final_H = None
        final_valid_points = None

        x_propagated = self.x
        P_inv = torch.linalg.inv(self.P)
        I = torch.eye(STATE_DIM, device=self.device, dtype=self.tran_dtype)
        term_thre_deg = self.config.reg_term_thre_deg
        term_thre_m = self.config.reg_term_thre_m

        for i in range(self.max_iteration):
            iteration_count = i + 1
            dx_new = boxminus(self.x, x_propagated)
            z, H, valid_points = self.h_model(pc_imu)
            N = valid_points.shape[0]
            source_point_count = pc_imu.shape[0]

            if N / source_point_count < 0.2 and i == self.max_iteration - 1:
                if not self.config.silence:
                    print(
                        "[bold yellow](Warning) registration failed: not enough valid points[/bold yellow]"
                    )
                valid_flag = False
                rejection_reason = "not_enough_valid_points"

            S, b, _, _, _ = _shape_lidar_normal_equations(
                H,
                self.R_inv,
                z,
                self.config.degeneracy_lever_arm_m,
                self.config.reg_position_sigma_m,
                self.config.reg_information_cap_on,
            )
            K_front = torch.linalg.inv(S + P_inv)
            K_front = 0.5 * (K_front + K_front.T)
            dx_ = -K_front @ b + (K_front @ S - I) @ dx_new
            self.x = boxplus(self.x, dx_)
            final_linearized_residual = z + H @ dx_
            final_H = H
            final_valid_points = valid_points
            tran_m = dx_[3:6].norm()
            rot_angle_deg = dx_[0:3].norm() * 180.0 / np.pi

            # 第一种迭代终止判定方式（有物理含义）
            if (
                rot_angle_deg < term_thre_deg
                and tran_m < term_thre_m
                and torch.all(torch.abs(dx_[6:]) < self.eps)
            ):
                if not self.config.silence:
                    print("Converged after", i, "iterations")
                converged = True

            # 第二种迭代终止判定方式
            # if torch.all(torch.abs(dx_) < self.eps):
            #     if not self.config.silence:
            #         print("Converged after", i, "iterations")
            #     converged = True

            if not valid_flag or converged:
                break

        self.P = 0.5 * (K_front + K_front.T)

        # The last query is linearized at the state immediately before the final
        # correction. Reproject its residual to the returned state without an
        # additional full neural-map query.
        z_final = final_linearized_residual
        H_final = final_H
        valid_points_final = final_valid_points
        pose_information = (H_final[:, :6].T * self.R_inv) @ H_final[:, :6]
        information_eigenvalues = torch.linalg.eigvalsh(pose_information)
        _, _, normalized_eigenvalues, eigenvectors, lidar_weights = (
            _shape_lidar_normal_equations(
                H_final,
                self.R_inv,
                z_final,
                self.config.degeneracy_lever_arm_m,
                self.config.reg_position_sigma_m,
                self.config.reg_information_cap_on,
            )
        )
        longitudinal_direction = torch.zeros(
            6, dtype=self.tran_dtype, device=self.device
        )
        longitudinal_direction[3:6] = x_propagated.rot[:, 0]
        direction_coefficients = eigenvectors.T @ longitudinal_direction
        direction_norm = torch.dot(direction_coefficients, direction_coefficients)
        longitudinal_weight = 0.0
        if direction_norm > 0.0:
            lambda_ref = 1.0 / self.config.reg_position_sigma_m**2
            capped_eigenvalues = normalized_eigenvalues * lidar_weights
            longitudinal_weight = float(
                torch.dot(direction_coefficients.square(), capped_eigenvalues)
                / (direction_norm * lambda_ref)
            )
        longitudinal_weight = min(max(longitudinal_weight, 0.0), 1.0)
        self.previous_lidar_longitudinal_weight = longitudinal_weight
        information_sum = torch.sum(normalized_eigenvalues)
        capped_fraction = 0.0
        if information_sum > 0.0:
            capped_fraction = float(
                torch.sum(normalized_eigenvalues * (1.0 - lidar_weights))
                / information_sum
            )
        condition_number = float("inf")
        if information_eigenvalues[0] > 0:
            condition_number = float(
                information_eigenvalues[-1] / information_eigenvalues[0]
            )
        correction = boxminus(self.x, x_propagated)
        residual_abs = torch.abs(z_final)
        residual_median = float("nan")
        residual_rmse = float("nan")
        residual_q90 = float("nan")
        if residual_abs.numel():
            residual_median = float(torch.median(residual_abs))
            residual_rmse = float(torch.sqrt(torch.mean(z_final**2)))
            residual_q90 = float(torch.quantile(residual_abs, 0.9))
        source_point_count = int(pc_imu.shape[0])
        valid_point_count = int(valid_points_final.shape[0])
        self.last_registration_diagnostics = {
            "source_point_count": source_point_count,
            "valid_sdf_point_count": valid_point_count,
            "valid_point_ratio": (
                valid_point_count / source_point_count if source_point_count else 0.0
            ),
            "sdf_residual_median": residual_median,
            "sdf_residual_rmse": residual_rmse,
            "sdf_residual_q90": residual_q90,
            "optimizer_converged": converged,
            "optimizer_iteration_count": iteration_count,
            "prediction_lidar_translation_m": float(correction[3:6].norm()),
            "prediction_lidar_rotation_deg": float(
                correction[0:3].norm() * 180.0 / np.pi
            ),
            **{
                f"information_eigenvalue_{index}": float(value)
                for index, value in enumerate(information_eigenvalues)
            },
            **{
                f"lidar_weight_{index}": float(value)
                for index, value in enumerate(lidar_weights)
            },
            "lidar_longitudinal_weight": longitudinal_weight,
            "lidar_information_capped_fraction": capped_fraction,
            "degeneracy_lambda_min_abs": float(normalized_eigenvalues[0]),
            "information_condition_number": condition_number,
            "rejection_reason": rejection_reason,
            **self.vme_diagnostics,
        }
        updated_pose = torch.eye(4, dtype=self.dtype, device=self.device)
        updated_pose[:3, :3] = self.x.rot.to(self.dtype)
        updated_pose[:3, 3] = self.x.pos.to(self.dtype)

        # 将状态量和协方差矩阵转移到CPU
        self.x.cpu()
        self.P = self.P.cpu()
        return updated_pose, valid_flag

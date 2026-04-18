#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soft support + lag-gain prior utilities."""

from typing import Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import torch


ArrayLike3D = Union[np.ndarray, torch.Tensor]


def _as_numpy_3d(data_3d: ArrayLike3D) -> np.ndarray:
    """Convert `[subjects, nodes, time]` input to float64 NumPy."""
    if isinstance(data_3d, torch.Tensor):
        array = data_3d.detach().cpu().numpy()
    else:
        array = np.asarray(data_3d)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D data [subjects, nodes, time], got shape {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError(f"All dimensions must be positive, got shape {array.shape}")
    return array.astype(np.float64, copy=False)


def _normalize_subject(subject_nt: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-subject normalized support signal `x_hat` and standardized lag signal `z`."""
    subject_tn = np.asarray(subject_nt, dtype=np.float64).T  # [T, N]
    p10 = np.percentile(subject_tn, 10, axis=0, keepdims=True)
    p90 = np.percentile(subject_tn, 90, axis=0, keepdims=True)
    denom = np.maximum(p90 - p10, eps)
    x_hat = np.clip((subject_tn - p10) / denom, 0.0, 1.0)
    mean = x_hat.mean(axis=0, keepdims=True)
    std = x_hat.std(axis=0, keepdims=True)
    z = (x_hat - mean) / np.maximum(std, eps)
    return x_hat, z


def _compute_soft_contingency(phi_tn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute soft contingency masses from `[T, N]` soft events."""
    time_points = phi_tn.shape[0]
    phi_nt = phi_tn.T
    phi_not_nt = 1.0 - phi_nt
    q11 = (phi_nt @ phi_nt.T) / time_points
    q10 = (phi_nt @ phi_not_nt.T) / time_points
    q01 = (phi_not_nt @ phi_nt.T) / time_points
    q00 = (phi_not_nt @ phi_not_nt.T) / time_points
    return q11, q10, q01, q00


def _compute_bounded_kappa(
    q11: np.ndarray,
    q10: np.ndarray,
    q01: np.ndarray,
    q00: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Soft-Patel bounded normalization."""
    expected = (q11 + q10) * (q11 + q01)
    upper = np.minimum(q11 + q10, q11 + q01)
    lower = np.maximum(0.0, 2.0 * q11 + q10 + q01 - 1.0)

    weight = np.zeros_like(q11)
    mask_gt = q11 > expected
    denom_gt = 2.0 * np.maximum(upper - expected, eps)
    weight[mask_gt] = 0.5 + (q11[mask_gt] - expected[mask_gt]) / denom_gt[mask_gt]

    mask_le = ~mask_gt
    denom_le = 2.0 * np.maximum(expected - lower, eps)
    weight[mask_le] = 0.5 - (q11[mask_le] - expected[mask_le]) / denom_le[mask_le]

    numerator = q11 - expected
    denominator = weight * (upper - expected) + (1.0 - weight) * (expected - lower)
    denominator = np.maximum(denominator, eps)
    return numerator / denominator


def _solve_ridge_system(batch_xtx: np.ndarray, batch_xty: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Solve one or more ridge systems with a safe pinv fallback."""
    feat_dim = batch_xtx.shape[-1]
    reg_eye = ridge_lambda * np.eye(feat_dim, dtype=batch_xtx.dtype)
    flat_xtx = (batch_xtx + reg_eye).reshape(-1, feat_dim, feat_dim)
    flat_xty = batch_xty.reshape(-1, feat_dim)
    coeffs = np.empty_like(flat_xty)
    for idx in range(flat_xtx.shape[0]):
        try:
            coeffs[idx] = np.linalg.solve(flat_xtx[idx], flat_xty[idx])
        except np.linalg.LinAlgError:
            coeffs[idx] = np.linalg.pinv(flat_xtx[idx]) @ flat_xty[idx]
    return coeffs.reshape(batch_xty.shape)


def _build_lagged_history(
    z_tn: np.ndarray,
    lags: Sequence[int],
    lag_weights: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build target values and weighted lag histories."""
    max_lag = max(int(v) for v in lags)
    time_points = int(z_tn.shape[0])
    if time_points <= max_lag:
        raise ValueError(
            f"All subjects must have time_points > max(lags), got time_points={time_points}, max_lag={max_lag}"
        )
    y_tn = z_tn[max_lag:, :]
    sqrt_weights = [np.sqrt(float(v)) for v in lag_weights]
    history = np.stack(
        [
            sqrt_weights[idx] * z_tn[max_lag - lag : time_points - lag, :]
            for idx, lag in enumerate(lags)
        ],
        axis=-1,
    )
    return y_tn, history  # [T_eff, N], [T_eff, N, M]


def _compute_subject_direction_gain(
    z_tn: np.ndarray,
    lags: Sequence[int],
    lag_weights: Sequence[float],
    ridge_lambda: float,
    eps: float,
) -> np.ndarray:
    """Compute one subject's lag-gain matrix `D_subject[source, target]`."""
    y_tn, history_tnm = _build_lagged_history(z_tn, lags, lag_weights)
    num_targets = y_tn.shape[1]
    lag_dim = history_tnm.shape[2]
    direction = np.zeros((num_targets, num_targets), dtype=np.float64)

    for target in range(num_targets):
        y_target = y_tn[:, target]
        self_history = history_tnm[:, target, :]
        xtx_self = self_history.T @ self_history
        xty_self = self_history.T @ y_target
        self_coef = _solve_ridge_system(
            xtx_self[None, :, :],
            xty_self[None, :],
            ridge_lambda=ridge_lambda,
        )[0]
        self_residual = y_target - self_history @ self_coef
        self_error = float(np.mean(np.square(self_residual)))
        if self_error <= eps:
            continue

        self_block = np.broadcast_to(
            self_history[:, None, :],
            (self_history.shape[0], num_targets, lag_dim),
        )
        aug_history = np.concatenate([self_block, history_tnm], axis=2)  # [T_eff, N, 2M]
        xtx_aug = np.einsum('tnp,tnq->npq', aug_history, aug_history)
        xty_aug = np.einsum('tnp,t->np', aug_history, y_target)
        aug_coef = _solve_ridge_system(xtx_aug, xty_aug, ridge_lambda=ridge_lambda)
        pred_aug = np.einsum('tnp,np->tn', aug_history, aug_coef)
        aug_error = np.mean(np.square(pred_aug - y_target[:, None]), axis=0)
        direction[:, target] = np.maximum(0.0, 1.0 - aug_error / (self_error + eps))

    np.fill_diagonal(direction, 0.0)
    return np.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_direction_reliability(direction_subjects: np.ndarray, eps: float) -> np.ndarray:
    """Compute reliability from subject-wise direction contrasts."""
    num_subjects, num_nodes, _ = direction_subjects.shape
    if num_subjects < 5:
        reliability = np.ones((num_nodes, num_nodes), dtype=np.float64)
        np.fill_diagonal(reliability, 1.0)
        return reliability

    delta_subjects = direction_subjects - np.transpose(direction_subjects, (0, 2, 1))
    delta_median = np.median(delta_subjects, axis=0)
    mad = np.median(np.abs(delta_subjects - delta_median[None, :, :]), axis=0)
    median_abs = np.median(np.abs(delta_subjects), axis=0)
    reliability = np.exp(-mad / (median_abs + eps))
    reliability = np.clip(reliability, 0.0, 1.0)
    np.fill_diagonal(reliability, 1.0)
    return np.nan_to_num(reliability, nan=1.0, posinf=1.0, neginf=0.0)


def _compute_asymmetric_score(
    support: np.ndarray,
    direction: np.ndarray,
    score_alpha: float,
    eps: float,
) -> np.ndarray:
    """Build the causal-convention asymmetric score `A[cause, effect]`."""
    num_nodes = support.shape[0]
    off_diag_mask = ~np.eye(num_nodes, dtype=bool)
    support_off = support[off_diag_mask]
    direction_delta = direction - direction.T
    delta_off = np.abs(direction_delta[off_diag_mask])

    support_scale = float(np.quantile(support_off, 0.95)) if support_off.size > 0 else 0.0
    delta_scale = float(np.quantile(delta_off, 0.90)) if delta_off.size > 0 else 0.0
    support_scaled = support / (support_scale + eps)
    delta_scaled = direction_delta / (delta_scale + eps)
    score = support_scaled * np.tanh(float(score_alpha) * delta_scaled)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(score, 0.0)
    return score


def compute_soft_prior_components(
    data_3d: ArrayLike3D,
    K: int = 5,
    beta: float = 10.0,
    lags: Sequence[int] = (1, 2, 3, 4, 5),
    lag_weights: Optional[Sequence[float]] = None,
    ridge_lambda: float = 1e-3,
    score_alpha: float = 1.0,
    variance_floor: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the V1 soft prior components `(S, D, R_D, A)`.

    Args:
        data_3d: Subject-major fMRI data `[subjects, nodes, time]`.
        K: Number of soft thresholds.
        beta: Sigmoid sharpness for soft events.
        lags: Positive lag steps used by lag-gain direction scoring.
        lag_weights: Optional non-negative lag weights aligned with `lags`.
        ridge_lambda: Ridge penalty for the lag-gain regressions.
        score_alpha: Direction sensitivity used by `A = S * tanh(alpha * delta)`.
        variance_floor: Threshold-validity cutoff on median node variance.
        eps: Numerical stabilizer.

    Returns:
        Tuple `(S, D, R_D, A)` as NumPy arrays with shape `[N, N]`.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if ridge_lambda < 0.0:
        raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")
    if score_alpha < 0.0:
        raise ValueError(f"score_alpha must be >= 0, got {score_alpha}")
    if variance_floor < 0.0:
        raise ValueError(f"variance_floor must be >= 0, got {variance_floor}")
    lag_list = tuple(int(v) for v in lags)
    if not lag_list:
        raise ValueError("lags must be non-empty")
    if any(v <= 0 for v in lag_list):
        raise ValueError(f"lags must all be positive, got {lag_list}")
    if lag_weights is None:
        weight_list = tuple(1.0 for _ in lag_list)
    else:
        weight_list = tuple(float(v) for v in lag_weights)
        if len(weight_list) != len(lag_list):
            raise ValueError(
                f"lag_weights must align with lags, got {len(weight_list)} vs {len(lag_list)}"
            )
        if any(v < 0.0 for v in weight_list):
            raise ValueError(f"lag_weights must be non-negative, got {weight_list}")
        if not any(v > 0.0 for v in weight_list):
            raise ValueError(f"lag_weights must contain at least one positive value, got {weight_list}")
    total_weight = float(sum(weight_list))
    weight_list = tuple(v / total_weight for v in weight_list)

    if beta < 8.0:
        warnings.warn(
            f"soft prior beta={beta:g} < 8 may collapse soft-event variance and degenerate support.",
            RuntimeWarning,
        )

    array_snt = _as_numpy_3d(data_3d)
    num_subjects, num_nodes, _ = array_snt.shape
    normalized_subjects = [_normalize_subject(array_snt[idx], eps=eps) for idx in range(num_subjects)]

    thresholds = np.arange(1, K + 1, dtype=np.float64) / float(K + 1)
    kappa_terms = []

    for threshold in thresholds:
        pooled_q11 = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        pooled_q10 = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        pooled_q01 = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        pooled_q00 = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        pooled_variances = []
        total_time = 0.0

        for x_hat_tn, _ in normalized_subjects:
            phi_tn = 1.0 / (1.0 + np.exp(-beta * (x_hat_tn - threshold)))
            pooled_variances.append(phi_tn.var(axis=0))
            q11, q10, q01, q00 = _compute_soft_contingency(phi_tn)
            time_points = float(phi_tn.shape[0])
            pooled_q11 += time_points * q11
            pooled_q10 += time_points * q10
            pooled_q01 += time_points * q01
            pooled_q00 += time_points * q00
            total_time += time_points

        median_variance = float(np.median(np.concatenate(pooled_variances, axis=0)))
        if median_variance < variance_floor:
            continue

        q11_bar = pooled_q11 / max(total_time, eps)
        q10_bar = pooled_q10 / max(total_time, eps)
        q01_bar = pooled_q01 / max(total_time, eps)
        q00_bar = pooled_q00 / max(total_time, eps)
        kappa = _compute_bounded_kappa(q11_bar, q10_bar, q01_bar, q00_bar, eps=eps)
        kappa_terms.append(np.maximum(kappa, 0.0))

    if kappa_terms:
        support = np.mean(np.stack(kappa_terms, axis=0), axis=0)
    else:
        warnings.warn(
            "No valid soft support thresholds passed the variance check; returning zero support.",
            RuntimeWarning,
        )
        support = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    direction_subjects = np.stack(
        [
            _compute_subject_direction_gain(
                z_tn=z_tn,
                lags=lag_list,
                lag_weights=weight_list,
                ridge_lambda=ridge_lambda,
                eps=eps,
            )
            for _, z_tn in normalized_subjects
        ],
        axis=0,
    )
    direction = np.median(direction_subjects, axis=0)
    reliability = _compute_direction_reliability(direction_subjects, eps=eps)

    support = 0.5 * (support + support.T)
    support = np.maximum(support, 0.0)
    direction = np.maximum(direction, 0.0)

    support = np.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0)
    direction = np.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(support, 0.0)
    np.fill_diagonal(direction, 0.0)

    score = _compute_asymmetric_score(support, direction, score_alpha=score_alpha, eps=eps)
    return support, direction, reliability, score

#!/usr/bin/env python3
import argparse
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


# ----------------------------
# Basic utilities
# ----------------------------

def parse_int_list(s: str) -> list[int]:
    return [int(item.strip()) for item in s.split(",") if item.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(item.strip()) for item in s.split(",") if item.strip()]


def l2_norm(err: np.ndarray, dx: float) -> float:
    return float(np.sqrt(dx * np.sum(err**2)))


def linf_norm(err: np.ndarray) -> float:
    return float(np.max(np.abs(err)))


def periodic_pad(u: np.ndarray, ng: int = 2) -> np.ndarray:
    up = np.empty(len(u) + 2 * ng, dtype=u.dtype)
    up[:ng] = u[-ng:]
    up[ng:-ng] = u
    up[-ng:] = u[:ng]
    return up


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    mask = (a * b) > 0.0
    out[mask] = np.sign(a[mask]) * np.minimum(np.abs(a[mask]), np.abs(b[mask]))
    return out


# ----------------------------
# PDE pieces
# ----------------------------

def burgers_flux(u: np.ndarray) -> np.ndarray:
    return 0.5 * u**2


def reconstruct_muscl(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    MUSCL piecewise-linear reconstruction with minmod limiting.
    Returns left/right states at interfaces i+1/2 on a periodic grid.
    """
    up = periodic_pad(u, ng=2)

    um = up[1:-3]   # i-1
    ui = up[2:-2]   # i
    up1 = up[3:-1]  # i+1

    slope = minmod(ui - um, up1 - ui)

    uL_cell = ui - 0.5 * slope
    uR_cell = ui + 0.5 * slope

    # interface i+1/2:
    uL = uR_cell
    uR = np.roll(uL_cell, -1)
    return uL, uR


def burgers_rusanov_flux(uL: np.ndarray, uR: np.ndarray) -> np.ndarray:
    fL = burgers_flux(uL)
    fR = burgers_flux(uR)
    a = np.maximum(np.abs(uL), np.abs(uR))
    return 0.5 * (fL + fR) - 0.5 * a * (uR - uL)


def burgers_central_flux(uL: np.ndarray, uR: np.ndarray) -> np.ndarray:
    return 0.5 * (burgers_flux(uL) + burgers_flux(uR))


def burgers_godunov_flux(uL: np.ndarray, uR: np.ndarray) -> np.ndarray:
    """
    Exact Godunov flux for scalar Burgers, f(u)=u^2/2.
    """
    fL = burgers_flux(uL)
    fR = burgers_flux(uR)
    out = np.empty_like(uL)

    rare = uL <= uR
    shock = ~rare

    # Rarefaction
    mask = rare & (uL >= 0.0)
    out[mask] = fL[mask]

    mask = rare & (uR <= 0.0)
    out[mask] = fR[mask]

    mask = rare & (uL < 0.0) & (uR > 0.0)
    out[mask] = 0.0

    # Shock
    s = 0.5 * (uL + uR)
    mask = shock & (s >= 0.0)
    out[mask] = fL[mask]

    mask = shock & (s < 0.0)
    out[mask] = fR[mask]

    return out


def numerical_flux_convective(u: np.ndarray, flux_type: str = "godunov") -> np.ndarray:
    uL, uR = reconstruct_muscl(u)
    flux_type = flux_type.lower()

    if flux_type == "rusanov":
        return burgers_rusanov_flux(uL, uR)
    if flux_type == "godunov":
        return burgers_godunov_flux(uL, uR)
    if flux_type == "central":
        return burgers_central_flux(uL, uR)

    raise ValueError("Unknown flux_type.")


def rhs_viscous_burgers(
    u: np.ndarray,
    nu: float,
    dx: float,
    flux_type: str = "godunov",
) -> np.ndarray:
    """
    u_t + (u^2/2)_x = nu u_xx
    """
    F = numerical_flux_convective(u, flux_type=flux_type)
    divF = (F - np.roll(F, 1)) / dx
    lap = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2
    return -divF + nu * lap


def ssprk2_step(
    u: np.ndarray,
    dt: float,
    nu: float,
    dx: float,
    flux_type: str = "godunov",
) -> np.ndarray:
    k1 = rhs_viscous_burgers(u, nu, dx, flux_type=flux_type)
    u1 = u + dt * k1
    k2 = rhs_viscous_burgers(u1, nu, dx, flux_type=flux_type)
    return 0.5 * u + 0.5 * (u1 + dt * k2)


def solve_viscous_burgers(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    tfinal: float,
    flux_type: str = "godunov",
    cfl_adv: float = 0.4,
    cfl_diff: float = 0.4,
) -> np.ndarray:
    u = u0.copy()
    dx = x[1] - x[0]
    t = 0.0

    while t < tfinal:
        umax = max(float(np.max(np.abs(u))), 1e-12)
        dt_adv = cfl_adv * dx / umax
        dt_diff = cfl_diff * dx**2 / max(nu, 1e-14)
        dt = min(dt_adv, dt_diff, tfinal - t)
        u = ssprk2_step(u, dt, nu, dx, flux_type=flux_type)
        t += dt

    return u


# ----------------------------
# Initial conditions
# ----------------------------

def make_initial_condition(x: np.ndarray, ic_type: str = "sine") -> np.ndarray:
    L = x[-1] - x[0] + (x[1] - x[0])
    xc = 0.5 * (x[0] + x[-1] + (x[1] - x[0]))

    ic_type = ic_type.lower()

    if ic_type == "sine":
        return np.sin(2.0 * np.pi * (x - x[0]) / L)

    if ic_type == "sine2":
        return np.sin(2.0 * np.pi * (x - x[0]) / L) + 0.25 * np.sin(4.0 * np.pi * (x - x[0]) / L)

    if ic_type == "sine-mean":
        return 1.0 + 0.5 * np.sin(2.0 * np.pi * (x - x[0]) / L)

    if ic_type == "bump":
        return np.exp(-20.0 * (x - xc) ** 2)

    if ic_type == "step":
        return np.where(x < x[0] + 0.5 * L, 1.0, 0.0)

    raise ValueError("Unknown ic_type.")


# ----------------------------
# Extrapolation / blending
# ----------------------------

def smoothness_sensor(u: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = np.abs(np.roll(u, -1) - 2.0 * u + np.roll(u, 1))
    den = np.abs(np.roll(u, -1) - np.roll(u, 1)) + eps
    return num / den


def extremum_mask(u: np.ndarray) -> np.ndarray:
    dm = u - np.roll(u, 1)
    dp = np.roll(u, -1) - u
    return ((dm * dp) > 0.0).astype(float)


def blending_weight(u: np.ndarray, S0: float = 0.1, p: float = 4.0, eps: float = 1e-12) -> np.ndarray:
    S = smoothness_sensor(u, eps=eps)
    chi = 1.0 / (1.0 + (S / S0) ** p)
    chi *= extremum_mask(u)
    return chi


def estimate_shock_resolution(u: np.ndarray, nu: float, dx: float, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Burgers viscous thickness estimate:
        delta ~ nu / Delta u
    """
    jump = np.abs(np.roll(u, -1) - u)
    delta = nu / np.maximum(jump, eps)
    n_cells = delta / dx
    return n_cells, delta


def choose_extrapolation_mode(
    u_nu: np.ndarray,
    nu: float,
    dx: float,
    mode: str = "auto",
    resolved_cells_threshold: float = 4.0,
    steep_fraction_threshold: float = 0.01,
) -> tuple[str, dict[str, Any]]:
    mode = mode.lower()

    if mode == "always":
        return "always", {"reason": "User requested blending everywhere."}

    if mode == "none":
        return "none", {"reason": "User requested raw Richardson extrapolation everywhere."}

    if mode != "auto":
        raise ValueError("mode must be one of: auto, always, none")

    jump = np.abs(np.roll(u_nu, -1) - u_nu)
    n_cells, _ = estimate_shock_resolution(u_nu, nu, dx)

    jump_max = np.max(jump)
    if float(jump_max) <= 1e-14:
        return "none", {
            "reason": "Solution is essentially flat.",
            "min_cells_in_steep_region": np.inf,
        }

    steep_mask = jump > steep_fraction_threshold * jump_max
    if not np.any(steep_mask):
        return "none", {
            "reason": "No appreciable steep region detected.",
            "min_cells_in_steep_region": np.inf,
        }

    min_cells = float(np.min(n_cells[steep_mask]))

    if min_cells >= resolved_cells_threshold:
        return "none", {
            "reason": "Estimated viscous layer resolved; using raw extrapolation.",
            "min_cells_in_steep_region": min_cells,
            "threshold": float(resolved_cells_threshold),
        }

    return "always", {
        "reason": "Estimated viscous layer underresolved; using blending.",
        "min_cells_in_steep_region": min_cells,
        "threshold": float(resolved_cells_threshold),
    }


def build_extrapolated_solution(
    u_nu: np.ndarray,
    u_2nu: np.ndarray,
    nu: float,
    dx: float,
    mode: str = "auto",
    S0: float = 0.1,
    p: float = 4.0,
    resolved_cells_threshold: float = 4.0,
    steep_fraction_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    selected_mode, info = choose_extrapolation_mode(
        u_nu=u_nu,
        nu=nu,
        dx=dx,
        mode=mode,
        resolved_cells_threshold=resolved_cells_threshold,
        steep_fraction_threshold=steep_fraction_threshold,
    )

    if selected_mode == "none":
        chi = np.ones_like(u_nu)
    else:
        chi = blending_weight(u_nu, S0=S0, p=p)

    u_rich = 2.0 * u_nu - u_2nu
    u_post = u_nu + chi * (u_nu - u_2nu)

    info["selected_mode"] = selected_mode
    return u_rich, u_post, chi, info


# ----------------------------
# Exact / reference targets
# ----------------------------

def cole_hopf_periodic(u0: np.ndarray, x: np.ndarray, nu: float, t: float) -> np.ndarray:
    """
    Exact periodic viscous Burgers solution via Cole-Hopf, including nonzero mean.
    """
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(len(x), d=dx)

    m = float(np.mean(u0))
    w0 = u0 - m

    # Build periodic potential q with q_x = w0, mean(q)=0
    w0_hat = np.fft.fft(w0)
    q_hat = np.zeros_like(w0_hat, dtype=complex)
    nz = np.abs(k) > 0.0
    q_hat[nz] = w0_hat[nz] / (1j * k[nz])
    q_hat[~nz] = 0.0
    q0 = np.real(np.fft.ifft(q_hat))

    phi0 = np.exp(-q0 / (2.0 * nu))
    phi0_hat = np.fft.fft(phi0)

    decay = np.exp(-nu * k**2 * t)
    phi_hat_t = phi0_hat * decay
    phi_y_hat_t = 1j * k * phi_hat_t

    # Galilean shift for nonzero mean
    shift = m * t
    phase = np.exp(-1j * k * shift)

    phi = np.real(np.fft.ifft(phi_hat_t * phase))
    phi_y = np.real(np.fft.ifft(phi_y_hat_t * phase))

    u = m - 2.0 * nu * phi_y / phi
    return u


def periodic_resample(u_old: np.ndarray, x_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Periodic resampling on uniform grids using Fourier interpolation/truncation.
    """
    N_old = len(x_old)
    dx_old = x_old[1] - x_old[0]
    k_old = 2.0 * np.pi * np.fft.fftfreq(N_old, d=dx_old)
    uhat_old = np.fft.fft(u_old)

    x0 = x_old[0]
    phases = np.exp(1j * np.outer(k_old, x_new - x0))
    u_new = np.real((uhat_old[:, None] * phases).sum(axis=0) / N_old)
    return u_new


def reference_solution_same_nu(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    tfinal: float,
    flux_type: str = "godunov",
    N_factor: int = 8,
    cfl_adv: float = 0.4,
    cfl_diff: float = 0.4,
) -> np.ndarray:
    """
    High-resolution same-viscosity reference.
    """
    N_ref = N_factor * len(x)
    L = x[-1] - x[0] + (x[1] - x[0])
    x_ref = x[0] + np.arange(N_ref) * (L / N_ref)

    u0_ref = periodic_resample(u0, x, x_ref)

    u_ref = solve_viscous_burgers(
        u0_ref,
        x_ref,
        nu,
        tfinal,
        flux_type=flux_type,
        cfl_adv=cfl_adv,
        cfl_diff=cfl_diff,
    )
    return periodic_resample(u_ref, x_ref, x)


def inviscid_reference_solution(
    u0: np.ndarray,
    x: np.ndarray,
    tfinal: float,
    flux_type: str = "godunov",
    N_factor: int = 8,
    nu_small: float = 1e-5,
    cfl_adv: float = 0.4,
    cfl_diff: float = 0.4,
) -> np.ndarray:
    """
    Fine-grid tiny-viscosity reference as a proxy for the inviscid entropy solution.
    """
    N_ref = N_factor * len(x)
    L = x[-1] - x[0] + (x[1] - x[0])
    x_ref = x[0] + np.arange(N_ref) * (L / N_ref)

    u0_ref = periodic_resample(u0, x, x_ref)
    u_ref = solve_viscous_burgers(
        u0_ref,
        x_ref,
        nu_small,
        tfinal,
        flux_type=flux_type,
        cfl_adv=cfl_adv,
        cfl_diff=cfl_diff,
    )
    return periodic_resample(u_ref, x_ref, x)


def compute_viscous_target(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    tfinal: float,
    exact_kind: str,
    flux_type: str,
    ref_factor: int,
    cfl_adv: float,
    cfl_diff: float,
) -> np.ndarray | None:
    if exact_kind == "none":
        return None

    if exact_kind == "colehopf":
        return cole_hopf_periodic(u0, x, nu, tfinal)

    if exact_kind == "reference":
        return reference_solution_same_nu(
            u0,
            x,
            nu,
            tfinal,
            flux_type=flux_type,
            N_factor=ref_factor,
            cfl_adv=cfl_adv,
            cfl_diff=cfl_diff,
        )

    raise ValueError("Unknown exact_kind.")


# ----------------------------
# Diagnostics
# ----------------------------

def compute_errors_for_run(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    tfinal: float,
    flux_type: str,
    mode: str,
    S0: float,
    p: float,
    resolved_cells_threshold: float,
    steep_fraction_threshold: float,
    exact_kind: str,
    ref_factor: int,
    nu_small_ref: float,
    cfl_adv: float,
    cfl_diff: float,
) -> dict[str, Any]:
    dx = x[1] - x[0]

    u_nu = solve_viscous_burgers(
        u0=u0,
        x=x,
        nu=nu,
        tfinal=tfinal,
        flux_type=flux_type,
        cfl_adv=cfl_adv,
        cfl_diff=cfl_diff,
    )
    u_2nu = solve_viscous_burgers(
        u0=u0,
        x=x,
        nu=2.0 * nu,
        tfinal=tfinal,
        flux_type=flux_type,
        cfl_adv=cfl_adv,
        cfl_diff=cfl_diff,
    )

    u_rich, u_post, chi, info = build_extrapolated_solution(
        u_nu=u_nu,
        u_2nu=u_2nu,
        nu=nu,
        dx=dx,
        mode=mode,
        S0=S0,
        p=p,
        resolved_cells_threshold=resolved_cells_threshold,
        steep_fraction_threshold=steep_fraction_threshold,
    )

    u_visc_target = compute_viscous_target(
        u0=u0,
        x=x,
        nu=nu,
        tfinal=tfinal,
        exact_kind=exact_kind,
        flux_type=flux_type,
        ref_factor=ref_factor,
        cfl_adv=cfl_adv,
        cfl_diff=cfl_diff,
    )

    u_inv_target = inviscid_reference_solution(
        u0=u0,
        x=x,
        tfinal=tfinal,
        flux_type=flux_type,
        N_factor=ref_factor,
        nu_small=nu_small_ref,
        cfl_adv=cfl_adv,
        cfl_diff=cfl_diff,
    )

    out: dict[str, Any] = {
        "u_nu": u_nu,
        "u_2nu": u_2nu,
        "u_rich": u_rich,
        "u_post": u_post,
        "chi": chi,
        "info": info,
        "u_visc_target": u_visc_target,
        "u_inv_target": u_inv_target,
        "dx": dx,
    }

    if u_visc_target is not None:
        err_nu = u_nu - u_visc_target
        out["err_nu_l2"] = l2_norm(err_nu, dx)
        out["err_nu_linf"] = linf_norm(err_nu)

    err_rich = u_rich - u_inv_target
    err_post = u_post - u_inv_target
    err_2nu = u_2nu - u_inv_target
    err_nu_inv = u_nu - u_inv_target

    out["err_nu_inv_l2"] = l2_norm(err_nu_inv, dx)
    out["err_nu_inv_linf"] = linf_norm(err_nu_inv)
    out["err_2nu_inv_l2"] = l2_norm(err_2nu, dx)
    out["err_2nu_inv_linf"] = linf_norm(err_2nu)
    out["err_rich_l2"] = l2_norm(err_rich, dx)
    out["err_rich_linf"] = linf_norm(err_rich)
    out["err_post_l2"] = l2_norm(err_post, dx)
    out["err_post_linf"] = linf_norm(err_post)

    return out


def estimate_loglog_slope(xvals: np.ndarray, yvals: np.ndarray) -> float:
    xvals = np.asarray(xvals, dtype=float)
    yvals = np.asarray(yvals, dtype=float)
    mask = (xvals > 0.0) & (yvals > 0.0) & np.isfinite(xvals) & np.isfinite(yvals)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    coeff = np.polyfit(np.log(xvals[mask]), np.log(yvals[mask]), 1)
    return float(coeff[0])


# ----------------------------
# Plotting
# ----------------------------

def plot_main_run(x: np.ndarray, u0: np.ndarray, run: dict[str, Any], exact_kind: str) -> None:
    u_nu = cast(np.ndarray, run["u_nu"])
    u_2nu = cast(np.ndarray, run["u_2nu"])
    u_rich = cast(np.ndarray, run["u_rich"])
    u_post = cast(np.ndarray, run["u_post"])
    chi = cast(np.ndarray, run["chi"])
    u_visc_target = cast(np.ndarray | None, run["u_visc_target"])
    u_inv_target = cast(np.ndarray | None, run["u_inv_target"])

    _, ax_ = plt.subplots(figsize=(10, 6))
    ax = cast(Axes, ax_)
    ax.plot(x, u0, label="initial", alpha=0.8)
    ax.plot(x, u_nu, label=r"$U^\nu$")
    ax.plot(x, u_2nu, label=r"$U^{2\nu}$")
    ax.plot(x, u_rich, "--", label=r"$2U^\nu-U^{2\nu}$")
    ax.plot(x, u_post, linewidth=2, label="postprocessed")

    if u_visc_target is not None:
        label = "viscous exact" if exact_kind == "colehopf" else "viscous reference"
        ax.plot(x, u_visc_target, "k", linewidth=2, label=label)

    if u_inv_target is not None:
        ax.plot(x, u_inv_target, color="0.4", linewidth=2, linestyle=":", label="inviscid reference")

    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Solution comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _, ax2_ = plt.subplots(figsize=(10, 3))
    ax2 = cast(Axes, ax2_)
    ax2.plot(x, chi, label=r"$\chi$")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$\chi$")
    ax2.set_title("Blending weight")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if u_visc_target is not None or u_inv_target is not None:
        _, ax3_ = plt.subplots(figsize=(10, 4))
        ax3 = cast(Axes, ax3_)

        if u_visc_target is not None:
            ax3.semilogy(
                x,
                np.abs(u_nu - u_visc_target),
                label=r"$|U^\nu-u^\nu_{\mathrm{target}}|$",
            )

        if u_inv_target is not None:
            ax3.semilogy(
                x,
                np.abs(u_rich - u_inv_target),
                "--",
                label=r"$|2U^\nu-U^{2\nu}-u^0_{\mathrm{ref}}|$",
            )
            ax3.semilogy(
                x,
                np.abs(u_post - u_inv_target),
                label=r"$|U^{\mathrm{post}}-u^0_{\mathrm{ref}}|$",
            )
            ax3.semilogy(
                x,
                np.abs(u_nu - u_inv_target),
                label=r"$|U^\nu-u^0_{\mathrm{ref}}|$",
            )

        ax3.set_xlabel("x")
        ax3.set_ylabel("pointwise error")
        ax3.set_title("Pointwise error")
        ax3.grid(True, alpha=0.3)
        ax3.legend()


def plot_dx_study(dx_vals: np.ndarray, data: dict[str, np.ndarray], exact_kind: str) -> None:
    _, ax_ = plt.subplots(figsize=(8, 5))
    ax = cast(Axes, ax_)

    if exact_kind != "none" and "nu_visc_l2" in data:
        ax.loglog(dx_vals, data["nu_visc_l2"], "o-", label=r"$\|U^\nu-u^\nu_{\mathrm{target}}\|_{L^2}$")

    ax.loglog(dx_vals, data["nu_inv_l2"], "o-", label=r"$\|U^\nu-u^0_{\mathrm{ref}}\|_{L^2}$")
    ax.loglog(dx_vals, data["rich_l2"], "o-", label=r"$\|2U^\nu-U^{2\nu}-u^0_{\mathrm{ref}}\|_{L^2}$")
    ax.loglog(dx_vals, data["post_l2"], "o-", label=r"$\|U^{\mathrm{post}}-u^0_{\mathrm{ref}}\|_{L^2}$")
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$L^2$ error")
    ax.set_title(r"$L^2$ error vs $\Delta x$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    _, ax2_ = plt.subplots(figsize=(8, 5))
    ax2 = cast(Axes, ax2_)

    if exact_kind != "none" and "nu_visc_linf" in data:
        ax2.loglog(dx_vals, data["nu_visc_linf"], "o-", label=r"$\|U^\nu-u^\nu_{\mathrm{target}}\|_{L^\infty}$")

    ax2.loglog(dx_vals, data["nu_inv_linf"], "o-", label=r"$\|U^\nu-u^0_{\mathrm{ref}}\|_{L^\infty}$")
    ax2.loglog(dx_vals, data["rich_linf"], "o-", label=r"$\|2U^\nu-U^{2\nu}-u^0_{\mathrm{ref}}\|_{L^\infty}$")
    ax2.loglog(dx_vals, data["post_linf"], "o-", label=r"$\|U^{\mathrm{post}}-u^0_{\mathrm{ref}}\|_{L^\infty}$")
    ax2.set_xlabel(r"$\Delta x$")
    ax2.set_ylabel(r"$L^\infty$ error")
    ax2.set_title(r"$L^\infty$ error vs $\Delta x$")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()


def plot_nu_study(nu_vals: np.ndarray, data: dict[str, np.ndarray]) -> None:
    _, ax_ = plt.subplots(figsize=(8, 5))
    ax = cast(Axes, ax_)
    ax.loglog(nu_vals, data["nu_inv_l2"], "o-", label=r"$\|U^\nu-u^0_{\mathrm{ref}}\|_{L^2}$")
    ax.loglog(nu_vals, data["rich_l2"], "o-", label=r"$\|2U^\nu-U^{2\nu}-u^0_{\mathrm{ref}}\|_{L^2}$")
    ax.loglog(nu_vals, data["post_l2"], "o-", label=r"$\|U^{\mathrm{post}}-u^0_{\mathrm{ref}}\|_{L^2}$")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$L^2$ error")
    ax.set_title(r"$L^2$ error vs $\nu$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    _, ax2_ = plt.subplots(figsize=(8, 5))
    ax2 = cast(Axes, ax2_)
    ax2.loglog(nu_vals, data["nu_inv_linf"], "o-", label=r"$\|U^\nu-u^0_{\mathrm{ref}}\|_{L^\infty}$")
    ax2.loglog(nu_vals, data["rich_linf"], "o-", label=r"$\|2U^\nu-U^{2\nu}-u^0_{\mathrm{ref}}\|_{L^\infty}$")
    ax2.loglog(nu_vals, data["post_linf"], "o-", label=r"$\|U^{\mathrm{post}}-u^0_{\mathrm{ref}}\|_{L^\infty}$")
    ax2.set_xlabel(r"$\nu$")
    ax2.set_ylabel(r"$L^\infty$ error")
    ax2.set_title(r"$L^\infty$ error vs $\nu$")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()


# ----------------------------
# Studies
# ----------------------------

def run_dx_study(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dx_vals_list: list[float] = []
    data_lists: dict[str, list[float]] = {
        "nu_inv_l2": [],
        "nu_inv_linf": [],
        "rich_l2": [],
        "rich_linf": [],
        "post_l2": [],
        "post_linf": [],
    }

    if args.exact != "none":
        data_lists["nu_visc_l2"] = []
        data_lists["nu_visc_linf"] = []

    for N in args.N_list:
        x = np.linspace(0.0, args.length, N, endpoint=False)
        u0 = make_initial_condition(x, ic_type=args.ic)

        run = compute_errors_for_run(
            u0=u0,
            x=x,
            nu=args.nu,
            tfinal=args.tfinal,
            flux_type=args.flux,
            mode=args.mode,
            S0=args.S0,
            p=args.p,
            resolved_cells_threshold=args.resolved_cells_threshold,
            steep_fraction_threshold=args.steep_fraction_threshold,
            exact_kind=args.exact,
            ref_factor=args.reference_factor,
            nu_small_ref=args.nu_small_ref,
            cfl_adv=args.cfl_adv,
            cfl_diff=args.cfl_diff,
        )

        dx_vals_list.append(float(run["dx"]))
        data_lists["nu_inv_l2"].append(float(run["err_nu_inv_l2"]))
        data_lists["nu_inv_linf"].append(float(run["err_nu_inv_linf"]))
        data_lists["rich_l2"].append(float(run["err_rich_l2"]))
        data_lists["rich_linf"].append(float(run["err_rich_linf"]))
        data_lists["post_l2"].append(float(run["err_post_l2"]))
        data_lists["post_linf"].append(float(run["err_post_linf"]))

        if args.exact != "none":
            data_lists["nu_visc_l2"].append(float(run["err_nu_l2"]))
            data_lists["nu_visc_linf"].append(float(run["err_nu_linf"]))

    dx_vals = np.asarray(dx_vals_list, dtype=float)
    data = {k: np.asarray(v, dtype=float) for k, v in data_lists.items()}

    print("\nDX study slopes:")
    if args.exact != "none":
        print(f"  viscous target, L2 slope   : {estimate_loglog_slope(dx_vals, data['nu_visc_l2']): .4f}")
        print(f"  viscous target, Linf slope : {estimate_loglog_slope(dx_vals, data['nu_visc_linf']): .4f}")
    print(f"  U^nu vs inv ref, L2 slope  : {estimate_loglog_slope(dx_vals, data['nu_inv_l2']): .4f}")
    print(f"  U^nu vs inv ref, Linf slope: {estimate_loglog_slope(dx_vals, data['nu_inv_linf']): .4f}")
    print(f"  rich vs inv ref, L2 slope  : {estimate_loglog_slope(dx_vals, data['rich_l2']): .4f}")
    print(f"  rich vs inv ref, Linf slope: {estimate_loglog_slope(dx_vals, data['rich_linf']): .4f}")
    print(f"  post vs inv ref, L2 slope  : {estimate_loglog_slope(dx_vals, data['post_l2']): .4f}")
    print(f"  post vs inv ref, Linf slope: {estimate_loglog_slope(dx_vals, data['post_linf']): .4f}")

    return dx_vals, data


def run_nu_study(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.linspace(0.0, args.length, args.N, endpoint=False)
    u0 = make_initial_condition(x, ic_type=args.ic)

    data_lists: dict[str, list[float]] = {
        "nu_inv_l2": [],
        "nu_inv_linf": [],
        "rich_l2": [],
        "rich_linf": [],
        "post_l2": [],
        "post_linf": [],
    }

    for nu in args.nu_list:
        run = compute_errors_for_run(
            u0=u0,
            x=x,
            nu=nu,
            tfinal=args.tfinal,
            flux_type=args.flux,
            mode=args.mode,
            S0=args.S0,
            p=args.p,
            resolved_cells_threshold=args.resolved_cells_threshold,
            steep_fraction_threshold=args.steep_fraction_threshold,
            exact_kind="none",
            ref_factor=args.reference_factor,
            nu_small_ref=args.nu_small_ref,
            cfl_adv=args.cfl_adv,
            cfl_diff=args.cfl_diff,
        )

        data_lists["nu_inv_l2"].append(float(run["err_nu_inv_l2"]))
        data_lists["nu_inv_linf"].append(float(run["err_nu_inv_linf"]))
        data_lists["rich_l2"].append(float(run["err_rich_l2"]))
        data_lists["rich_linf"].append(float(run["err_rich_linf"]))
        data_lists["post_l2"].append(float(run["err_post_l2"]))
        data_lists["post_linf"].append(float(run["err_post_linf"]))

    data = {k: np.asarray(v, dtype=float) for k, v in data_lists.items()}
    nu_vals = np.asarray(args.nu_list, dtype=float)

    print("\nNU study slopes:")
    print(f"  U^nu vs inv ref, L2 slope  : {estimate_loglog_slope(nu_vals, data['nu_inv_l2']): .4f}")
    print(f"  U^nu vs inv ref, Linf slope: {estimate_loglog_slope(nu_vals, data['nu_inv_linf']): .4f}")
    print(f"  rich vs inv ref, L2 slope  : {estimate_loglog_slope(nu_vals, data['rich_l2']): .4f}")
    print(f"  rich vs inv ref, Linf slope: {estimate_loglog_slope(nu_vals, data['rich_linf']): .4f}")
    print(f"  post vs inv ref, L2 slope  : {estimate_loglog_slope(nu_vals, data['post_l2']): .4f}")
    print(f"  post vs inv ref, Linf slope: {estimate_loglog_slope(nu_vals, data['post_linf']): .4f}")

    return nu_vals, data


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Viscous Burgers with viscosity extrapolation, exact/reference comparisons, and nu/dx studies."
    )

    parser.add_argument("--length", type=float, default=2.0 * np.pi, help="Domain length.")
    parser.add_argument("--N", type=int, default=800, help="Grid points for main run / nu study.")
    parser.add_argument(
        "--N-list",
        type=parse_int_list,
        default=[100, 200, 400, 800],
        help="Comma-separated N list for dx study, e.g. 100,200,400,800",
    )
    parser.add_argument("--nu", type=float, default=2.5e-3, help="Base viscosity for main run / dx study.")
    parser.add_argument(
        "--nu-list",
        type=parse_float_list,
        default=[1e-2, 5e-3, 2.5e-3, 1.25e-3],
        help="Comma-separated nu list for nu study.",
    )
    parser.add_argument("--tfinal", type=float, default=0.4, help="Final time.")
    parser.add_argument(
        "--ic",
        type=str,
        default="sine",
        choices=["sine", "sine2", "sine-mean", "bump", "step"],
        help="Initial condition.",
    )
    parser.add_argument(
        "--flux",
        type=str,
        default="godunov",
        choices=["rusanov", "godunov", "central"],
        help="Convective flux.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "always", "none"],
        help="Extrapolation mode.",
    )
    parser.add_argument(
        "--exact",
        type=str,
        default="reference",
        choices=["none", "reference", "colehopf"],
        help="Viscous exact/reference target for U^nu.",
    )
    parser.add_argument(
        "--study",
        type=str,
        default="both",
        choices=["none", "dx", "nu", "both"],
        help="Run error studies.",
    )
    parser.add_argument("--S0", type=float, default=0.08, help="Blending sensor threshold.")
    parser.add_argument("--p", type=float, default=4.0, help="Blending power.")
    parser.add_argument(
        "--resolved-cells-threshold",
        type=float,
        default=4.0,
        help="Auto-mode threshold for cells across viscous layer.",
    )
    parser.add_argument(
        "--steep-fraction-threshold",
        type=float,
        default=0.02,
        help="Steep-region threshold as fraction of max one-cell jump.",
    )
    parser.add_argument(
        "--reference-factor",
        type=int,
        default=8,
        help="Refinement factor for reference solutions.",
    )
    parser.add_argument(
        "--nu-small-ref",
        type=float,
        default=1e-5,
        help="Tiny viscosity for inviscid reference.",
    )
    parser.add_argument("--cfl-adv", type=float, default=0.4, help="Advection CFL.")
    parser.add_argument("--cfl-diff", type=float, default=0.4, help="Diffusion CFL.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots.")

    return parser.parse_args()


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()

    x = np.linspace(0.0, args.length, args.N, endpoint=False)
    u0 = make_initial_condition(x, ic_type=args.ic)

    run = compute_errors_for_run(
        u0=u0,
        x=x,
        nu=args.nu,
        tfinal=args.tfinal,
        flux_type=args.flux,
        mode=args.mode,
        S0=args.S0,
        p=args.p,
        resolved_cells_threshold=args.resolved_cells_threshold,
        steep_fraction_threshold=args.steep_fraction_threshold,
        exact_kind=args.exact,
        ref_factor=args.reference_factor,
        nu_small_ref=args.nu_small_ref,
        cfl_adv=args.cfl_adv,
        cfl_diff=args.cfl_diff,
    )

    print("Main run parameters:")
    print(f"  N                         : {args.N}")
    print(f"  nu                        : {args.nu}")
    print(f"  tfinal                    : {args.tfinal}")
    print(f"  ic                        : {args.ic}")
    print(f"  flux                      : {args.flux}")
    print(f"  mode                      : {args.mode}")
    print(f"  exact target              : {args.exact}")
    print("Extrapolation diagnostics:")
    for k, v in cast(dict[str, Any], run["info"]).items():
        print(f"  {k:26s}: {v}")

    if args.exact != "none":
        print("\nMain run norm errors:")
        print(f"  ||U^nu - viscous target||_L2   = {cast(float, run['err_nu_l2']):.6e}")
        print(f"  ||U^nu - viscous target||_Linf = {cast(float, run['err_nu_linf']):.6e}")

    print(f"  ||U^nu - inviscid ref||_L2     = {cast(float, run['err_nu_inv_l2']):.6e}")
    print(f"  ||U^nu - inviscid ref||_Linf   = {cast(float, run['err_nu_inv_linf']):.6e}")
    print(f"  ||rich - inviscid ref||_L2     = {cast(float, run['err_rich_l2']):.6e}")
    print(f"  ||rich - inviscid ref||_Linf   = {cast(float, run['err_rich_linf']):.6e}")
    print(f"  ||post - inviscid ref||_L2     = {cast(float, run['err_post_l2']):.6e}")
    print(f"  ||post - inviscid ref||_Linf   = {cast(float, run['err_post_linf']):.6e}")

    dx_vals: np.ndarray | None = None
    dx_data: dict[str, np.ndarray] | None = None
    nu_vals: np.ndarray | None = None
    nu_data: dict[str, np.ndarray] | None = None

    if args.study in ("dx", "both"):
        dx_vals, dx_data = run_dx_study(args)

    if args.study in ("nu", "both"):
        nu_vals, nu_data = run_nu_study(args)

    if not args.no_plots:
        plot_main_run(x, u0, run, args.exact)

        if dx_vals is not None and dx_data is not None:
            plot_dx_study(dx_vals, dx_data, args.exact)

        if nu_vals is not None and nu_data is not None:
            plot_nu_study(nu_vals, nu_data)

        plt.show()


if __name__ == "__main__":
    main()
    

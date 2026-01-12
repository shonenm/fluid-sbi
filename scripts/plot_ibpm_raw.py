#!/usr/bin/env python3
"""
IBPM生データの可視化スクリプト

Tecplot形式のIBPM出力を読み込んで、速度場と渦度を可視化します。
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_tecplot_file(filepath):
    """
    Tecplot ASCII形式のファイルを読み込む

    Returns:
        x, y, u, v, vorticity の5つの配列 (各199x199)
    """
    with open(filepath) as f:
        lines = f.readlines()

    # ヘッダーをスキップしてデータ部分を抽出
    data_start = 0
    for i, line in enumerate(lines):
        if "DATAPACKING" in line:
            data_start = i + 2  # DT行の次から
            break

    # データを読み込み
    data = []
    for line in lines[data_start:]:
        if line.strip():
            values = line.strip().split()
            if len(values) == 5:
                data.append([float(v) for v in values])

    data = np.array(data)

    # データを抽出
    x_flat = data[:, 0]
    y_flat = data[:, 1]
    u_flat = data[:, 2]
    v_flat = data[:, 3]
    vort_flat = data[:, 4]

    # グリッドサイズを推定
    nx = len(np.unique(x_flat))
    ny = len(np.unique(y_flat))

    # 2D配列に変形
    x = x_flat.reshape(ny, nx)
    y = y_flat.reshape(ny, nx)
    u = u_flat.reshape(ny, nx)
    v = v_flat.reshape(ny, nx)
    vort = vort_flat.reshape(ny, nx)

    return x, y, u, v, vort


def plot_single_timestep(filepath, output_dir=None, show=True):
    """
    単一タイムステップのデータを可視化
    """
    print(f"Reading {filepath.name}...")
    x, y, u, v, vort = read_tecplot_file(filepath)

    # タイムステップを抽出
    match = re.search(r"ibpm(\d+)\.plt", filepath.name)
    if match:
        timestep = int(match.group(1))
        time = timestep * 0.02  # dt = 0.02
    else:
        timestep = 0
        time = 0.0

    # 速度の大きさ
    speed = np.sqrt(u**2 + v**2)

    # 図を作成
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"IBPM Raw Data - Step {timestep:05d} (t = {time:.2f})", fontsize=16, fontweight="bold")

    # (0,0) 速度の大きさ
    ax = axes[0, 0]
    im0 = ax.contourf(x, y, speed, levels=50, cmap="jet")
    ax.contour(x, y, speed, levels=10, colors="k", linewidths=0.3, alpha=0.3)
    # 円柱の位置を示す
    circle = plt.Circle((0, 0), 0.5, color="white", fill=True, edgecolor="black", linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Velocity Magnitude |u|", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.colorbar(im0, ax=ax, label="|u|")

    # (0,1) u成分
    ax = axes[0, 1]
    im1 = ax.contourf(x, y, u, levels=50, cmap="RdBu_r")
    ax.contour(x, y, u, levels=10, colors="k", linewidths=0.3, alpha=0.3)
    circle = plt.Circle((0, 0), 0.5, color="gray", fill=True, edgecolor="black", linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("u-velocity", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax, label="u")

    # (1,0) v成分
    ax = axes[1, 0]
    im2 = ax.contourf(x, y, v, levels=50, cmap="RdBu_r")
    ax.contour(x, y, v, levels=10, colors="k", linewidths=0.3, alpha=0.3)
    circle = plt.Circle((0, 0), 0.5, color="gray", fill=True, edgecolor="black", linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("v-velocity", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax, label="v")

    # (1,1) 渦度
    ax = axes[1, 1]
    vort_levels = np.linspace(-5, 5, 51)
    im3 = ax.contourf(x, y, vort, levels=vort_levels, cmap="RdBu_r", extend="both")
    ax.contour(x, y, vort, levels=10, colors="k", linewidths=0.3, alpha=0.3)
    circle = plt.Circle((0, 0), 0.5, color="gray", fill=True, edgecolor="black", linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Vorticity ω = ∂v/∂x - ∂u/∂y", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.colorbar(im3, ax=ax, label="ω")

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"ibpm_raw_{timestep:05d}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_force_coefficients(force_file, output_file=None, show=True):
    """
    力係数の時系列をプロット
    """
    print(f"Reading force data from {force_file}...")
    data = np.loadtxt(force_file)

    step = data[:, 0]
    time = data[:, 1]
    fx = data[:, 2]
    fy = data[:, 3]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("IBPM Force Coefficients", fontsize=16, fontweight="bold")

    # X方向力（抗力）
    ax = axes[0]
    ax.plot(time, fx, "b-", linewidth=2, label="Fx (Drag)")
    ax.axhline(y=fx[50:].mean(), color="r", linestyle="--", linewidth=1.5, label=f"Mean (t>1.0): {fx[50:].mean():.4f}")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Fx", fontsize=12)
    ax.set_title("X-direction Force (Drag)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Y方向力（揚力）
    ax = axes[1]
    ax.plot(time, fy, "g-", linewidth=2, label="Fy (Lift)")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Fy", fontsize=12)
    ax.set_title("Y-direction Force (Lift)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # 統計情報を追加
    textstr = "Statistics (t > 1.0):\n"
    textstr += f"Fx: {fx[50:].mean():.4f} ± {fx[50:].std():.4f}\n"
    textstr += f"Fy: {fy[50:].mean():.2e} ± {fy[50:].std():.2e}"
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_timesteps(data_dir, timesteps, output_dir=None, show=True):
    """
    複数のタイムステップを並べて表示
    """
    data_dir = Path(data_dir)
    n_steps = len(timesteps)

    fig, axes = plt.subplots(n_steps, 3, figsize=(18, 6 * n_steps))
    if n_steps == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("IBPM Flow Evolution", fontsize=16, fontweight="bold")

    for i, step in enumerate(timesteps):
        filepath = data_dir / f"ibpm{step:05d}.plt"
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue

        print(f"Reading step {step}...")
        x, y, u, v, vort = read_tecplot_file(filepath)
        time = step * 0.02
        speed = np.sqrt(u**2 + v**2)

        # 速度の大きさ
        ax = axes[i, 0]
        im0 = ax.contourf(x, y, speed, levels=30, cmap="jet")
        circle = plt.Circle((0, 0), 0.5, color="white", fill=True, edgecolor="black", linewidth=2)
        ax.add_patch(circle)
        ax.set_ylabel(f"t = {time:.2f}", fontsize=12, fontweight="bold")
        if i == 0:
            ax.set_title("Velocity Magnitude", fontsize=14)
        ax.set_aspect("equal")
        plt.colorbar(im0, ax=ax)

        # u成分
        ax = axes[i, 1]
        im1 = ax.contourf(x, y, u, levels=30, cmap="RdBu_r")
        circle = plt.Circle((0, 0), 0.5, color="gray", fill=True, edgecolor="black", linewidth=2)
        ax.add_patch(circle)
        if i == 0:
            ax.set_title("u-velocity", fontsize=14)
        ax.set_aspect("equal")
        plt.colorbar(im1, ax=ax)

        # 渦度
        ax = axes[i, 2]
        vort_levels = np.linspace(-5, 5, 31)
        im2 = ax.contourf(x, y, vort, levels=vort_levels, cmap="RdBu_r", extend="both")
        circle = plt.Circle((0, 0), 0.5, color="gray", fill=True, edgecolor="black", linewidth=2)
        ax.add_patch(circle)
        if i == 0:
            ax.set_title("Vorticity", fontsize=14)
        ax.set_aspect("equal")
        plt.colorbar(im2, ax=ax)

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "ibpm_evolution.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Plot IBPM raw data")
    parser.add_argument(
        "--data-dir", type=str, default="/workspace/data/ibpm_full", help="Directory containing IBPM output files"
    )
    parser.add_argument("--output-dir", type=str, default="/workspace/data/ibpm_plots", help="Directory to save plots")
    parser.add_argument("--timestep", type=int, nargs="+", default=None, help="Specific timestep(s) to plot")
    parser.add_argument("--evolution", action="store_true", help="Plot flow evolution at multiple timesteps")
    parser.add_argument("--force", action="store_true", help="Plot force coefficients")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots (only save)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    # 力係数のプロット
    if args.force:
        force_file = data_dir / "ibpm.force"
        if force_file.exists():
            plot_force_coefficients(force_file, output_file=output_dir / "force_coefficients.png", show=show)
        else:
            print(f"Force file not found: {force_file}")

    # 流れの発展を表示
    if args.evolution:
        timesteps = [0, 50, 100, 150, 200, 250]
        plot_multiple_timesteps(data_dir, timesteps, output_dir=output_dir, show=show)

    # 特定のタイムステップをプロット
    if args.timestep:
        for step in args.timestep:
            filepath = data_dir / f"ibpm{step:05d}.plt"
            if filepath.exists():
                plot_single_timestep(filepath, output_dir=output_dir, show=show)
            else:
                print(f"File not found: {filepath}")

    # デフォルト：代表的なタイムステップをプロット
    if not args.force and not args.evolution and not args.timestep:
        print("\n=== Plotting IBPM Raw Data ===")
        print(f"Output directory: {output_dir}\n")

        # 力係数をプロット
        print("1. Force coefficients...")
        force_file = data_dir / "ibpm.force"
        if force_file.exists():
            plot_force_coefficients(force_file, output_file=output_dir / "force_coefficients.png", show=show)

        # 流れの発展をプロット
        print("\n2. Flow evolution (t=0, 1, 2, 3, 4, 5)...")
        timesteps = [0, 50, 100, 150, 200, 250]
        plot_multiple_timesteps(data_dir, timesteps, output_dir=output_dir, show=show)

        # 代表的なタイムステップの詳細をプロット
        print("\n3. Detailed views (t=0, 1, 3, 5)...")
        for step in [0, 50, 150, 250]:
            filepath = data_dir / f"ibpm{step:05d}.plt"
            if filepath.exists():
                plot_single_timestep(filepath, output_dir=output_dir, show=show)

        print(f"\n=== All plots saved to {output_dir} ===")


if __name__ == "__main__":
    main()

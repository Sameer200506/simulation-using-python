"""
core/geometry.py — Functions that produce boolean masks on a Grid.

Each function takes a Grid and geometric parameters and returns a boolean
ndarray (shape = grid.shape) marking which cells belong to the region.

Keeping geometry separate from the grid means we can compose shapes:
    mask = mark_cylinder(...) & ~mark_cylinder(...)   # hollow cylinder
"""
import numpy as np
from .grid import Grid


def mark_cylinder(
    grid: Grid,
    radius: float,
    z_bottom: float = None,
    z_top: float = None,
    cx: float = 0.0,
    cy: float = 0.0,
) -> np.ndarray:
    """
    Return a boolean mask for all cells whose centre falls inside a
    vertical (z-axis) cylinder.

    Parameters
    ----------
    grid      : Grid instance
    radius    : cylinder radius [m]
    z_bottom  : lower z-limit [m]; defaults to grid.z_min
    z_top     : upper z-limit [m]; defaults to grid.z_max
    cx, cy    : cylinder axis x/y position [m]; defaults to domain centre
    """
    z_bottom = z_bottom if z_bottom is not None else grid.z_min
    z_top    = z_top    if z_top    is not None else grid.z_max

    r_sq    = (grid.XX - cx) ** 2 + (grid.YY - cy) ** 2
    in_r    = r_sq <= radius ** 2
    in_z    = (grid.ZZ >= z_bottom) & (grid.ZZ <= z_top)
    return in_r & in_z


def mark_box(
    grid: Grid,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    z_lo: float, z_hi: float,
) -> np.ndarray:
    """Return a boolean mask for all cells inside an axis-aligned box."""
    return (
        (grid.XX >= x_lo) & (grid.XX <= x_hi) &
        (grid.YY >= y_lo) & (grid.YY <= y_hi) &
        (grid.ZZ >= z_lo) & (grid.ZZ <= z_hi)
    )


def mark_annulus(
    grid: Grid,
    r_inner: float,
    r_outer: float,
    z_bottom: float = None,
    z_top: float = None,
    cx: float = 0.0,
    cy: float = 0.0,
) -> np.ndarray:
    """Return mask for a hollow cylinder (annulus between r_inner and r_outer)."""
    outer = mark_cylinder(grid, r_outer, z_bottom, z_top, cx, cy)
    inner = mark_cylinder(grid, r_inner, z_bottom, z_top, cx, cy)
    return outer & ~inner

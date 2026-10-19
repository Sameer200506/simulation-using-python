"""
geometry.py
-----------
Build the 2-D cross-section geometry for four 26650 cylindrical Li-ion
batteries arranged in a 2×2 grid inside a rectangular PCM block.

Coordinate system : x (width), y (height).  All lengths in metres.

Physical-group IDs
──────────────────
  Surfaces (dim=2)  1–4  → Battery_1 … Battery_4
                    5    → PCM domain
  Curves   (dim=1)  20   → outer PCM boundary  (Robin / convection BC)
"""

import numpy as np
import gmsh


# ── Physical-group tags ───────────────────────────────────────────────────────
# These integer labels are shared across geometry / mesh / solver / postprocess.
BAT_TAGS     = [1, 2, 3, 4]   # one tag per cell
PCM_TAG      = 5
OUTER_BC_TAG = 20

# ── Geometry constants [m] ────────────────────────────────────────────────────
BAT_RADIUS = 0.013   # 26650 cell radius = 13 mm
BAT_PITCH  = 0.030   # centre-to-centre spacing (30 mm)
PCM_WIDTH  = 0.100   # 100 mm
PCM_HEIGHT = 0.100   # 100 mm


def build_geometry(mesh_size_bat: float = 0.0015,
                   mesh_size_pcm: float = 0.004) -> None:
    """
    Initialise gmsh, construct the composite geometry and generate a 2-D mesh.

    The mesh is left in the gmsh model (not finalised here) so that the caller
    can convert it to DOLFINx via gmshio.model_to_mesh before finalising gmsh.

    Parameters
    ----------
    mesh_size_bat : target element size inside / near batteries [m].
    mesh_size_pcm : target element size in the PCM bulk [m].
    """
    gmsh.initialize()
    gmsh.model.add("pcm_battery_2d")
    gmsh.option.setNumber("General.Verbosity", 2)   # suppress most messages

    # ── Battery centre positions (2×2 symmetric grid) ─────────────────────────
    cx     = PCM_WIDTH  / 2
    cy     = PCM_HEIGHT / 2
    half_p = BAT_PITCH  / 2

    centers = [
        (cx - half_p, cy - half_p),   # Battery 1 – bottom-left
        (cx + half_p, cy - half_p),   # Battery 2 – bottom-right
        (cx - half_p, cy + half_p),   # Battery 3 – top-left
        (cx + half_p, cy + half_p),   # Battery 4 – top-right
    ]

    # ── Primitive surfaces (OpenCASCADE kernel) ───────────────────────────────
    rect_tag  = gmsh.model.occ.addRectangle(0, 0, 0, PCM_WIDTH, PCM_HEIGHT)
    disk_tags = [
        gmsh.model.occ.addDisk(x, y, 0, BAT_RADIUS, BAT_RADIUS)
        for (x, y) in centers
    ]

    # ── Boolean fragment: produces five conforming, non-overlapping surfaces ──
    # The rectangle is split by the four circles.  Edges at battery–PCM
    # interfaces are shared → the resulting mesh has no hanging nodes.
    gmsh.model.occ.fragment(
        [(2, rect_tag)],
        [(2, d) for d in disk_tags],
    )
    gmsh.model.occ.synchronize()

    # ── Identify resulting surfaces by centroid proximity ────────────────────
    bat_surf: dict[int, int] = {}   # battery-index → gmsh surface tag
    pcm_surfs: list[int]     = []

    for _, tag in gmsh.model.getEntities(2):
        x, y, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        is_bat = False
        for i, (bx, by) in enumerate(centers):
            # A battery disk's centroid must lie within half its own radius
            if np.hypot(x - bx, y - by) < BAT_RADIUS * 0.5:
                bat_surf[i] = tag
                is_bat = True
                break
        if not is_bat:
            pcm_surfs.append(tag)

    if len(bat_surf) != 4:
        raise RuntimeError(
            f"Expected 4 battery surfaces after fragment, found {len(bat_surf)}."
            " Check battery positions / PCM block size."
        )

    # ── Physical groups: surfaces ─────────────────────────────────────────────
    for i, phys_tag in enumerate(BAT_TAGS):
        gmsh.model.addPhysicalGroup(2, [bat_surf[i]], phys_tag,
                                    name=f"Battery_{i + 1}")
    gmsh.model.addPhysicalGroup(2, pcm_surfs, PCM_TAG, name="PCM")

    # ── Outer boundary curves ─────────────────────────────────────────────────
    # After fragment, a curve adjacent to exactly ONE surface is a boundary
    # curve (outer PCM edge).  Curves adjacent to TWO surfaces are internal
    # battery–PCM interface arcs → no physical group, no BC needed there.
    outer_curves: list[int] = []
    for _, ctag in gmsh.model.getEntities(1):
        adj_surfs, _ = gmsh.model.getAdjacencies(1, ctag)
        if len(adj_surfs) == 1:
            outer_curves.append(ctag)

    gmsh.model.addPhysicalGroup(1, outer_curves, OUTER_BC_TAG,
                                name="OuterBoundary")

    # ── Mesh-size field: finer elements near battery surfaces ─────────────────
    fld = gmsh.model.mesh.field

    # Collect all curves that bound the four battery disks
    bat_bound_curves: list[int] = []
    for i in range(4):
        _, crv_tags = gmsh.model.getAdjacencies(2, bat_surf[i])
        bat_bound_curves.extend(int(c) for c in crv_tags)

    dist_id = fld.add("Distance")
    fld.setNumbers(dist_id, "CurvesList", bat_bound_curves)
    fld.setNumber(dist_id, "Sampling", 150)

    # Ramp element size from mesh_size_bat (at battery surface)
    # to mesh_size_pcm (far from batteries)
    thr_id = fld.add("Threshold")
    fld.setNumber(thr_id, "InField",  dist_id)
    fld.setNumber(thr_id, "SizeMin",  mesh_size_bat)
    fld.setNumber(thr_id, "SizeMax",  mesh_size_pcm)
    fld.setNumber(thr_id, "DistMin",  BAT_RADIUS * 0.3)
    fld.setNumber(thr_id, "DistMax",  BAT_RADIUS * 4.0)
    fld.setAsBackgroundMesh(thr_id)

    # Disable default element-size scaling from bounding box
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature",      0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints",         0)

    # ── Generate and optimise 2-D linear mesh ─────────────────────────────────
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)               # linear (P1) triangles
    gmsh.model.mesh.optimize("Netgen")        # improve element quality

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    print(f"[geometry] Mesh generated: {n_nodes} nodes, "
          f"{len(gmsh.model.getEntities(2))} surfaces")


# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    "BAT_TAGS", "PCM_TAG", "OUTER_BC_TAG",
    "BAT_RADIUS", "BAT_PITCH", "PCM_WIDTH", "PCM_HEIGHT",
    "build_geometry",
]

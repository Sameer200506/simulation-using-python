"""
mesh.py
-------
Convert the gmsh model to a DOLFINx mesh and save it as XDMF.

Workflow
--------
1. Call geometry.build_geometry() – populates the gmsh model with geometry
   and a 2-D mesh.
2. Convert via dolfinx.io.gmshio.model_to_mesh (rank-0 read).
3. Finalise gmsh (frees C++ resources).
4. Write mesh + tags to XDMF for inspection in ParaView.

Returns
-------
msh        : dolfinx.mesh.Mesh           – 2-D computational mesh
cell_tags  : dolfinx.mesh.MeshTags      – subdomain markers (dim = 2)
facet_tags : dolfinx.mesh.MeshTags      – boundary markers  (dim = 1)
"""

from pathlib import Path

import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile


def generate_mesh(
    output_dir:    Path,
    mesh_size_bat: float = 0.0015,
    mesh_size_pcm: float = 0.004,
):
    """
    Build the geometry, generate the mesh and convert it to DOLFINx format.

    Parameters
    ----------
    output_dir    : directory where mesh.xdmf is written.
    mesh_size_bat : characteristic element size at battery surfaces [m].
    mesh_size_pcm : characteristic element size in PCM bulk [m].

    Returns
    -------
    (msh, cell_tags, facet_tags)
    """
    from geometry import build_geometry   # explicit local import

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build geometry and mesh inside gmsh ───────────────────────────
    build_geometry(mesh_size_bat=mesh_size_bat,
                   mesh_size_pcm=mesh_size_pcm)

    # ── Step 2: Convert to DOLFINx ────────────────────────────────────────────
    # rank=0  → only MPI rank 0 reads the gmsh model; mesh is broadcast.
    # gdim=2  → embed the 2-D geometry in 2-D coordinate space (not 3-D).
    msh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model,
        MPI.COMM_WORLD,
        rank=0,
        gdim=2,
    )
    gmsh.finalize()   # release gmsh resources; DOLFINx mesh is self-contained

    msh.name        = "pcm_battery"
    cell_tags.name  = "subdomains"
    facet_tags.name = "boundaries"

    # ── Step 3: Create required connectivity for facet tags ───────────────────
    # DOLFINx needs facet → cell adjacency to write facet MeshTags to XDMF.
    tdim = msh.topology.dim       # = 2
    msh.topology.create_connectivity(tdim - 1, tdim)   # facet → cell
    msh.topology.create_connectivity(tdim, tdim - 1)   # cell  → facet

    # ── Step 4: Save for ParaView ─────────────────────────────────────────────
    mesh_path = output_dir / "mesh.xdmf"
    with XDMFFile(msh.comm, str(mesh_path), "w") as xf:
        xf.write_mesh(msh)
        xf.write_meshtags(cell_tags,  msh.geometry)
        xf.write_meshtags(facet_tags, msh.geometry)

    n_cells = msh.topology.index_map(tdim).size_global
    print(f"[mesh] Mesh written  → {mesh_path}")
    print(f"[mesh] Total cells   : {n_cells}")
    print(f"[mesh] Cell-tag vals : {set(cell_tags.values.tolist())}")
    print(f"[mesh] Facet-tag vals: {set(facet_tags.values.tolist())}")

    return msh, cell_tags, facet_tags

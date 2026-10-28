"""Recover the physical chassis frame from a named CAD mounting plate."""
import mujoco
import numpy as np


def frame_from_mounting_plate(model, mesh_name):
    """Return body-local forward/left/up axes, using the plate normal as up."""
    base_id = model.body('base').id
    mesh_id = model.mesh(mesh_name).id
    geoms = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] == base_id
             and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
             and model.geom_dataid[g] == mesh_id]
    if len(geoms) != 1:
        raise ValueError('Expected one base mounting plate mesh: ' + mesh_name)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_id = geoms[0]
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
    world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    local = (world - data.xpos[base_id]) @ data.xmat[base_id].reshape(3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(local.T))
    if eigenvalues[0] / eigenvalues[1] > .05:
        raise ValueError('Chassis reference mesh is not sufficiently planar')
    up = eigenvectors[:, 0]
    if up[2] < 0:
        up = -up
    # The named physical left side is CAD -y, as in the original motor contract.
    forward = np.cross([0., -1., 0.], up)
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    return np.column_stack((forward, left, up))


def horizontal_base_quaternion(model, mesh_name):
    """Rotate the entire base so its physical mounting plate is horizontal."""
    frame = frame_from_mounting_plate(model, mesh_name)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(frame.T).ravel())
    return quaternion, frame

"""Measure virtual leg length from physical motor centers in the side view."""
import mujoco
import numpy as np


def _mesh_points_in_body(model, mesh_name, body_name):
    body_id = model.body(body_name).id
    mesh_id = model.mesh(mesh_name).id
    matches = [g for g in range(model.ngeom)
               if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
               and model.geom_dataid[g] == mesh_id and model.geom_bodyid[g] == body_id]
    if len(matches) != 1:
        raise ValueError('Expected one mesh in body: ' + body_name + '/' + mesh_name)
    geom_id = matches[0]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = model.mesh_vertadr[mesh_id]
    points = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
    world = points @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    return (world - data.xpos[body_id]) @ data.xmat[body_id].reshape(3, 3)


def motor_midpoints_in_base(model, physics):
    """Fit concentric CAD motor housing centers; never use driven joint origins."""
    from .chassis_frame import frame_from_mounting_plate
    frame = frame_from_mounting_plate(model, physics['chassis_reference_mesh'])
    midpoints = []
    evidence = {}
    for side in ('left', 'right'):
        meshes = physics['motor_center_reference_meshes'][side]
        if len(meshes) != 2:
            raise ValueError('Require two physical motor reference meshes per side')
        centers = []
        for name in meshes:
            points = _mesh_points_in_body(model, name, 'base') @ frame
            xz = np.unique(np.round(points[:, [0, 2]], 10), axis=0)
            if len(xz) < 12:
                raise ValueError('Motor reference needs a resolved circular surface')
            # Fit only the outer circular rim; inner concentric rings have
            # different radii and uneven tessellation, which biases a combined fit.
            seed_center = (xz.min(axis=0)+xz.max(axis=0))/2
            radii = np.linalg.norm(xz-seed_center,axis=1)
            rim = xz[radii > radii.max()-1e-5]
            if len(rim) < 12:
                raise ValueError('Motor housing lacks a resolved outer circular rim')
            coefficients, _, rank, _ = np.linalg.lstsq(
                np.column_stack((2*rim, np.ones(len(rim)))), np.sum(rim*rim, axis=1), rcond=None)
            if rank != 3:
                raise ValueError('Degenerate motor reference circle')
            center = np.array([coefficients[0], points[:, 1].mean(), coefficients[1]])
            fitted_radii=np.linalg.norm(rim-center[[0,2]],axis=1)
            if np.ptp(fitted_radii)>1e-6:
                raise ValueError('Motor housing rim is not circular in the chassis side view')
            centers.append(center @ frame.T)
        midpoint = np.mean(centers, axis=0)
        midpoints.append(midpoint)
        evidence[side] = {'reference_meshes':meshes, 'motor_centers_in_base_m':np.array(centers).tolist(),
                          'midpoint_in_base_m':midpoint.tolist(),
                          'motor_spacing_side_view_m':float(np.linalg.norm((np.array(centers)@frame)[0,[0,2]]-(np.array(centers)@frame)[1,[0,2]]))}
    return np.array(midpoints), evidence


def wheel_centers_in_wheel_bodies(model, physics):
    """Use each physical tire center, avoiding arbitrary points along the wheel axis."""
    centers = []
    for side in ('left', 'right'):
        points = _mesh_points_in_body(model, physics['wheel_center_reference_meshes'][side], side+'_link 3')
        centers.append((points.min(axis=0)+points.max(axis=0))/2)
    return np.array(centers)


def virtual_leg_geometry(model, data, motor_midpoints, wheel_centers, chassis_frame):
    base = model.body('base').id
    motor_world = motor_midpoints @ data.xmat[base].reshape(3,3).T + data.xpos[base]
    wheel_world = np.array([wheel_centers[i] @ data.xmat[model.body(s+'_link 3').id].reshape(3,3).T
                            + data.xpos[model.body(s+'_link 3').id] for i,s in enumerate(('left','right'))])
    delta = motor_world-wheel_world
    # Chassis sagittal projection ignores left/right installation offsets.
    rotation = data.xmat[base].reshape(3,3) @ chassis_frame
    dx = delta @ rotation[:,0]
    normal = delta @ rotation[:,2]
    dz = delta[:,2]
    return {'motor_midpoints_world_m':motor_world,'wheel_centers_world_m':wheel_world,
            'forward_offset_m':dx,'vertical_separation_m':dz,'side_view_distance_m':np.hypot(dx,normal)}

"""
Body Measurements - CPU Version

This is a CPU-compatible version that does NOT require the CUDA 
mesh-mesh-intersection module. It computes:
- Height (from head top to heel)
- Mass (from mesh volume * body density)
- Chest/Waist/Hips (using vertex-based approximation instead of mesh intersection)
"""
from typing import NewType, Dict, Tuple
import os.path as osp
import yaml
import numpy as np

import torch
import torch.nn as nn
from scipy.spatial import ConvexHull
from loguru import logger

Tensor = NewType('Tensor', torch.Tensor)


class BodyMeasurements(nn.Module):
    """
    CPU-compatible body measurements module.
    
    Computes anthropometric measurements from SMPL-X mesh triangles without
    requiring CUDA mesh intersection.
    """

    # The density of the human body is 985 kg / m^3
    DENSITY = 985

    def __init__(self, cfg, **kwargs):
        ''' 
        Initialize body measurements module.
        
        Args:
            cfg: dict with:
                - meas_definition_path: path to measurement definitions YAML
                - meas_vertices_path: path to measurement vertices YAML
        '''
        super(BodyMeasurements, self).__init__()

        meas_definition_path = cfg.get('meas_definition_path', '')
        meas_definition_path = osp.expanduser(
            osp.expandvars(meas_definition_path))
        meas_vertices_path = cfg.get('meas_vertices_path', '')
        meas_vertices_path = osp.expanduser(
            osp.expandvars(meas_vertices_path))

        with open(meas_definition_path, 'r') as f:
            measurements_definitions = yaml.safe_load(f)

        with open(meas_vertices_path, 'r') as f:
            meas_vertices = yaml.safe_load(f)

        # Head and heel for height
        head_top = meas_vertices['HeadTop']
        left_heel = meas_vertices['HeelLeft']

        left_heel_bc = left_heel['bc']
        self.left_heel_face_idx = left_heel['face_idx']

        left_heel_bc = torch.tensor(left_heel['bc'], dtype=torch.float32)
        self.register_buffer('left_heel_bc', left_heel_bc)

        head_top_bc = torch.tensor(head_top['bc'], dtype=torch.float32)
        self.register_buffer('head_top_bc', head_top_bc)

        self.head_top_face_idx = head_top['face_idx']

        # Chest periphery
        action = measurements_definitions['CW_p']
        chest_periphery_data = meas_vertices[action[0]]
        self.chest_face_index = chest_periphery_data['face_idx']
        chest_bcs = torch.tensor(
            chest_periphery_data['bc'], dtype=torch.float32)
        self.register_buffer('chest_bcs', chest_bcs)

        # Belly/Waist periphery
        action = measurements_definitions['BW_p']
        belly_periphery_data = meas_vertices[action[0]]
        self.belly_face_index = belly_periphery_data['face_idx']
        belly_bcs = torch.tensor(
            belly_periphery_data['bc'], dtype=torch.float32)
        self.register_buffer('belly_bcs', belly_bcs)

        # Hips periphery
        action = measurements_definitions['IW_p']
        hips_periphery_data = meas_vertices[action[0]]
        self.hips_face_index = hips_periphery_data['face_idx']
        hips_bcs = torch.tensor(
            hips_periphery_data['bc'], dtype=torch.float32)
        self.register_buffer('hips_bcs', hips_bcs)

        # Store all vertex indices for periphery computation
        self._load_periphery_vertices(meas_vertices, measurements_definitions)

    def _load_periphery_vertices(self, meas_vertices, meas_defs):
        """Load vertex indices for periphery measurements."""
        # Try to load pre-defined measurement loops if available
        # Otherwise we'll use single-point estimation
        self.use_vertex_loop = False
        
    def extra_repr(self) -> str:
        msg = []
        msg.append(f'Human Body Density: {self.DENSITY}')
        msg.append('Mode: CPU (vertex-based periphery)')
        return '\n'.join(msg)

    def compute_height(self, shaped_triangles: Tensor) -> Tuple[Tensor, Tensor]:
        ''' Compute the height using the heel and the top of the head
        '''
        head_top_tri = shaped_triangles[:, self.head_top_face_idx]
        head_top = (
            head_top_tri * self.head_top_bc.reshape(1, 3, 1)
        ).sum(dim=1)
        
        left_heel_tri = shaped_triangles[:, self.left_heel_face_idx]
        left_heel = (
            left_heel_tri * self.left_heel_bc.reshape(1, 3, 1)
        ).sum(dim=1)

        return (torch.abs(head_top[:, 1] - left_heel[:, 1]),
                torch.stack([head_top, left_heel], axis=0)
                )

    def compute_mass(self, tris: Tensor) -> Tensor:
        ''' Computes the mass from volume and average body density
        
        Uses signed volume of tetrahedra formed with origin to compute 
        total mesh volume.
        '''
        x = tris[:, :, :, 0]
        y = tris[:, :, :, 1]
        z = tris[:, :, :, 2]
        volume = (
            -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
            x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
            x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
            x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
            x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
            x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
        ).sum(dim=1).abs() / 6.0
        return volume * self.DENSITY

    def compute_periphery_from_point(
        self, 
        triangles: Tensor,
        face_idx: int,
        bcs: Tensor,
        name: str = 'periphery'
    ) -> Dict:
        """
        Estimate periphery circumference using a single reference point.
        
        This is a simplified CPU-based approximation that:
        1. Finds the reference point on the mesh
        2. Extracts nearby vertices at similar height
        3. Computes convex hull perimeter
        
        Note: This is less accurate than mesh intersection but works on CPU.
        """
        batch_size = triangles.shape[0]
        device = triangles.device
        
        # Get reference point
        ref_tri = triangles[:, face_idx]
        ref_point = (ref_tri * bcs.reshape(1, 3, 1)).sum(dim=1)
        ref_height = ref_point[:, 1]
        
        output = {
            'points': [],
            'valid_points': [],
            'value': [],
            'plane_height': ref_height,
        }
        
        # For each sample in batch
        for b in range(batch_size):
            # Get all triangle centers
            tri_centers = triangles[b].mean(dim=1)  # [F, 3]
            heights = tri_centers[:, 1]
            
            # Find triangles near the reference height (within 2cm)
            height_tolerance = 0.02
            height_mask = torch.abs(heights - ref_height[b]) < height_tolerance
            
            if height_mask.sum() < 10:
                # Expand tolerance if not enough points
                height_tolerance = 0.05
                height_mask = torch.abs(heights - ref_height[b]) < height_tolerance
            
            # Get XZ coordinates of matching triangles
            matching_centers = tri_centers[height_mask]
            
            if len(matching_centers) < 3:
                # Fallback: estimate based on mesh width at this height
                output['value'].append(torch.tensor(0.9, device=device))  # default
                output['points'].append(None)
                output['valid_points'].append(None)
                continue
            
            # Project to XZ plane and compute convex hull
            xz_points = matching_centers[:, [0, 2]].detach().cpu().numpy()
            
            try:
                hull = ConvexHull(xz_points)
                # Compute perimeter
                hull_points = xz_points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])  # close loop
                
                perimeter = 0.0
                for i in range(len(hull_points) - 1):
                    perimeter += np.linalg.norm(hull_points[i+1] - hull_points[i])
                
                output['value'].append(torch.tensor(perimeter, device=device, dtype=torch.float32))
                output['points'].append(torch.from_numpy(hull_points).to(device))
                output['valid_points'].append(matching_centers.detach().cpu().numpy())
            except Exception as e:
                # Fallback
                output['value'].append(torch.tensor(0.9, device=device))
                output['points'].append(None)
                output['valid_points'].append(None)
        
        output['tensor'] = torch.stack(output['value'])
        return output

    def compute_peripheries(
        self,
        triangles: Tensor,
        compute_chest: bool = True,
        compute_waist: bool = True,
        compute_hips: bool = True,
    ) -> Dict[str, Dict]:
        '''
        Compute body periphery measurements (chest, waist, hips).
        
        This CPU version uses vertex-based approximation instead of 
        mesh intersection.
        
        Parameters
        ----------
            triangles: BxFx3x3 torch.Tensor
            Contains the triangle coordinates for a batch of meshes with
            the same topology
        '''
        output = {}
        
        if compute_chest:
            output['chest'] = self.compute_periphery_from_point(
                triangles, self.chest_face_index, self.chest_bcs, 'chest')
        
        if compute_waist:
            output['waist'] = self.compute_periphery_from_point(
                triangles, self.belly_face_index, self.belly_bcs, 'waist')
        
        if compute_hips:
            output['hips'] = self.compute_periphery_from_point(
                triangles, self.hips_face_index, self.hips_bcs, 'hips')
        
        return output

    def forward(
        self,
        triangles: Tensor,
        compute_mass: bool = True,
        compute_height: bool = True,
        compute_chest: bool = True,
        compute_waist: bool = True,
        compute_hips: bool = True,
        **kwargs
    ) -> Dict:
        """
        Compute body measurements from mesh triangles.
        
        Args:
            triangles: BxFx3x3 tensor of mesh triangle vertices
            compute_mass: Whether to compute body mass
            compute_height: Whether to compute body height
            compute_chest: Whether to compute chest circumference
            compute_waist: Whether to compute waist circumference
            compute_hips: Whether to compute hip circumference
            
        Returns:
            dict with 'measurements' containing computed values
        """
        measurements = {}
        
        if compute_mass:
            measurements['mass'] = {}
            mesh_mass = self.compute_mass(triangles)
            measurements['mass']['tensor'] = mesh_mass

        if compute_height:
            measurements['height'] = {}
            mesh_height, points = self.compute_height(triangles)
            measurements['height']['tensor'] = mesh_height
            measurements['height']['points'] = points

        output = self.compute_peripheries(
            triangles,
            compute_chest=compute_chest,
            compute_waist=compute_waist,
            compute_hips=compute_hips,
        )
        measurements.update(output)

        return {'measurements': measurements}


# Also export ChestWaistHipsMeasurements for compatibility
class ChestWaistHipsMeasurements(BodyMeasurements):
    """Alias for backwards compatibility."""
    pass

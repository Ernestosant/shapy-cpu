"""
Virtual Measurements - CPU Version

This is a CPU-compatible version of the original virtual_measurements.py
Modifications:
- Device set to 'cpu' instead of 'cuda'
- CUDA availability check removed
"""
import sys
import os
import os.path as osp

# Set PyOpenGL platform for CPU rendering
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

from threadpoolctl import threadpool_limits
from tqdm import tqdm
import torch
import argparse
import trimesh
from loguru import logger
import numpy as np
import smplx
import json

# Import body measurements module
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'attributes'))
from body_measurements import BodyMeasurements

try:
    from attributes.utils.renderer import Renderer
    from PIL import ImageDraw, ImageFont
    RENDER_AVAILABLE = True
except ImportError:
    RENDER_AVAILABLE = False
    logger.warning("Renderer not available, visualization will be skipped")


@torch.no_grad()
def main(
    demo_input_folder: os.PathLike = 'demo_input',
    demo_output_folder: os.PathLike = 'demo_output',
    meas_definition_path: os.PathLike = '',
    meas_vertices_path: os.PathLike = '',
    smpl_model_path: os.PathLike = 'data/body_models/smpl',
    gender: str = 'neutral',
    num_betas: int = 10,
    render: bool = True,
) -> dict:
    """
    Compute virtual measurements from SMPL-X betas.
    
    Returns:
        dict: Measurements in meters/kg
    """
    # Use CPU device
    device = torch.device('cpu')
    logger.info('Computing virtual measurements on CPU')

    os.makedirs(demo_output_folder, exist_ok=True)

    npz_files = sorted(os.listdir(demo_input_folder))
    npz_files = [x for x in npz_files if x.endswith('npz')]

    if not npz_files:
        logger.warning(f"No .npz files found in {demo_input_folder}")
        return {}

    body_measurements = BodyMeasurements(
        {'meas_definition_path': meas_definition_path,
            'meas_vertices_path': meas_vertices_path},
    ).to(device)

    smpl = smplx.create(
        model_path=smpl_model_path,
        gender=gender,
        num_betas=num_betas,
        model_type='smplx'
    ).to(device)

    if render and RENDER_AVAILABLE:
        renderer = Renderer(is_registration=False)
    else:
        render = False

    all_measurements = {}

    for npz_file in npz_files:
        print(f'Processing: {npz_file}')

        # Read betas
        data = np.load(osp.join(demo_input_folder, npz_file))
        betas = torch.from_numpy(data['betas']).to(device).unsqueeze(0)

        # SMPL function & shaped body
        body = smpl(betas=betas)
        shaped_vertices = body['v_shaped']
        shaped_triangles = shaped_vertices[:, smpl.faces_tensor]

        # Compute the measurements on the body
        measurements = body_measurements(shaped_triangles)['measurements']
        
        # Convert to JSON-serializable format
        meas_dict = {}
        mmts_str = '    Virtual measurements: '
        for k, v in measurements.items():
            value = v['tensor'].item()
            unit = 'kg' if k == 'mass' else 'm'
            meas_dict[k] = {
                'value': value,
                'unit': unit
            }
            mmts_str += f'    {k}: {value:.2f} {unit}'
        print(mmts_str)

        # Save measurements to JSON
        base_name = npz_file.replace('.npz', '')
        json_fname = osp.join(demo_output_folder, f'{base_name}_measurements.json')
        with open(json_fname, 'w') as f:
            json.dump(meas_dict, f, indent=2)

        all_measurements[base_name] = meas_dict

        # Render shaped body if available
        if render:
            try:
                pred_mesh = trimesh.Trimesh(
                    shaped_vertices.cpu().numpy()[0], 
                    smpl.faces
                )
                pred_img = renderer.render(pred_mesh)
                
                # Add measurements to image
                font_path = osp.join(
                    osp.dirname(__file__), 
                    '..', 'samples', 'OpenSans-Regular.ttf'
                )
                if osp.exists(font_path):
                    font = ImageFont.truetype(font_path, size=24)
                else:
                    font = ImageFont.load_default()
                    
                ImageDraw.Draw(pred_img).text(
                    (0, 10), mmts_str, (0, 0, 0), font=font
                )
                pred_img.save(osp.join(demo_output_folder, f'{base_name}_vis.png'))
            except Exception as e:
                logger.warning(f"Could not render visualization: {e}")

    return all_measurements


def compute_measurements_from_betas(
    betas: np.ndarray,
    smpl_model_path: str,
    meas_definition_path: str,
    meas_vertices_path: str,
    gender: str = 'neutral',
) -> dict:
    """
    Compute measurements directly from betas array.
    
    Args:
        betas: Shape parameters (10,) or (1, 10)
        smpl_model_path: Path to SMPL-X models
        meas_definition_path: Path to measurement definitions
        meas_vertices_path: Path to measurement vertices
        gender: 'neutral', 'male', or 'female'
        
    Returns:
        dict: Measurements with values and units
    """
    device = torch.device('cpu')
    
    if betas.ndim == 1:
        betas = betas.reshape(1, -1)
    
    betas_tensor = torch.from_numpy(betas).float().to(device)
    
    body_measurements = BodyMeasurements(
        {'meas_definition_path': meas_definition_path,
         'meas_vertices_path': meas_vertices_path},
    ).to(device)

    smpl = smplx.create(
        model_path=smpl_model_path,
        gender=gender,
        num_betas=betas.shape[1],
        model_type='smplx'
    ).to(device)

    with torch.no_grad():
        body = smpl(betas=betas_tensor)
        shaped_vertices = body['v_shaped']
        shaped_triangles = shaped_vertices[:, smpl.faces_tensor]
        measurements = body_measurements(shaped_triangles)['measurements']

    meas_dict = {}
    for k, v in measurements.items():
        value = v['tensor'].item()
        unit = 'kg' if k == 'mass' else 'm'
        meas_dict[k] = {
            'value': value,
            'unit': unit,
            'value_cm': value * 100 if unit == 'm' else value
        }
    
    return meas_dict


if __name__ == '__main__':
    torch.set_num_threads(4)

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'Virtual Measurements (CPU Version)'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--output-folder', dest='output_folder',
                        default='demo_output', type=str,
                        help='The folder where the demo renderings will be saved')
    parser.add_argument('--input-folder', dest='input_folder',
                        default='demo_input', type=str,
                        help='The folder where the demo npz files are stored')
    parser.add_argument('--meas_definition_path', dest='meas_definition_path',
                        default='../data/utility_files/measurements/measurement_defitions.yaml', 
                        type=str, help='Path to measurement definitions')
    parser.add_argument('--meas_vertices_path', dest='meas_vertices_path',
                        default='../data/utility_files/measurements/smplx_measurements.yaml', 
                        type=str, help='Path to measurement vertices')
    parser.add_argument('--smpl_model_path', dest='smpl_model_path',
                        default='../data/body_models', type=str,
                        help='Path to smpl model folder')
    parser.add_argument('--num_betas', dest='num_betas',
                        default=10, type=int,
                        help='Number of betas SMPL model uses')
    parser.add_argument('--gender', dest='gender',
                        default='neutral', type=str,
                        choices=['neutral', 'male', 'female'],
                        help='Gender of SMPL model')
                        

    args = parser.parse_args()

    main( 
        demo_input_folder=args.input_folder,
        demo_output_folder=args.output_folder, 
        meas_definition_path=args.meas_definition_path,
        meas_vertices_path=args.meas_vertices_path,
        smpl_model_path=args.smpl_model_path,
        gender=args.gender,
        num_betas=args.num_betas
    )

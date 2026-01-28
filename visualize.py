#!/usr/bin/env python3
"""
SHAPY Visualization Script

Interactive 3D visualization of body mesh and anthropometric measurements.

Usage:
    python visualize.py --mesh output/photo.obj --measurements output/photo_measurements.json
    python visualize.py --mesh output/photo.obj  # Without measurements
    
Features:
    - Interactive 3D mesh viewer (rotation, zoom, pan)
    - Measurements panel with height, weight, chest, waist, hips
    - Multiple view angles (front, side, back)
    - Screenshot export
"""

import sys
import os
import os.path as osp
import argparse
import json
from pathlib import Path

import numpy as np

# Try to import visualization libraries
PYVISTA_AVAILABLE = False
OPEN3D_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    import pyvista as pv
    from pyvista import themes
    PYVISTA_AVAILABLE = True
except ImportError:
    pass

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


def load_mesh(mesh_path: str):
    """Load mesh from file."""
    import trimesh
    mesh = trimesh.load(mesh_path, process=False)
    return mesh


def load_measurements(measurements_path: str) -> dict:
    """Load measurements from JSON file."""
    with open(measurements_path, 'r') as f:
        return json.load(f)


def format_measurements(measurements: dict) -> str:
    """Format measurements for display."""
    lines = ["=" * 40, "ANTHROPOMETRIC MEASUREMENTS", "=" * 40]
    
    # Priority order for display
    priority = ['height', 'mass', 'chest', 'waist', 'hips']
    
    for key in priority:
        if key in measurements:
            m = measurements[key]
            if key == 'mass':
                lines.append(f"  Weight:  {m['value']:.1f} kg")
            else:
                value_cm = m.get('value_cm', m['value'] * 100)
                lines.append(f"  {key.capitalize()}:  {value_cm:.1f} cm")
    
    # Add any other measurements
    for key, m in measurements.items():
        if key not in priority:
            if m['unit'] == 'kg':
                lines.append(f"  {key.capitalize()}:  {m['value']:.1f} kg")
            else:
                value_cm = m.get('value_cm', m['value'] * 100)
                lines.append(f"  {key.capitalize()}:  {value_cm:.1f} cm")
    
    lines.append("=" * 40)
    return "\n".join(lines)


def visualize_pyvista(mesh_path: str, measurements: dict = None, 
                      screenshot_path: str = None):
    """
    Visualize using PyVista (recommended for Windows).
    """
    import trimesh
    
    print("Loading mesh with PyVista...")
    mesh = trimesh.load(mesh_path, process=False)
    
    # Convert to PyVista mesh
    faces = np.hstack([[3] + list(f) for f in mesh.faces])
    pv_mesh = pv.PolyData(mesh.vertices, faces)
    
    # Create plotter
    plotter = pv.Plotter(title="SHAPY Body Visualization")
    
    # Add mesh with nice coloring
    plotter.add_mesh(
        pv_mesh,
        color='wheat',
        smooth_shading=True,
        ambient=0.3,
        diffuse=0.6,
        specular=0.5,
    )
    
    # Add measurements as text
    if measurements:
        text = format_measurements(measurements)
        plotter.add_text(
            text,
            position='upper_left',
            font_size=10,
            color='black',
        )
    
    # Add axes
    plotter.add_axes()
    
    # Set camera for front view
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.2)
    
    # Add key bindings info
    plotter.add_text(
        "Controls: LMB-rotate, MMB-pan, Scroll-zoom, Q-quit",
        position='lower_left',
        font_size=8,
        color='gray',
    )
    
    # Screenshot if requested
    if screenshot_path:
        plotter.screenshot(screenshot_path)
        print(f"Screenshot saved: {screenshot_path}")
    
    # Show interactive window
    print("Opening interactive viewer...")
    print("  - Left mouse button: Rotate")
    print("  - Middle mouse button: Pan")
    print("  - Scroll wheel: Zoom")
    print("  - Q: Quit")
    
    plotter.show()


def visualize_open3d(mesh_path: str, measurements: dict = None,
                     screenshot_path: str = None):
    """
    Visualize using Open3D.
    """
    print("Loading mesh with Open3D...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    # Set mesh color
    mesh.paint_uniform_color([0.9, 0.8, 0.7])
    
    # Print measurements
    if measurements:
        print("\n" + format_measurements(measurements))
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SHAPY Body Visualization", width=1024, height=768)
    
    vis.add_geometry(mesh)
    
    # Set render options
    opt = vis.get_render_option()
    opt.light_on = True
    opt.background_color = np.array([0.9, 0.9, 0.9])
    
    # Set view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.5)
    
    # Screenshot if requested
    if screenshot_path:
        vis.capture_screen_image(screenshot_path)
        print(f"Screenshot saved: {screenshot_path}")
    
    print("\nOpening interactive viewer...")
    print("  - Left mouse button: Rotate")
    print("  - Middle mouse button: Pan")
    print("  - Scroll wheel: Zoom")
    print("  - Q/Esc: Quit")
    
    vis.run()
    vis.destroy_window()


def visualize_matplotlib(mesh_path: str, measurements: dict = None,
                         screenshot_path: str = None):
    """
    Visualize using Matplotlib (fallback, non-interactive).
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh
    
    print("Loading mesh with Matplotlib (static view)...")
    mesh = trimesh.load(mesh_path, process=False)
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    
    # 3D mesh subplot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Sample faces for faster rendering
    max_faces = 5000
    if len(mesh.faces) > max_faces:
        idx = np.random.choice(len(mesh.faces), max_faces, replace=False)
        faces_to_plot = mesh.faces[idx]
    else:
        faces_to_plot = mesh.faces
    
    # Create polygon collection
    verts = mesh.vertices[faces_to_plot]
    poly = Poly3DCollection(verts, alpha=0.8)
    poly.set_facecolor('wheat')
    poly.set_edgecolor('gray')
    poly.set_linewidth(0.1)
    
    ax1.add_collection3d(poly)
    
    # Set axis limits
    bounds = mesh.bounds
    max_range = np.max(bounds[1] - bounds[0])
    center = (bounds[0] + bounds[1]) / 2
    
    ax1.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
    ax1.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
    ax1.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Body Mesh')
    
    # Measurements subplot
    if measurements:
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        
        text = format_measurements(measurements)
        ax2.text(0.1, 0.5, text, fontsize=12, fontfamily='monospace',
                 verticalalignment='center', transform=ax2.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax2.set_title('Measurements')
    
    plt.tight_layout()
    
    # Save screenshot if requested
    if screenshot_path:
        plt.savefig(screenshot_path, dpi=150, bbox_inches='tight')
        print(f"Screenshot saved: {screenshot_path}")
    
    print("Displaying static view (close window to exit)...")
    plt.show()


def create_multi_view(mesh_path: str, measurements: dict = None,
                      output_path: str = None):
    """
    Create a multi-view image with front, side, and back views.
    Uses PyVista if available, otherwise falls back to matplotlib.
    """
    import trimesh
    
    print("Creating multi-view visualization...")
    mesh = trimesh.load(mesh_path, process=False)
    
    # Try PyVista first (better quality)
    if PYVISTA_AVAILABLE and os.environ.get('DISPLAY') is not None:
        # Convert to PyVista
        faces = np.hstack([[3] + list(f) for f in mesh.faces])
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        
        # Create plotter with subplots
        plotter = pv.Plotter(shape=(1, 3), off_screen=(output_path is not None))
        
        views = [
            ('Front', 'xy'),
            ('Side', 'xz'),
            ('Back', '-xy'),
        ]
        
        for i, (title, view) in enumerate(views):
            plotter.subplot(0, i)
            plotter.add_mesh(pv_mesh, color='wheat', smooth_shading=True)
            plotter.camera_position = view
            plotter.add_text(title, font_size=12)
        
        if output_path:
            plotter.screenshot(output_path)
            print(f"Multi-view saved: {output_path}")
            plotter.close()
        else:
            plotter.show()
        return
    
    # Fallback to matplotlib (works in headless mode)
    if not MATPLOTLIB_AVAILABLE:
        print("Error: Matplotlib required for multi-view in headless mode.")
        print("Install with: pip install matplotlib")
        return
    
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    print("Using matplotlib for multi-view (headless mode)...")
    
    # Sample faces for faster rendering
    max_faces = 3000
    if len(mesh.faces) > max_faces:
        idx = np.random.choice(len(mesh.faces), max_faces, replace=False)
        faces_to_plot = mesh.faces[idx]
    else:
        faces_to_plot = mesh.faces
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 6))
    
    views = [
        ('Front', (0, 0)),      # azim=0, elev=0
        ('Side', (90, 0)),     # azim=90, elev=0
        ('Back', (180, 0)),    # azim=180, elev=0
    ]
    
    # Calculate bounds once
    bounds = mesh.bounds
    max_range = np.max(bounds[1] - bounds[0])
    center = (bounds[0] + bounds[1]) / 2
    
    for i, (title, (azim, elev)) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        
        # Create polygon collection
        verts = mesh.vertices[faces_to_plot]
        poly = Poly3DCollection(verts, alpha=0.9)
        poly.set_facecolor('wheat')
        poly.set_edgecolor('gray')
        poly.set_linewidth(0.05)
        
        ax.add_collection3d(poly)
        
        # Set axis limits
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Multi-view saved: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='SHAPY Visualization - View 3D body mesh and measurements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py --mesh output/photo.obj
  python visualize.py --mesh output/photo.obj --measurements output/photo_measurements.json
  python visualize.py --mesh output/photo.obj --screenshot visualization.png
  python visualize.py --mesh output/photo.obj --multi-view --screenshot views.png
        """
    )
    
    parser.add_argument('--mesh', '-m', type=str, required=True,
                        help='Path to mesh file (.obj, .ply)')
    parser.add_argument('--measurements', '-M', type=str, default=None,
                        help='Path to measurements JSON file')
    parser.add_argument('--screenshot', '-s', type=str, default=None,
                        help='Save screenshot to file')
    parser.add_argument('--multi-view', action='store_true',
                        help='Create multi-view image (front, side, back)')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'pyvista', 'open3d', 'matplotlib'],
                        help='Visualization backend (default: auto)')
    
    args = parser.parse_args()
    
    # Verify mesh exists
    if not osp.exists(args.mesh):
        print(f"Error: Mesh file not found: {args.mesh}")
        sys.exit(1)
    
    # Load measurements if provided
    measurements = None
    if args.measurements:
        if osp.exists(args.measurements):
            measurements = load_measurements(args.measurements)
        else:
            print(f"Warning: Measurements file not found: {args.measurements}")
    
    # Auto-detect measurements file if not provided
    if measurements is None:
        mesh_dir = osp.dirname(args.mesh)
        mesh_name = osp.splitext(osp.basename(args.mesh))[0]
        auto_meas_path = osp.join(mesh_dir, f"{mesh_name}_measurements.json")
        if osp.exists(auto_meas_path):
            print(f"Auto-detected measurements: {auto_meas_path}")
            measurements = load_measurements(auto_meas_path)
    
    # Multi-view mode
    if args.multi_view:
        create_multi_view(args.mesh, measurements, args.screenshot)
        return
    
    # Select backend
    backend = args.backend
    if backend == 'auto':
        # Check if we're in headless mode (screenshot only, no display)
        headless = args.screenshot is not None and os.environ.get('DISPLAY') is None
        
        if headless:
            # In headless mode, prefer matplotlib (works without DISPLAY)
            if MATPLOTLIB_AVAILABLE:
                backend = 'matplotlib'
            else:
                print("Error: Matplotlib required for headless screenshot mode.")
                print("Install with: pip install matplotlib")
                sys.exit(1)
        elif PYVISTA_AVAILABLE:
            backend = 'pyvista'
        elif OPEN3D_AVAILABLE:
            backend = 'open3d'
        elif MATPLOTLIB_AVAILABLE:
            backend = 'matplotlib'
        else:
            print("Error: No visualization library available.")
            print("Install one of: pip install pyvista open3d matplotlib")
            sys.exit(1)
    
    # Run visualization
    print(f"Using {backend} backend...")
    
    if backend == 'pyvista':
        if not PYVISTA_AVAILABLE:
            print("PyVista not available. Install with: pip install pyvista")
            sys.exit(1)
        visualize_pyvista(args.mesh, measurements, args.screenshot)
    
    elif backend == 'open3d':
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Install with: pip install open3d")
            sys.exit(1)
        visualize_open3d(args.mesh, measurements, args.screenshot)
    
    elif backend == 'matplotlib':
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Install with: pip install matplotlib")
            sys.exit(1)
        visualize_matplotlib(args.mesh, measurements, args.screenshot)


if __name__ == '__main__':
    main()

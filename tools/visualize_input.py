import numpy as np
import pyvista as pv
from pathlib import Path
import logging
from scipy.spatial import cKDTree
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Visualizer")

# --- Global Configuration ---
# Define paths here directly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "input_txt" / "nonsmooth_geometry"

MODEL_NAME = "Cylinder"
NODES_PATH = DATA_DIR / f"{MODEL_NAME}_nodes.txt"
NORMALS_PATH = DATA_DIR / f"{MODEL_NAME}_normals.txt"
PATCHES_PATH = DATA_DIR / f"{MODEL_NAME}_patches.txt"

# Set to True to render without opening a window (for testing)
OFF_SCREEN = False
# ----------------------------

def load_data(file_path):
    """
    Load 3D coordinate data from a text file.
    Assumes whitespace-separated values, one point (x y z) per line.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data = np.loadtxt(str(file_path))
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {data.shape}")
        logger.info(f"Loaded {len(data)} points from {file_path.name}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise

def estimate_avg_spacing(points, k=2):
    """
    Estimate the average spacing between points using k-NN.
    Used to determine appropriate scaling for glyphs and radii.
    """
    if len(points) < 2:
        return 1.0
    tree = cKDTree(points)
    # Query k nearest neighbors (1st is self, 2nd is nearest)
    dists, _ = tree.query(points, k=k)
    # Average distance to the nearest neighbor
    avg_dist = np.mean(dists[:, 1])
    return avg_dist

def visualize_data(nodes_path, normals_path, patches_path, off_screen=False):
    """
    Main visualization function using PyVista.
    
    Args:
        nodes_path (str): Path to nodes file.
        normals_path (str): Path to normals file.
        patches_path (str): Path to patches file.
        off_screen (bool): If True, render without opening a window (for testing).
    """
    # 1. Load Data
    logger.info("Loading data files...")
    try:
        nodes = load_data(nodes_path)
        normals = load_data(normals_path)
        patches = load_data(patches_path)
    except Exception as e:
        logger.error("Data loading failed. Exiting.")
        return

    if len(nodes) != len(normals):
        logger.warning(f"Count mismatch: Nodes ({len(nodes)}) vs Normals ({len(normals)})")

    # 2. Prepare Geometries
    logger.info("Preparing geometries...")
    
    # -- Nodes (Point Cloud) --
    pdata_nodes = pv.PolyData(nodes)
    
    # -- Normals (Arrows) --
    # Calculate scale factor for arrows based on node density
    node_spacing = estimate_avg_spacing(nodes)
    arrow_scale = node_spacing * 0.6
    
    # Add normals to nodes PolyData to generate glyphs
    pdata_nodes["normals"] = normals
    arrows = pdata_nodes.glyph(orient="normals", scale=False, factor=arrow_scale)
    
    # -- Patches (Semi-transparent Spheres) --
    # Estimate patch radius based on patch centers density
    # Patches usually cover the domain, so radius ~ spacing
    patch_spacing = estimate_avg_spacing(patches)
    patch_radius = patch_spacing * 0.7 
    
    pdata_patches = pv.PolyData(patches)
    # Create sphere glyphs
    sphere_geom = pv.Sphere(radius=patch_radius, theta_resolution=16, phi_resolution=16)
    spheres_patches = pdata_patches.glyph(geom=sphere_geom, scale=False)

    # 3. Setup Plotter
    logger.info("Setting up visualization scene...")
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.title = "PyCFPU Geometry Visualizer"
    plotter.set_background("white")

    # Add Actors
    # Nodes: Blue spheres
    actor_nodes = plotter.add_mesh(
        pdata_nodes, 
        color='#1f77b4', 
        point_size=6, 
        render_points_as_spheres=True, 
        label='Nodes'
    )
    
    # Normals: Green arrows
    actor_normals = plotter.add_mesh(
        arrows, 
        color='#2ca02c', 
        label='Normals'
    )
    
    # Patches: Red semi-transparent spheres
    actor_patches = plotter.add_mesh(
        spheres_patches, 
        color='#d62728', 
        opacity=0.3, 
        label='Patches',
        smooth_shading=True
    )

    plotter.add_axes()
    # Add a custom legend
    plotter.add_legend(labels=[
        ('Nodes', '#1f77b4'),
        ('Normals', '#2ca02c'),
        ('Patches', '#d62728')
    ], bcolor='white', border=True)

    # 4. Add Interactive Widgets
    if not off_screen:
        # Toggle Visibility Checkboxes
        size = 25
        border_size = 3
        
        def toggle_nodes(state):
            actor_nodes.SetVisibility(state)
            
        def toggle_normals(state):
            actor_normals.SetVisibility(state)
            
        def toggle_patches(state):
            actor_patches.SetVisibility(state)

        # Position: Bottom Left
        start_y = 20
        gap = 40
        
        plotter.add_checkbox_button_widget(
            toggle_nodes, value=True, 
            color_on='#1f77b4', color_off='grey', 
            position=(10, start_y)
        )
        plotter.add_text("Show Nodes", position=(45, start_y), font_size=10, color='black')

        plotter.add_checkbox_button_widget(
            toggle_normals, value=True, 
            color_on='#2ca02c', color_off='grey', 
            position=(10, start_y + gap)
        )
        plotter.add_text("Show Normals", position=(45, start_y + gap), font_size=10, color='black')

        plotter.add_checkbox_button_widget(
            toggle_patches, value=True, 
            color_on='#d62728', color_off='grey', 
            position=(10, start_y + gap*2)
        )
        plotter.add_text("Show Patches", position=(45, start_y + gap*2), font_size=10, color='black')

        # Slider for Patch Opacity
        def set_opacity(value):
            actor_patches.GetProperty().SetOpacity(value)
            
        plotter.add_slider_widget(
            set_opacity, 
            [0, 1], 
            value=0.3, 
            title="Patch Opacity", 
            pointa=(0.7, 0.9), 
            pointb=(0.9, 0.9),
            style='modern'
        )

        # Key event for screenshot
        def save_screenshot():
            fname = "visualization_screenshot.png"
            plotter.screenshot(fname)
            logger.info(f"Screenshot saved to {fname}")
            # Show temporary message
            plotter.add_text(f"Saved {fname}", position='upper_left', font_size=12, name='msg', color='black')
            
        plotter.add_key_event("s", save_screenshot)
        plotter.add_text("Press 's' to save screenshot", position='upper_right', font_size=10, color='black')

    # 5. Show
    logger.info("Visualization ready.")
    logger.info("Interactive Controls:")
    logger.info("  - Mouse Left Drag: Rotate")
    logger.info("  - Mouse Shift+Left: Pan")
    logger.info("  - Mouse Scroll: Zoom")
    logger.info("  - 's' Key: Save Screenshot")
    
    if off_screen:
        plotter.show(auto_close=False)
        plotter.screenshot("render_offscreen_test.png")
        logger.info("Off-screen render saved to render_offscreen_test.png")
    else:
        plotter.show()

def main():
    logger.info("Starting visualizer with global configuration...")
    logger.info(f"Nodes: {NODES_PATH}")
    logger.info(f"Normals: {NORMALS_PATH}")
    logger.info(f"Patches: {PATCHES_PATH}")
    
    visualize_data(NODES_PATH, NORMALS_PATH, PATCHES_PATH, OFF_SCREEN)

if __name__ == "__main__":
    main()

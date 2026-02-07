"""
Generate Webots .wbt world files for all grid layouts from env_lava_variants.py.

Each layout is converted to a physical arena with:
- HazardObstacle nodes for lava cells
- TIAGo robots at start positions with correct goals and alpha values
- Target markers at goal positions
- Empathy indicator spheres

Usage:
    python webots_sim/generate_worlds.py
    python webots_sim/generate_worlds.py --layouts narrow passing_bay
    python webots_sim/generate_worlds.py --alpha0 0.0 --alpha1 6.0
"""

import sys
import os
import math
import argparse

# Add project root to path so we can import tom.envs
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tom.envs.env_lava_variants import LAYOUTS, get_layout

# ===========================================================================
# Constants
# ===========================================================================

CELL_SIZE = 0.7       # meters per grid cell (TIAGo is ~0.54m wide)
ARENA_MARGIN = 1.0    # extra margin around the grid (meters)
HAZARD_HEIGHT = 0.15  # height of hazard boxes
HAZARD_GAP = 0.05     # gap between adjacent hazards for visual clarity
ROBOT_Z = 0.095       # TIAGo spawn height
GOAL_Z = 0.02         # Goal marker height
MARKER_Z = 1.8        # Empathy indicator height

# Default alpha values
DEFAULT_ALPHA_0 = 0.0   # Agent 0: selfish
DEFAULT_ALPHA_1 = 6.0   # Agent 1: empathic


# ===========================================================================
# Coordinate mapping
# ===========================================================================

def grid_to_continuous(gx, gy, width, height):
    """Map grid cell (gx, gy) to continuous (cx, cy) centered at origin.

    Grid y=0 is top -> positive Y in Webots.
    Grid x=0 is left -> negative X in Webots.
    """
    cx = (gx - (width - 1) / 2.0) * CELL_SIZE
    cy = ((height - 1) / 2.0 - gy) * CELL_SIZE
    return cx, cy


def compute_arena_size(width, height):
    """Compute arena floor dimensions for a grid."""
    arena_x = width * CELL_SIZE + ARENA_MARGIN
    arena_y = height * CELL_SIZE + ARENA_MARGIN
    return arena_x, arena_y


# ===========================================================================
# Lava cell merging (row-based)
# ===========================================================================

def get_lava_cells(layout):
    """Compute lava cells as complement of safe_cells."""
    all_cells = set((x, y) for x in range(layout.width) for y in range(layout.height))
    safe = set(map(tuple, layout.safe_cells))
    return all_cells - safe


def merge_lava_rows(lava_cells, width, height):
    """Merge adjacent lava cells in each row into rectangles.

    Returns list of (x_start, x_end, y) tuples where x_start..x_end
    are inclusive grid column indices for a merged rectangle.
    """
    merged = []
    for y in range(height):
        row_lava = sorted([x for (x, yy) in lava_cells if yy == y])
        if not row_lava:
            continue
        start = row_lava[0]
        end = row_lava[0]
        for x in row_lava[1:]:
            if x == end + 1:
                end = x
            else:
                merged.append((start, end, y))
                start = x
                end = x
        merged.append((start, end, y))
    return merged


# ===========================================================================
# World file generation
# ===========================================================================

def generate_world(layout_name, alpha_0=DEFAULT_ALPHA_0, alpha_1=DEFAULT_ALPHA_1):
    """Generate a complete .wbt world file string for a layout."""
    layout = get_layout(layout_name)
    W, H = layout.width, layout.height
    arena_x, arena_y = compute_arena_size(W, H)

    # Viewpoint height scales with arena size
    vp_z = max(6.0, max(arena_x, arena_y) * 1.3)

    # Robot positions and goals
    s0x, s0y = grid_to_continuous(*layout.start_positions[0], W, H)
    s1x, s1y = grid_to_continuous(*layout.start_positions[1], W, H)
    g0x, g0y = grid_to_continuous(*layout.goal_positions[0], W, H)
    g1x, g1y = grid_to_continuous(*layout.goal_positions[1], W, H)

    # Robot heading (face toward goal)
    heading_0 = math.atan2(g0y - s0y, g0x - s0x)
    heading_1 = math.atan2(g1y - s1y, g1x - s1x)

    # Lava -> hazard obstacles
    lava_cells = get_lava_cells(layout)
    merged = merge_lava_rows(lava_cells, W, H)

    lines = []
    lines.append('#VRML_SIM R2025a utf8')
    lines.append(f'# Auto-generated world for layout: {layout_name}')
    lines.append(f'# Grid: {W}x{H}, Arena: {arena_x:.1f}x{arena_y:.1f}m')
    lines.append('')
    lines.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"')
    lines.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"')
    lines.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/floors/protos/RectangleArena.proto"')
    lines.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/pal_robotics/tiago/protos/Tiago.proto"')
    lines.append('EXTERNPROTO "../protos/Target.proto"')
    lines.append('EXTERNPROTO "../protos/HazardObstacle.proto"')
    lines.append('')

    # WorldInfo
    title = f"TIAGo {layout_name.replace('_', ' ').title()}"
    lines.append('WorldInfo {')
    lines.append(f'  title "{title}"')
    lines.append('  basicTimeStep 16')
    lines.append('}')
    lines.append('')

    # Viewpoint (top-down)
    lines.append('Viewpoint {')
    lines.append('  orientation -0.5773 0.5773 0.5773 2.0944')
    lines.append(f'  position 0 0 {vp_z:.1f}')
    lines.append('}')
    lines.append('')

    # Background
    lines.append('TexturedBackground {')
    lines.append('  texture "empty_office"')
    lines.append('}')
    lines.append('')
    lines.append('TexturedBackgroundLight {')
    lines.append('  texture "empty_office"')
    lines.append('}')
    lines.append('')

    # Arena floor
    lines.append(f'# Arena: {arena_x:.1f}m x {arena_y:.1f}m')
    lines.append('RectangleArena {')
    lines.append(f'  floorSize {arena_x:.1f} {arena_y:.1f}')
    lines.append('}')
    lines.append('')

    # Hazard obstacles
    lines.append(f'# Hazard obstacles ({len(merged)} merged blocks from {len(lava_cells)} lava cells)')
    hz = HAZARD_HEIGHT / 2.0
    for idx, (x_start, x_end, gy) in enumerate(merged):
        # Center of merged block
        cx_start, _ = grid_to_continuous(x_start, gy, W, H)
        cx_end, _ = grid_to_continuous(x_end, gy, W, H)
        cx = (cx_start + cx_end) / 2.0
        _, cy = grid_to_continuous(0, gy, W, H)

        # Size: spans from x_start to x_end
        n_cells = x_end - x_start + 1
        size_x = n_cells * CELL_SIZE - HAZARD_GAP
        size_y = CELL_SIZE - HAZARD_GAP

        lines.append('HazardObstacle {')
        lines.append(f'  translation {cx:.3f} {cy:.3f} {hz:.3f}')
        lines.append(f'  name "hazard_{idx}"')
        lines.append(f'  size {size_x:.2f} {size_y:.2f} {HAZARD_HEIGHT}')
        lines.append('}')

    lines.append('')

    # Goal markers
    lines.append('# Goal markers')
    lines.append('Target {')
    lines.append(f'  translation {g0x:.3f} {g0y:.3f} {GOAL_Z}')
    lines.append('  name "Goal_TIAGo_1"')
    lines.append('  color 0.2 0.5 1')
    lines.append('  radius 0.15')
    lines.append('}')
    lines.append('')
    lines.append('Target {')
    lines.append(f'  translation {g1x:.3f} {g1y:.3f} {GOAL_Z}')
    lines.append('  name "Goal_TIAGo_2"')
    lines.append('  color 1 0.4 0.7')
    lines.append('  radius 0.15')
    lines.append('}')
    lines.append('')

    # Empathy indicators
    lines.append('# Empathy indicators (RED=selfish, GREEN=empathic)')
    lines.append('DEF MARKER_SELFISH Solid {')
    lines.append(f'  translation {s0x:.3f} {s0y:.3f} {MARKER_Z}')
    lines.append('  children [')
    lines.append('    Shape {')
    lines.append('      appearance PBRAppearance {')
    lines.append('        baseColor 1 0.2 0.2')
    lines.append('        emissiveColor 0.5 0 0')
    lines.append('        roughness 0.3')
    lines.append('        metalness 0')
    lines.append('      }')
    lines.append('      geometry Sphere {')
    lines.append('        radius 0.12')
    lines.append('      }')
    lines.append('    }')
    lines.append('  ]')
    lines.append('  name "marker_selfish"')
    lines.append('}')
    lines.append('')
    lines.append('DEF MARKER_EMPATHIC Solid {')
    lines.append(f'  translation {s1x:.3f} {s1y:.3f} {MARKER_Z}')
    lines.append('  children [')
    lines.append('    Shape {')
    lines.append('      appearance PBRAppearance {')
    lines.append('        baseColor 0.2 1 0.4')
    lines.append('        emissiveColor 0 0.5 0')
    lines.append('        roughness 0.3')
    lines.append('        metalness 0')
    lines.append('      }')
    lines.append('      geometry Sphere {')
    lines.append('        radius 0.12')
    lines.append('      }')
    lines.append('    }')
    lines.append('  ]')
    lines.append('  name "marker_empathic"')
    lines.append('}')
    lines.append('')

    # TIAGo robots
    lines.append('# TIAGo 1 - agent_id=0')
    lines.append(f'# alpha={alpha_0} ({"selfish" if alpha_0 < 0.1 else "empathic"})')
    lines.append('Tiago {')
    lines.append(f'  translation {s0x:.3f} {s0y:.3f} {ROBOT_Z}')
    lines.append(f'  rotation 0 0 1 {heading_0:.5f}')
    lines.append('  name "TIAGo_1"')
    lines.append('  controller "tiago_empathic"')
    lines.append('  supervisor TRUE')
    lines.append(f'  customData "{g0x:.2f},{g0y:.2f},{alpha_0},{0}"')
    lines.append('}')
    lines.append('')

    lines.append('# TIAGo 2 - agent_id=1')
    lines.append(f'# alpha={alpha_1} ({"selfish" if alpha_1 < 0.1 else "empathic"})')
    lines.append('Tiago {')
    lines.append(f'  translation {s1x:.3f} {s1y:.3f} {ROBOT_Z}')
    lines.append(f'  rotation 0 0 1 {heading_1:.5f}')
    lines.append('  name "TIAGo_2"')
    lines.append('  controller "tiago_empathic"')
    lines.append('  supervisor TRUE')
    lines.append(f'  customData "{g1x:.2f},{g1y:.2f},{alpha_1},{1}"')
    lines.append('}')
    lines.append('')

    return '\n'.join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Webots worlds from grid layouts')
    parser.add_argument('--layouts', nargs='*', default=None,
                        help='Layout names to generate (default: all)')
    parser.add_argument('--alpha0', type=float, default=DEFAULT_ALPHA_0,
                        help=f'Alpha for agent 0 (default: {DEFAULT_ALPHA_0})')
    parser.add_argument('--alpha1', type=float, default=DEFAULT_ALPHA_1,
                        help=f'Alpha for agent 1 (default: {DEFAULT_ALPHA_1})')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: webots_sim/worlds/)')
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds')
    os.makedirs(out_dir, exist_ok=True)

    # Determine which layouts to generate
    layout_names = args.layouts if args.layouts else list(LAYOUTS.keys())

    print(f"Generating {len(layout_names)} world files in {out_dir}/")
    print(f"Alpha values: agent_0={args.alpha0}, agent_1={args.alpha1}")
    print()

    for name in layout_names:
        if name not in LAYOUTS:
            print(f"  WARNING: Unknown layout '{name}', skipping")
            continue

        layout = get_layout(name)
        arena_x, arena_y = compute_arena_size(layout.width, layout.height)
        lava_cells = get_lava_cells(layout)
        merged = merge_lava_rows(lava_cells, layout.width, layout.height)

        wbt_content = generate_world(name, alpha_0=args.alpha0, alpha_1=args.alpha1)

        filename = f"tiago_{name}.wbt"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(wbt_content)

        print(f"  {filename:35s} grid={layout.width}x{layout.height}  "
              f"arena={arena_x:.1f}x{arena_y:.1f}m  "
              f"hazards={len(merged)} blocks ({len(lava_cells)} cells)")

    print(f"\nDone. Generated {len(layout_names)} world files.")


if __name__ == '__main__':
    main()

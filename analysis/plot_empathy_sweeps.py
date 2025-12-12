"""
Analysis and visualization for empathy sweep experiments.

Generates:
1. Cooperation vs empathy curves (1D symmetric)
2. Asymmetry heatmaps (2D)
3. Layout complexity plots
4. Paralysis curves
5. Role effect comparisons

Usage:
    python analysis/plot_empathy_sweeps.py results/empathy_sweep_YYYYMMDD_HHMMSS.csv
    python analysis/plot_empathy_sweeps.py --latest  # Use most recent results
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom colormap for heatmaps
COOPERATION_CMAP = LinearSegmentedColormap.from_list(
    "cooperation",
    ["#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4"]
)


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results from CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiments from {csv_path}")
    return df


def find_latest_results(results_dir: str = "results") -> str:
    """Find the most recent empathy sweep results file."""
    pattern = os.path.join(results_dir, "empathy_sweep_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No empathy sweep results found in {results_dir}")
    return max(files, key=os.path.getctime)


# =============================================================================
# Plot 1: Cooperation vs Empathy (1D Symmetric)
# =============================================================================

def plot_cooperation_vs_empathy(df: pd.DataFrame, output_dir: str):
    """
    Plot cooperation probability vs symmetric empathy level for each layout.

    Creates one line per layout showing how cooperation changes with alpha.
    """
    # Filter to symmetric cases only
    sym_df = df[df['alpha_i'] == df['alpha_j']].copy()

    if len(sym_df) == 0:
        print("No symmetric empathy data found, skipping cooperation vs empathy plot")
        return

    # Group by layout and alpha
    grouped = sym_df.groupby(['layout', 'alpha_i']).agg({
        'both_success': ['mean', 'std', 'count'],
        'paralysis': 'mean',
        'cell_collision': 'mean'
    }).reset_index()

    grouped.columns = ['layout', 'alpha', 'success_mean', 'success_std', 'n',
                       'paralysis_mean', 'collision_mean']

    # Calculate confidence intervals
    grouped['success_ci'] = 1.96 * grouped['success_std'] / np.sqrt(grouped['n'])

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    layouts = grouped['layout'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(layouts)))

    # Plot 1: Success rate
    ax = axes[0]
    for i, layout in enumerate(sorted(layouts)):
        layout_data = grouped[grouped['layout'] == layout]
        ax.plot(layout_data['alpha'], layout_data['success_mean'],
                'o-', color=colors[i], label=layout, linewidth=2, markersize=6)
        ax.fill_between(layout_data['alpha'],
                        layout_data['success_mean'] - layout_data['success_ci'],
                        layout_data['success_mean'] + layout_data['success_ci'],
                        alpha=0.2, color=colors[i])

    ax.set_xlabel('Empathy (alpha)', fontsize=12)
    ax.set_ylabel('P(Both Success)', fontsize=12)
    ax.set_title('Cooperation vs Empathy (Symmetric)', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: Paralysis rate
    ax = axes[1]
    for i, layout in enumerate(sorted(layouts)):
        layout_data = grouped[grouped['layout'] == layout]
        ax.plot(layout_data['alpha'], layout_data['paralysis_mean'],
                'o-', color=colors[i], label=layout, linewidth=2, markersize=6)

    ax.set_xlabel('Empathy (alpha)', fontsize=12)
    ax.set_ylabel('P(Paralysis)', fontsize=12)
    ax.set_title('Paralysis vs Empathy (Symmetric)', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Plot 3: Collision rate
    ax = axes[2]
    for i, layout in enumerate(sorted(layouts)):
        layout_data = grouped[grouped['layout'] == layout]
        ax.plot(layout_data['alpha'], layout_data['collision_mean'],
                'o-', color=colors[i], label=layout, linewidth=2, markersize=6)

    ax.set_xlabel('Empathy (alpha)', fontsize=12)
    ax.set_ylabel('P(Collision)', fontsize=12)
    ax.set_title('Collision vs Empathy (Symmetric)', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'cooperation_vs_empathy.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Plot 2: Asymmetry Heatmaps
# =============================================================================

def plot_asymmetry_heatmaps(df: pd.DataFrame, output_dir: str):
    """
    Create 2D heatmaps showing success/paralysis as function of (alpha_i, alpha_j).

    Creates one set of heatmaps per layout.
    """
    layouts = df['layout'].unique()

    for layout in layouts:
        layout_df = df[df['layout'] == layout]

        # Group by alpha_i, alpha_j
        grouped = layout_df.groupby(['alpha_i', 'alpha_j']).agg({
            'both_success': 'mean',
            'paralysis': 'mean',
            'cell_collision': 'mean'
        }).reset_index()

        if len(grouped) < 4:  # Need enough data for heatmap
            continue

        # Get unique alpha values
        alpha_vals = sorted(grouped['alpha_i'].unique())

        # Create pivot tables
        success_pivot = grouped.pivot(index='alpha_j', columns='alpha_i', values='both_success')
        paralysis_pivot = grouped.pivot(index='alpha_j', columns='alpha_i', values='paralysis')
        collision_pivot = grouped.pivot(index='alpha_j', columns='alpha_i', values='cell_collision')

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f'Layout: {layout.upper()}', fontsize=14, fontweight='bold')

        # Success heatmap
        ax = axes[0]
        sns.heatmap(success_pivot, ax=ax, annot=True, fmt='.2f',
                    cmap=COOPERATION_CMAP, vmin=0, vmax=1,
                    cbar_kws={'label': 'P(Success)'})
        ax.set_xlabel('Agent i Empathy', fontsize=11)
        ax.set_ylabel('Agent j Empathy', fontsize=11)
        ax.set_title('Cooperation Rate', fontsize=12)
        ax.invert_yaxis()

        # Paralysis heatmap
        ax = axes[1]
        sns.heatmap(paralysis_pivot, ax=ax, annot=True, fmt='.2f',
                    cmap='Reds', vmin=0, vmax=1,
                    cbar_kws={'label': 'P(Paralysis)'})
        ax.set_xlabel('Agent i Empathy', fontsize=11)
        ax.set_ylabel('Agent j Empathy', fontsize=11)
        ax.set_title('Paralysis Rate', fontsize=12)
        ax.invert_yaxis()

        # Collision heatmap
        ax = axes[2]
        sns.heatmap(collision_pivot, ax=ax, annot=True, fmt='.2f',
                    cmap='Oranges', vmin=0, vmax=1,
                    cbar_kws={'label': 'P(Collision)'})
        ax.set_xlabel('Agent i Empathy', fontsize=11)
        ax.set_ylabel('Agent j Empathy', fontsize=11)
        ax.set_title('Collision Rate', fontsize=12)
        ax.invert_yaxis()

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'heatmap_{layout}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


# =============================================================================
# Plot 3: Layout Complexity Analysis
# =============================================================================

def plot_complexity_analysis(df: pd.DataFrame, output_dir: str):
    """
    Plot relationship between layout complexity and required empathy.

    Shows critical alpha level needed to achieve >80% cooperation.
    """
    if 'complexity' not in df.columns:
        print("No complexity data, skipping complexity analysis")
        return

    # Filter to symmetric cases
    sym_df = df[df['alpha_i'] == df['alpha_j']].copy()

    if len(sym_df) == 0:
        return

    # For each layout, find critical alpha where success > 0.8
    results = []
    for layout in sym_df['layout'].unique():
        layout_df = sym_df[sym_df['layout'] == layout]
        complexity = layout_df['complexity'].iloc[0]

        grouped = layout_df.groupby('alpha_i')['both_success'].mean()

        # Find first alpha where success > 0.8
        critical_alpha = None
        for alpha in sorted(grouped.index):
            if grouped[alpha] >= 0.8:
                critical_alpha = alpha
                break

        if critical_alpha is None:
            critical_alpha = 1.0  # Never reached 80%

        results.append({
            'layout': layout,
            'complexity': complexity,
            'critical_alpha': critical_alpha,
            'max_success': grouped.max()
        })

    results_df = pd.DataFrame(results)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Critical alpha vs complexity
    ax = axes[0]
    ax.scatter(results_df['complexity'], results_df['critical_alpha'], s=100, alpha=0.7)
    for _, row in results_df.iterrows():
        ax.annotate(row['layout'], (row['complexity'], row['critical_alpha']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Layout Complexity', fontsize=12)
    ax.set_ylabel('Critical Empathy (alpha*)', fontsize=12)
    ax.set_title('Empathy Required for 80% Cooperation', fontsize=14)

    # Plot 2: Max success vs complexity
    ax = axes[1]
    ax.scatter(results_df['complexity'], results_df['max_success'], s=100, alpha=0.7)
    for _, row in results_df.iterrows():
        ax.annotate(row['layout'], (row['complexity'], row['max_success']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Layout Complexity', fontsize=12)
    ax.set_ylabel('Maximum Success Rate', fontsize=12)
    ax.set_title('Best Achievable Cooperation', fontsize=14)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'complexity_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Plot 4: Role Effect (Config A vs B)
# =============================================================================

def plot_role_effects(df: pd.DataFrame, output_dir: str):
    """
    Compare outcomes between Config A and Config B (swapped start positions).

    Shows whether spatial advantage affects empathy effectiveness.
    """
    if 'start_config' not in df.columns or len(df['start_config'].unique()) < 2:
        print("No start_config variation, skipping role effects plot")
        return

    # Filter to symmetric cases
    sym_df = df[df['alpha_i'] == df['alpha_j']].copy()

    if len(sym_df) == 0:
        return

    # Group by layout, config, alpha
    grouped = sym_df.groupby(['layout', 'start_config', 'alpha_i']).agg({
        'both_success': 'mean',
        'paralysis': 'mean'
    }).reset_index()

    layouts = grouped['layout'].unique()

    # Create subplots
    n_layouts = len(layouts)
    n_cols = min(3, n_layouts)
    n_rows = (n_layouts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)

    for idx, layout in enumerate(sorted(layouts)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        layout_data = grouped[grouped['layout'] == layout]

        for config in ['A', 'B']:
            config_data = layout_data[layout_data['start_config'] == config]
            if len(config_data) > 0:
                linestyle = '-' if config == 'A' else '--'
                ax.plot(config_data['alpha_i'], config_data['both_success'],
                        f'o{linestyle}', label=f'Config {config}', linewidth=2)

        ax.set_xlabel('Empathy (alpha)')
        ax.set_ylabel('P(Success)')
        ax.set_title(layout.upper())
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Remove empty subplots
    for idx in range(len(layouts), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('Effect of Agent Starting Position (Config A vs B)', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'role_effects.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Plot 5: Paralysis Phase Diagram
# =============================================================================

def plot_paralysis_phases(df: pd.DataFrame, output_dir: str):
    """
    Plot success, paralysis, and collision as stacked areas for constrained layouts.

    Shows phase transitions as empathy increases.
    """
    # Filter to symmetric cases and constrained layouts
    constrained_layouts = ['bottleneck', 'passing_bay', 'double_bottleneck', 't_junction', 'narrow']
    sym_df = df[(df['alpha_i'] == df['alpha_j']) &
                (df['layout'].isin(constrained_layouts))].copy()

    if len(sym_df) == 0:
        print("No data for constrained layouts, skipping paralysis phase plot")
        return

    layouts = sym_df['layout'].unique()
    n_layouts = len(layouts)

    fig, axes = plt.subplots(1, n_layouts, figsize=(5*n_layouts, 4), squeeze=False)

    for idx, layout in enumerate(sorted(layouts)):
        ax = axes[0, idx]
        layout_df = sym_df[sym_df['layout'] == layout]

        grouped = layout_df.groupby('alpha_i').agg({
            'both_success': 'mean',
            'paralysis': 'mean',
            'cell_collision': 'mean',
            'failure': 'mean'
        }).reset_index()

        # Stacked area plot
        alphas = grouped['alpha_i']
        ax.fill_between(alphas, 0, grouped['both_success'],
                        alpha=0.7, label='Success', color='#2ecc71')
        ax.fill_between(alphas, grouped['both_success'],
                        grouped['both_success'] + grouped['paralysis'],
                        alpha=0.7, label='Paralysis', color='#f39c12')
        ax.fill_between(alphas, grouped['both_success'] + grouped['paralysis'],
                        grouped['both_success'] + grouped['paralysis'] + grouped['cell_collision'],
                        alpha=0.7, label='Collision', color='#e74c3c')

        ax.set_xlabel('Empathy (alpha)')
        ax.set_ylabel('Probability')
        ax.set_title(layout.upper())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=8)

    plt.suptitle('Behavioral Phases vs Empathy', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'paralysis_phases.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Plot 6: Exploitation Analysis
# =============================================================================

def plot_exploitation_analysis(df: pd.DataFrame, output_dir: str):
    """
    Analyze exploitation in asymmetric empathy settings.

    Shows arrival gap (who arrives first) as function of empathy asymmetry.
    """
    # Filter to asymmetric cases
    asym_df = df[df['alpha_i'] != df['alpha_j']].copy()

    if len(asym_df) == 0:
        print("No asymmetric data, skipping exploitation analysis")
        return

    # Calculate empathy difference
    asym_df['empathy_diff'] = asym_df['alpha_i'] - asym_df['alpha_j']

    # Group by empathy difference and analyze
    grouped = asym_df.groupby(['layout', 'empathy_diff']).agg({
        'arrival_gap': 'mean',
        'both_success': 'mean'
    }).reset_index()

    # Create plot
    layouts = grouped['layout'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(layouts)))

    # Plot 1: Arrival gap vs empathy difference
    ax = axes[0]
    for i, layout in enumerate(sorted(layouts)):
        layout_data = grouped[grouped['layout'] == layout]
        ax.plot(layout_data['empathy_diff'], layout_data['arrival_gap'],
                'o-', color=colors[i], label=layout, linewidth=2)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Empathy Difference (alpha_i - alpha_j)', fontsize=12)
    ax.set_ylabel('Arrival Gap (steps_j - steps_i)', fontsize=12)
    ax.set_title('Who Arrives First?', fontsize=14)
    ax.legend(loc='best', fontsize=9)

    # Plot 2: Success rate vs empathy difference
    ax = axes[1]
    for i, layout in enumerate(sorted(layouts)):
        layout_data = grouped[grouped['layout'] == layout]
        ax.plot(layout_data['empathy_diff'], layout_data['both_success'],
                'o-', color=colors[i], label=layout, linewidth=2)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Empathy Difference (alpha_i - alpha_j)', fontsize=12)
    ax.set_ylabel('P(Both Success)', fontsize=12)
    ax.set_title('Coordination Success', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exploitation_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Summary Statistics
# =============================================================================

def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """Generate summary statistics table."""
    # Filter to symmetric cases
    sym_df = df[df['alpha_i'] == df['alpha_j']].copy()

    if len(sym_df) == 0:
        return

    # Summary by layout
    summary = sym_df.groupby('layout').agg({
        'both_success': ['mean', 'std'],
        'paralysis': 'mean',
        'cell_collision': 'mean',
        'timesteps': 'mean'
    }).round(3)

    summary.columns = ['Success (mean)', 'Success (std)', 'Paralysis', 'Collision', 'Avg Steps']

    # Save to CSV
    save_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(save_path)
    print(f"Saved: {save_path}")

    # Also print
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate empathy sweep analysis plots")
    parser.add_argument("csv_path", nargs='?', default=None,
                       help="Path to results CSV file")
    parser.add_argument("--latest", action="store_true",
                       help="Use most recent results file")
    parser.add_argument("--output-dir", default="results/figs",
                       help="Output directory for plots")

    args = parser.parse_args()

    # Find input file
    if args.latest or args.csv_path is None:
        csv_path = find_latest_results()
        print(f"Using latest results: {csv_path}")
    else:
        csv_path = args.csv_path

    # Load data
    df = load_results(csv_path)

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    print("\nGenerating plots...")

    plot_cooperation_vs_empathy(df, output_dir)
    plot_asymmetry_heatmaps(df, output_dir)
    plot_complexity_analysis(df, output_dir)
    plot_role_effects(df, output_dir)
    plot_paralysis_phases(df, output_dir)
    plot_exploitation_analysis(df, output_dir)
    generate_summary_table(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

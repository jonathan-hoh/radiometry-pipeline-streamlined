#!/usr/bin/env python3
"""
QUEST Attitude Determination Visualization

Creates a schematic illustration of the QUEST algorithm showing observed vectors,
catalog vectors, optimal rotation alignment, eigenvalue spectra, quaternion
representation, and Monte Carlo uncertainty ellipses.

Usage: python tools/quest_attitude_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.linalg as la
from matplotlib.patches import Ellipse

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
    ])

def generate_test_vectors():
    """Generate test observed and reference vectors for QUEST demonstration"""
    # Reference vectors (catalog - inertial frame)
    reference_vectors = np.array([
        [1, 0, 0],      # Vector to star 1
        [0, 1, 0],      # Vector to star 2  
        [0.7071, 0.7071, 0],  # Vector to star 3
        [0.5774, 0.5774, 0.5774]  # Vector to star 4
    ])
    
    # True rotation (30° around z-axis, 20° around x-axis)
    theta_z = np.radians(30)
    theta_x = np.radians(20)
    
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    true_rotation = Rx @ Rz
    
    # Observed vectors (body frame) - perfect case
    observed_vectors = (true_rotation @ reference_vectors.T).T
    
    # Add small amount of noise for realism
    noise_level = 0.02
    observed_vectors += np.random.normal(0, noise_level, observed_vectors.shape)
    
    # Renormalize
    observed_vectors = observed_vectors / np.linalg.norm(observed_vectors, axis=1, keepdims=True)
    
    return reference_vectors, observed_vectors, true_rotation

def quest_algorithm(observed_vectors, reference_vectors, weights=None):
    """Simplified QUEST algorithm implementation"""
    n_vectors = len(observed_vectors)
    
    if weights is None:
        weights = np.ones(n_vectors)
    
    # Build attitude profile matrix K
    B = np.zeros((3, 3))
    for i in range(n_vectors):
        B += weights[i] * np.outer(observed_vectors[i], reference_vectors[i])
    
    S = B + B.T
    sigma = np.trace(B)
    
    # Z vector
    Z = np.array([
        B[1,2] - B[2,1],
        B[2,0] - B[0,2], 
        B[0,1] - B[1,0]
    ])
    
    # K matrix
    K = np.block([
        [S - sigma*np.eye(3), Z.reshape(-1,1)],
        [Z.reshape(1,-1), sigma]
    ])
    
    # Find largest eigenvalue and corresponding eigenvector
    eigenvals, eigenvecs = la.eigh(K)
    max_idx = np.argmax(eigenvals)
    optimal_quaternion = eigenvecs[:, max_idx]
    
    # Ensure positive scalar part
    if optimal_quaternion[3] < 0:
        optimal_quaternion = -optimal_quaternion
    
    return optimal_quaternion, eigenvals, K

def monte_carlo_quest_uncertainty(reference_vectors, true_rotation, n_trials=100):
    """Generate Monte Carlo trials to estimate QUEST uncertainty"""
    quaternions = []
    
    for trial in range(n_trials):
        # Add noise to observed vectors
        noise_level = 0.01
        observed_vectors = (true_rotation @ reference_vectors.T).T
        observed_vectors += np.random.normal(0, noise_level, observed_vectors.shape)
        observed_vectors = observed_vectors / np.linalg.norm(observed_vectors, axis=1, keepdims=True)
        
        # Run QUEST
        q_est, _, _ = quest_algorithm(observed_vectors, reference_vectors)
        quaternions.append(q_est)
    
    return np.array(quaternions)

def create_quest_visualization():
    """Create comprehensive QUEST attitude determination visualization"""
    # Generate test data
    np.random.seed(42)
    ref_vectors, obs_vectors, true_rotation = generate_test_vectors()
    
    # Run QUEST algorithm
    optimal_quat, eigenvals, K_matrix = quest_algorithm(obs_vectors, ref_vectors)
    
    # Monte Carlo uncertainty analysis
    mc_quaternions = monte_carlo_quest_uncertainty(ref_vectors, true_rotation)
    
    # Create figure with complex layout
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1],
                         hspace=0.3, wspace=0.3)
    
    # Main 3D visualization showing vector alignment
    ax_main = fig.add_subplot(gs[0, :2], projection='3d')
    
    # Plot reference vectors (catalog)
    origin = np.array([0, 0, 0])
    colors = ['red', 'green', 'blue', 'purple']
    
    for i, (ref_vec, color) in enumerate(zip(ref_vectors, colors)):
        ax_main.quiver(origin[0], origin[1], origin[2], 
                      ref_vec[0], ref_vec[1], ref_vec[2],
                      color=color, alpha=0.7, linewidth=3, 
                      arrow_length_ratio=0.1, label=f'Catalog {i+1}')
    
    # Plot observed vectors (rotated)
    offset = np.array([3, 0, 0])  # Offset for visual separation
    for i, (obs_vec, color) in enumerate(zip(obs_vectors, colors)):
        ax_main.quiver(offset[0], offset[1], offset[2],
                      obs_vec[0], obs_vec[1], obs_vec[2],
                      color=color, alpha=0.9, linewidth=3,
                      arrow_length_ratio=0.1, linestyle='--')
    
    # Add curved arrows showing rotation
    theta = np.linspace(0, 2*np.pi, 50)
    for i in range(3):
        radius = 0.3
        x_curve = offset[0]/2 + radius * np.cos(theta)
        y_curve = radius * np.sin(theta) * (0.5 if i==1 else 0.3)
        z_curve = np.full_like(theta, i * 0.4 - 0.4)
        ax_main.plot(x_curve, y_curve, z_curve, 'orange', alpha=0.6, linewidth=2)
    
    # Labels and formatting
    ax_main.text(0, 0, -1.5, 'Reference Vectors\n(Catalog/Inertial)', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    ax_main.text(3, 0, -1.5, 'Observed Vectors\n(Body Frame)', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    ax_main.text(1.5, 0, 1, 'QUEST\nOptimal Rotation', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.8))
    
    ax_main.set_xlabel('X')
    ax_main.set_ylabel('Y') 
    ax_main.set_zlabel('Z')
    ax_main.set_title('Vector Alignment via QUEST Algorithm', 
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.legend(loc='upper right')
    
    # Eigenvalue spectrum
    ax_eigen = fig.add_subplot(gs[0, 2])
    eigenval_indices = np.arange(len(eigenvals))
    bars = ax_eigen.bar(eigenval_indices, eigenvals, 
                       color=['red' if i == np.argmax(eigenvals) else 'blue' 
                             for i in range(len(eigenvals))],
                       alpha=0.7, edgecolor='black')
    
    ax_eigen.set_xlabel('Eigenvalue Index')
    ax_eigen.set_ylabel('Eigenvalue')
    ax_eigen.set_title('QUEST Eigenvalue Spectrum', fontweight='bold')
    ax_eigen.grid(True, alpha=0.3)
    
    # Highlight maximum eigenvalue
    max_idx = np.argmax(eigenvals)
    ax_eigen.text(max_idx, eigenvals[max_idx], 
                 f'λ_max = {eigenvals[max_idx]:.4f}',
                 ha='center', va='bottom', fontweight='bold')
    
    # Quaternion representation
    ax_quat = fig.add_subplot(gs[0, 3])
    
    # Draw quaternion as unit sphere with components
    q0, q1, q2, q3 = optimal_quat
    
    # Quaternion components as bar chart
    quat_labels = ['q₀\n(scalar)', 'q₁\n(x)', 'q₂\n(y)', 'q₃\n(z)']
    quat_values = [q0, q1, q2, q3]
    quat_colors = ['gold', 'red', 'green', 'blue']
    
    bars = ax_quat.bar(range(4), quat_values, color=quat_colors, alpha=0.7, edgecolor='black')
    ax_quat.set_xticks(range(4))
    ax_quat.set_xticklabels(quat_labels, fontsize=10)
    ax_quat.set_ylabel('Quaternion Component')
    ax_quat.set_title('Optimal Quaternion', fontweight='bold')
    ax_quat.grid(True, alpha=0.3)
    ax_quat.axhline(y=0, color='black', linewidth=0.5)
    
    # Add values on bars
    for bar, val in zip(bars, quat_values):
        height = bar.get_height()
        ax_quat.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
    
    # Monte Carlo error ellipses
    ax_error = fig.add_subplot(gs[1, :2])
    
    # Project quaternions to 2D for visualization (use q1, q2 components)
    q1_vals = mc_quaternions[:, 1]
    q2_vals = mc_quaternions[:, 2]
    
    # Calculate covariance and error ellipse
    cov_matrix = np.cov(q1_vals, q2_vals)
    eigenvals_cov, eigenvecs_cov = la.eigh(cov_matrix)
    
    # Error ellipse (2-sigma)
    angle = np.degrees(np.arctan2(eigenvecs_cov[1, 1], eigenvecs_cov[0, 1]))
    width = 2 * np.sqrt(eigenvals_cov[1]) * 4  # 2-sigma
    height = 2 * np.sqrt(eigenvals_cov[0]) * 4
    
    ellipse = Ellipse((np.mean(q1_vals), np.mean(q2_vals)), 
                     width, height, angle=angle, 
                     facecolor='red', alpha=0.3, edgecolor='red', linewidth=2)
    ax_error.add_patch(ellipse)
    
    # Scatter plot of Monte Carlo trials
    ax_error.scatter(q1_vals, q2_vals, alpha=0.6, s=20, color='blue')
    ax_error.scatter(optimal_quat[1], optimal_quat[2], 
                    color='red', s=100, marker='*', 
                    edgecolors='black', linewidth=1, label='QUEST Solution')
    
    ax_error.set_xlabel('q₁ (Quaternion X Component)')
    ax_error.set_ylabel('q₂ (Quaternion Y Component)')
    ax_error.set_title('Monte Carlo Uncertainty Analysis\n(2σ Error Ellipse)', fontweight='bold')
    ax_error.grid(True, alpha=0.3)
    ax_error.legend()
    ax_error.axis('equal')
    
    # Algorithm flow diagram
    ax_flow = fig.add_subplot(gs[1, 2:])
    ax_flow.axis('off')
    
    # Create flow diagram
    boxes = [
        (0.1, 0.8, 0.25, 0.15, 'Observed\nVectors\n(Body Frame)', 'lightgreen'),
        (0.1, 0.4, 0.25, 0.15, 'Reference\nVectors\n(Catalog)', 'lightblue'),
        (0.5, 0.6, 0.25, 0.15, 'Build K Matrix\n(Attitude Profile)', 'yellow'),
        (0.8, 0.6, 0.15, 0.15, 'Eigenvalue\nDecomposition', 'orange'),
        (0.5, 0.2, 0.25, 0.15, 'Optimal\nQuaternion', 'pink')
    ]
    
    for x, y, w, h, text, color in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, 
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black',
                                    alpha=0.7)
        ax_flow.add_patch(rect)
        ax_flow.text(x + w/2, y + h/2, text, ha='center', va='center',
                    fontsize=9, fontweight='bold')
    
    # Add arrows
    arrows = [
        ((0.35, 0.87), (0.5, 0.72)),   # Observed to K
        ((0.35, 0.47), (0.5, 0.62)),   # Reference to K  
        ((0.75, 0.67), (0.8, 0.67)),   # K to Eigenvalue
        ((0.87, 0.6), (0.7, 0.35)),    # Eigenvalue to Quaternion
    ]
    
    for start, end in arrows:
        ax_flow.annotate('', xy=end, xytext=start,
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.set_title('QUEST Algorithm Flow', fontsize=14, fontweight='bold')
    
    # Performance metrics
    ax_metrics = fig.add_subplot(gs[2, :])
    ax_metrics.axis('off')
    
    # Calculate performance metrics
    rotation_est = quaternion_to_rotation_matrix(optimal_quat)
    rotation_error = la.norm(rotation_est - true_rotation, 'fro')
    attitude_error_deg = np.degrees(np.arccos((np.trace(rotation_est @ true_rotation.T) - 1) / 2))
    
    mc_std_q1 = np.std(mc_quaternions[:, 1])
    mc_std_q2 = np.std(mc_quaternions[:, 2])
    
    metrics_text = (
        f"QUEST Performance Metrics:\n"
        f"• Attitude Error: {attitude_error_deg:.3f}° (rotation accuracy)\n"
        f"• Rotation Matrix Error: {rotation_error:.4f} (Frobenius norm)\n"
        f"• Quaternion Uncertainty: σ(q₁)={mc_std_q1:.4f}, σ(q₂)={mc_std_q2:.4f}\n"
        f"• Maximum Eigenvalue: {np.max(eigenvals):.4f} (optimization confidence)\n"
        f"• Optimal Quaternion: [{optimal_quat[0]:.3f}, {optimal_quat[1]:.3f}, {optimal_quat[2]:.3f}, {optimal_quat[3]:.3f}]"
    )
    
    ax_metrics.text(0.05, 0.5, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Main title
    fig.suptitle('QUEST Attitude Determination: Vector Alignment, Optimization, and Uncertainty\n' +
                'Quaternion ESTimator for Spacecraft Attitude from Star Tracker Measurements', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def main():
    """Generate QUEST attitude determination visualization"""
    print("Generating QUEST attitude determination visualization...")
    print("This may take a moment due to Monte Carlo analysis...")
    
    fig = create_quest_visualization()
    
    # Save figure
    output_path = "quest_attitude_determination.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved as: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
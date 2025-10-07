import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# Function to compute approximate optimal rotation matrix via SVD (simple Wahba solver)
def compute_optimal_rotation(b_vectors, r_vectors, weights=None):
    if weights is None:
        weights = np.ones(len(b_vectors))
    H = np.dot((weights[:, np.newaxis] * r_vectors).T, b_vectors)  # Weighted cross-covariance
    U, S, Vt = np.linalg.svd(H)
    rot_matrix = np.dot(Vt.T, U.T)
    if np.linalg.det(rot_matrix) < 0:  # Ensure proper rotation
        Vt[-1, :] *= -1
        rot_matrix = np.dot(Vt.T, U.T)
    return rot_matrix

# Sample data: 4 observed unit vectors (b_i) and reference unit vectors (r_i)
b_vectors = np.array([
    [0.8, 0.4, 0.2], [0.2, 0.8, 0.4], [-0.4, 0.2, 0.8], [0.4, -0.2, 0.8]
])
b_vectors = b_vectors / np.linalg.norm(b_vectors, axis=1)[:, np.newaxis]

r_vectors = np.array([
    [0.7, 0.5, 0.3], [0.3, 0.7, 0.5], [-0.5, 0.3, 0.7], [0.5, -0.3, 0.7]
])
r_vectors = r_vectors / np.linalg.norm(r_vectors, axis=1)[:, np.newaxis]

# Compute optimal rotation (assuming equal weights for simplicity)
rot_matrix = compute_optimal_rotation(b_vectors, r_vectors)

# Apply rotation to reference vectors for alignment
aligned_r_vectors = np.dot(r_vectors, rot_matrix.T)  # Rotate r_i to align with b_i

# Subtle unit sphere mesh
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

# Figure 1: Misaligned (saved as separate PNG)
fig1 = plt.figure(figsize=(8, 8), dpi=300)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title("Misaligned (Before Rotation)", fontsize=16)

for i in range(len(b_vectors)):
    # Observed (blue)
    ax1.quiver(0, 0, 0, b_vectors[i, 0], b_vectors[i, 1], b_vectors[i, 2],
               color='blue', label='Observed b_i' if i == 0 else "", linewidth=2)
    # Reference (red)
    ax1.quiver(0, 0, 0, r_vectors[i, 0], r_vectors[i, 1], r_vectors[i, 2],
               color='red', label='Reference r_i' if i == 0 else "", linewidth=2)
    # Difference vector (dashed yellow, from b_i tip to r_i tip)
    diff_start = b_vectors[i]
    diff_end = r_vectors[i] - b_vectors[i]
    ax1.quiver(diff_start[0], diff_start[1], diff_start[2],
               diff_end[0], diff_end[1], diff_end[2],
               color='yellow', linestyle='dashed', linewidth=1.5,
               label='Differences' if i == 0 else "")

# Color-coded equation (yellow to match differences)
ax1.text(0, 0, 1.3, r"Minimize $L(R) = \sum w_i \| \mathbf{b}_i - R \mathbf{r}_i \|^2$", 
         fontsize=14, ha='center', color='yellow')

# Add sphere
ax1.plot_wireframe(x, y, z, color='gray', alpha=0.3, linewidth=1.0)

ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_zlim([-1, 1])
ax1.legend(loc='upper right')
ax1.axis('off')  # Hide axes for cleaner look
plt.tight_layout()
plt.savefig('wahba_misaligned.png', dpi=300)
plt.close(fig1)  # Close to avoid display

# Figure 2: Aligned (saved as separate PNG)
fig2 = plt.figure(figsize=(8, 8), dpi=300)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("Aligned (After Optimal Rotation)", fontsize=16)

for i in range(len(b_vectors)):
    # Observed (blue)
    ax2.quiver(0, 0, 0, b_vectors[i, 0], b_vectors[i, 1], b_vectors[i, 2],
               color='blue', label='Observed b_i' if i == 0 else "", linewidth=2)
    # Aligned reference (green)
    ax2.quiver(0, 0, 0, aligned_r_vectors[i, 0], aligned_r_vectors[i, 1], aligned_r_vectors[i, 2],
               color='green', label='Rotated r_i' if i == 0 else "", linewidth=2)

# Equation/label
ax2.text(0, 0, 1.3, "Optimal Rotation R (or q)", fontsize=14, ha='center')

# Add curved rotation arrow (using annotation with correct styles)
ax2.annotate('', xy=(0.8, 0.5), xycoords='axes fraction',
             xytext=(0.2, 0.5), textcoords='axes fraction',
             arrowprops=dict(
                 arrowstyle='->',  # Arrowhead style
                 connectionstyle='arc3,rad=0.5',  # Curved connection
                 color='black',
                 linewidth=3,
                 shrinkA=5, shrinkB=5  # Shrink to avoid overlapping edges
             ))

# Add sphere
ax2.plot_wireframe(x, y, z, color='gray', alpha=0.3, linewidth=1.0)

ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim([-1, 1])
ax2.legend(loc='upper right')
ax2.axis('off')  # Hide axes for cleaner look
plt.tight_layout()
plt.savefig('wahba_aligned.png', dpi=300)
plt.close(fig2)  # Close to avoid display
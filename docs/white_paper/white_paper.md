# Star Tracker Radiometry Simulation: Mathematical Foundations

## Introduction

This white paper provides a detailed mathematical exposition of the key physical and algorithmic components in the star tracker radiometry simulation pipeline. The simulation serves as a digital twin for Basic Angle Star Tracker (BAST) systems, enabling performance prediction for spacecraft attitude determination. We focus on seven core mathematical sections that drive the codebase, deriving the governing equations from first principles while maintaining consistency with the implemented software architecture.

Each section includes:
- Problem statement
- Critical parameters
- Step-by-step mathematical derivation
- Connection to codebase implementation
- Physical insights and limitations

## Section 1: Scene Generation Mathematics

### Problem Statement

To simulate realistic star tracker scenarios, we must transform arbitrary spacecraft attitudes into precise star positions on the detector. This requires converting celestial coordinates through multiple reference frames while preserving angular relationships for pattern matching.

### Critical Parameters

- Rotation matrices $R(\phi, \theta, \psi)$
- Quaternion components $[q_0, q_1, q_2, q_3]$
- Focal plane coordinates $(x_f, y_f)$
- Detector pixel positions $(u, v)$
- Angular preservation error $\Delta \theta$

### Mathematical Derivation

The coordinate transformation chain begins with stars in celestial coordinates (right ascension $\alpha$, declination $\delta$) and ends with pixel positions on the detector. We derive this step-by-step, incorporating spacecraft attitude via rotations.

#### 1. Spherical to Cartesian Inertial Vectors

Stars are cataloged in the inertial celestial frame using spherical coordinates. Convert to Cartesian unit vectors:

$$
\mathbf{v}_I = \begin{pmatrix}
\cos \delta \cos \alpha \\
\cos \delta \sin \alpha \\
\sin \delta
\end{pmatrix}
$$

This is a unit vector ($\|\mathbf{v}_I\| = 1$) pointing from the origin to the star on the celestial sphere. The boresight (camera pointing direction) is similarly converted to $\mathbf{v}_B$.

**Code Connection**: Implemented in `radec_to_inertial_vector` in `attitude_transform.py`.

#### 2. Attitude Representation and Rotation Matrices

Spacecraft attitude is represented by Euler angles $(\phi, \theta, \psi)$ (roll, pitch, yaw) or quaternions. We derive the rotation matrix $R$ that transforms from inertial to camera frame.

**From Euler Angles (ZYX Convention)**:

Individual rotations:
- Yaw (Z): $R_z(\psi) = \begin{pmatrix} \cos \psi & -\sin \psi & 0 \\ \sin \psi & \cos \psi & 0 \\ 0 & 0 & 1 \end{pmatrix}$
- Pitch (Y): $R_y(\theta) = \begin{pmatrix} \cos \theta & 0 & \sin \theta \\ 0 & 1 & 0 \\ -\sin \theta & 0 & \cos \theta \end{pmatrix}$
- Roll (X): $R_x(\phi) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos \phi & -\sin \phi \\ 0 & \sin \phi & \cos \phi \end{pmatrix}$

Composite: $R = R_z(\psi) R_y(\theta) R_x(\phi)$

**From Quaternions**:

For unit quaternion $\mathbf{q} = [q_0, q_1, q_2, q_3]$,

$$
R = \begin{pmatrix}
q_0^2 + q_1^2 - q_2^2 - q_3^2 & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\
2(q_1 q_2 + q_0 q_3) & q_0^2 - q_1^2 + q_2^2 - q_3^2 & 2(q_2 q_3 - q_0 q_1) \\
2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & q_0^2 - q_1^2 - q_2^2 + q_3^2
\end{pmatrix}
$$

**Transformation**: $\mathbf{v}_C = R \mathbf{v}_I$ (camera frame vector). Stars with $v_{C,z} \leq 0$ are behind the camera and filtered.

**Angular Preservation Proof**: Rotations are orthogonal ($R^T R = I$), so inner products are preserved: $\mathbf{v}_{I1}^T \mathbf{v}_{I2} = \mathbf{v}_{C1}^T \mathbf{v}_{C2}$, hence angles $\theta = \cos^{-1}(\mathbf{v}_1^T \mathbf{v}_2)$ are invariant.

**Code Connection**: `euler_to_rotation_matrix` and `quaternion_to_rotation_matrix` in `attitude_transform.py`; applied in `transform_to_camera_frame`.

#### 3. Pinhole Camera Projection to Focal Plane

Project camera-frame vectors to 2D focal plane using pinhole model:

$$
x_f = f \frac{v_{C,x}}{v_{C,z}}, \quad y_f = f \frac{v_{C,y}}{v_{C,z}}
$$

where $f$ is focal length (in microns). This is perspective division.

**Code Connection**: `project_to_focal_plane` in `attitude_transform.py`.

#### 4. Focal Plane to Pixel Coordinates

Convert focal plane (microns) to pixels:

$$
u = u_0 + \frac{x_f}{p}, \quad v = v_0 + \frac{y_f}{p}
$$

where $(u_0, v_0)$ is principal point (detector center), $p$ is pixel pitch (5.5 µm for CMV4000).

Filter positions outside detector bounds (e.g., 0 to 2048 pixels) with margin.

**Code Connection**: `focal_plane_to_pixels` and `filter_detector_bounds` in `attitude_transform.py`.

#### 5. Integrated Gnomonic Projection (Legacy Mode)

For boresight-centered projections without full attitude, use gnomonic:

Tangent plane basis: East $\mathbf{e} = \frac{\mathbf{v}_B \times \mathbf{n}}{\|\mathbf{v}_B \times \mathbf{n}\|}$, North $\mathbf{n}' = \mathbf{v}_B \times \mathbf{e}$ (north pole $\mathbf{n} = [0,0,1]$).

Projected: $\mathbf{p} = \frac{\mathbf{v}_I}{\mathbf{v}_I \cdot \mathbf{v}_B} - \mathbf{v}_B$

Coordinates: $x_t = \mathbf{p} \cdot \mathbf{e}$, $y_t = \mathbf{p} \cdot \mathbf{n}'$

Then to pixels as above.

**Code Connection**: `_sky_to_detector` in `scene_generator.py`.

### Physical Insights and Limitations

- **Accuracy**: Sub-pixel precision requires floating-point positions; angular error $\Delta \theta \approx \frac{p}{f}$ radians per pixel.
- **Limitations**: Assumes ideal pinhole (no distortion); valid for small fields-of-view (<20°).
- **Preservation**: Essential for pattern matching; validated by checking pairwise angles.

This chain enables multi-star scene generation with arbitrary attitudes, preserving geometry for downstream algorithms.

## Section 2: Radiometric Chain Physics

### Problem Statement

To model light interaction with our camera sensor, we need physical constraints linking stellar magnitude to detector response. This establishes the signal strength available for detection and determines noise-limited performance bounds.

### Critical Parameters

- Photon flux $\Phi$ (photons/s)
- Quantum efficiency $QE$
- Optical transmission $\tau$
- Detector full well capacity $N_{FW}$
- Dark current $I_d$
- Signal-to-noise ratio $SNR$

### Mathematical Derivation

The radiometric chain converts stellar apparent magnitude to photon count at the detector, incorporating optical and sensor properties. We derive this from first principles, focusing on broadband visible light.

#### 1. Stellar Flux from Apparent Magnitude

Apparent magnitude $m$ follows the Pogson scale, where flux ratio for magnitude difference $\Delta m$ is $10^{-0.4 \Delta m} = 2.512^{-\Delta m}$.

Reference: Sun's apparent magnitude $m_\odot = -26.74$, solar constant $S = 1366$ W/m².

For zero-magnitude star flux $F_0$ (W/m²) in passband:

First, photon energy at central wavelength $\lambda_c = (\lambda_{min} + \lambda_{max})/2$:

$$
E_{ph} = \frac{h c}{\lambda_c}
$$

where $h$ is Planck's constant, $c$ is speed of light.

Solar photon flux $\Phi_\odot = S / E_{ph}$ (photons/s/m², approximated for broadband).

Zero-magnitude photon flux:

$$
\Phi_0 = \Phi_\odot \times 2.512^{m_\odot}
$$

For star of magnitude $m$:

$$
\Phi = \Phi_0 \times 2.512^{-m}
$$

**Simplification in Code**: Uses $2.5^{-m}$ instead of 2.512 for approximation, and scales from solar values directly.

**Code Connection**: In `star.calculate_flux`, it computes M0_flux_const = (solar_flux_constant * 1e-6) / (2.5 ** abs(solar_magnitude)), then ph_flux = M0_flux_const * (2.5 ** (-1 * self.magnitude)) / ph_energy. Note 1e-6 adjusts units.

#### 2. Collected Photons at Aperture

Aperture area $A = \pi (D/2)^2$, where $D$ is diameter.

Incident photons/s: $\Phi \times A$

After optical transmission $\tau$:

$\Phi_{opt} = \Phi \times A \times \tau$

Over integration time $t$ (s):

Total photons $N_{ph} = \Phi_{opt} \times t$

**Code Connection**: `calculate_optical_signal` computes sig = star.flux * camera.aperature_area * camera.optic.transmission * (scene.int_time / 1000)

#### 3. Detector Response and Electron Generation

Quantum efficiency $QE$ (electrons/photon) converts photons to photoelectrons:

$N_e = N_{ph} \times QE$

Noise sources added later (Section 3).

Full well $N_{FW}$ limits maximum $N_e$ per pixel.

Dark current $I_d$ (electrons/s/pixel) adds $I_d \times t$ electrons.

**Code Connection**: QE is a parameter in fpa class, but signal calculation stops at photons; noise handled separately in simulations.

#### 4. Noise Contributions

- Shot noise: Poisson, variance $N_{ph}$
- Dark noise: variance $I_d t$
- Read noise: Gaussian, variance $\sigma_r^2$

Total noise variance $\sigma^2 = N_e + I_d t + \sigma_r^2$ (after QE).

SNR = $N_e / \sigma$

**Detection Threshold**: Typically requires SNR > 5-10 for reliable detection.

**Code Connection**: While not explicitly in calculate_optical_signal, used in poisson simulations elsewhere.

#### 5. Spectral Integration

For accuracy, integrate over spectrum:

$\Phi = \int_{\lambda_{min}}^{\lambda_{max}} F(\lambda) \frac{\lambda}{h c} d\lambda$

But code uses central wavelength approximation.

### Physical Insights and Limitations

- **Magnitude Scale**: Each +1 magnitude is 2.512× fainter; code uses 2.5 approximation (1.7% error).
- **Assumptions**: Blackbody-like stars, uniform QE; ignores atmospheric effects.
- **Limits**: Saturation at bright magnitudes ($N_e > N_{FW}$); faint stars limited by noise.
- **SNR Prediction**: Guides detection thresholds; e.g., for given $m$, predict if detectable.

This chain provides the input signal for image formation, directly impacting all downstream performance.

## Section 3: Statistical Image Formation

### Problem Statement

Real detectors exhibit shot noise and readout noise that fundamentally limit centroiding precision. We must model Poisson photon statistics and detector noise to predict achievable accuracy and validate sub-pixel PSF placement methods.

### Critical Parameters

- Poisson parameter $\lambda$
- Read noise variance $\sigma_r^2$
- Pixel response non-uniformity $\epsilon_{PRNU}$
- PSF sampling error $\Delta x_{sample}$
- Total noise equivalent electrons $N_{noise}$

### Mathematical Derivation

Image formation combines optical PSF with stochastic photon arrival and detector effects. We derive the probabilistic model for pixel intensities.

#### 1. PSF as Probability Distribution

The PSF $P(i,j)$ gives the probability a photon lands in pixel (i,j). Normalized:

$$
\sum_{i,j} P(i,j) = 1
$$

From Zemax data, normalize intensity matrix.

**Code Connection**: `normalize_psf` in psf_photon_simulation.py divides by sum if non-zero.

#### 2. Poisson Photon Arrival

For total photons $N_{ph}$, expected in pixel (i,j): $\lambda_{i,j} = N_{ph} P(i,j)$

Actual count $k_{i,j} \sim Poisson(\lambda_{i,j})$, probability:

$$
Pr(k_{i,j} = k) = \frac{\lambda_{i,j}^k e^{-\lambda_{i,j}}}{k!}
$$

Variance = $\lambda_{i,j}$ (shot noise).

For Monte Carlo, simulate per photon or per pixel.

**Code Connection**: `simulate_psf_with_poisson_noise` computes expected_counts = psf_normalized * num_photons, then np.random.poisson(expected_counts) for each trial.

Alternative: `simulate_photon_distribution` samples photon locations from flattened PSF probabilities.

#### 3. Detector Response Modeling

Add gain $g$ (e-/photon, but QE already in $N_{ph}$), offset $o$, read noise $r \sim \mathcal{N}(0, \sigma_r^2)$, dark current $d \sim Poisson(I_d t)$.

Digital count:

$$
DN_{i,j} = g (k_{i,j} + d_{i,j}) + o + r_{i,j}
$$

Clamped to [0, 2^{bits}-1].

**Code Connection**: Not explicitly in simulation; poisson_image is the output, but comments mention adding noise.

#### 4. Sub-Pixel PSF Placement and Interpolation

For position (x,y), shift PSF using fractional pixel shifts. Use bilinear or spline interpolation.

Error $\Delta x_{sample} \approx \frac{p}{12}$ for bilinear (p pixel size).

**Code Connection**: In multi_star_radiometry.py, _place_psf_at_position_subpixel uses scipy.ndimage.shift with order=1 (linear).

#### 5. Total Noise and Error Propagation

Total variance per pixel: $\sigma_{i,j}^2 = g^2 (\lambda_{i,j} + I_d t) + \sigma_r^2$

For integrated signal, SNR = total electrons / sqrt(total variance).

Reconstruction error propagates through sampling.

**Cramér-Rao Bound** for position estimation (Section 4).

**Code Connection**: Simulations compute mean_image and std_image over trials.

### Physical Insights and Limitations

- **Poisson Limit**: For bright stars, shot noise dominates; faint stars by read noise.
- **Sampling**: Fine PSFs require interpolation to avoid aliasing.
- **Limitations**: Assumes independent pixels; ignores PRNU, crosstalk.
- **Validation**: Monte Carlo averages match expected PSF.

This forms the noisy image for detection algorithms.

## Section 4: Detection and Centroiding Theory

### Problem Statement

Extracting sub-pixel star positions from noisy images requires optimal thresholding and moment calculations. The centroiding algorithm must achieve 0.1-0.3 pixel accuracy while rejecting noise and handling varying PSF shapes.

### Critical Parameters

- Detection threshold $T_{detect}$
- Centroid coordinates $(x_c, y_c)$
- Centroiding uncertainty $\sigma_{centroid}$
- Moment calculation weights $w_i$
- Region selection criteria

### Mathematical Derivation

Detection identifies star regions; centroiding computes positions. We derive adaptive methods.

#### 1. Adaptive Thresholding for Detection

To handle varying background, compute local statistics in blocks of size B×B.

For block mean $\mu_b$, std $\sigma_b$:

Threshold $T_b = \mu_b + k \sigma_b$ (k=3-5).

Binary mask: pixels > T_b.

**Code Connection**: In detect_stars_and_calculate_centroids, uses cv2.resize for multi-scale, then local mean/std for threshold.

#### 2. Connected Components Labeling

Label connected regions in binary mask using 8-connectivity.

Filter by area: min_pixels ≤ area ≤ max_pixels (e.g., 3-50).

Select brightest region by total intensity for single-star; all valid for multi.

**Code Connection**: cv2.connectedComponentsWithStats in group_pixels (identify.py), filters by size.

#### 3. Moment-Based Centroiding

For region with intensities I(x,y):

$$
x_c = \frac{\sum x I(x,y)}{\sum I(x,y)}, \quad y_c = \frac{\sum y I(x,y)}{\sum I(x,y)}
$$

This is center of mass.

For uncertainty, variance from photon noise:

$$
\sigma_{x_c}^2 \approx \frac{\sum (x - x_c)^2 I(x,y)}{\left( \sum I(x,y) \right)^2 } \times \sum I(x,y)
$$

(Since var(I) = I for Poisson).

Cramér-Rao lower bound for position estimator:

$$
\sigma_{centroid} \geq \frac{\sigma_{psf}}{\sqrt{N_{ph}}}
$$

where $\sigma_{psf}$ is PSF width.

**Code Connection**: calculate_centroid in identify.py computes weighted moments.

#### 4. Bias and Window Effects

Finite window biases if PSF tails cut off. Optimal window ~3-5 × FWHM.

For Gaussian PSF, bias ~ exp(-r^2 / 2\sigma^2) / r.

**Code Connection**: Size filtering mitigates; adaptive block helps.

#### 5. Alternative: Peak Detection

For multi-star, find local maxima above threshold, filter by separation.

**Code Connection**: detect_stars_peak_method in peak_detection.py uses max_filter.

### Physical Insights and Limitations

- **Accuracy**: Sub-pixel via weighting; limited by SNR.
- **Robustness**: Adaptive handles non-uniform background.
- **Limitations**: Assumes isolated stars; crowding needs advanced deblending.
- **Multi-Star**: Peak detection better for overlapping PSFs.

This provides positions for bearing vectors.

## Section 6: Pattern Recognition Mathematics

### Problem Statement

Identifying which observed stars correspond to catalog entries requires geometric pattern matching using inter-star angles. The BAST algorithm must reliably match triangular patterns despite measurement errors and false detections.

### Critical Parameters

- Inter-star angles $\theta_{ij}$
- Angle tolerance $\Delta \theta_{tol}$
- Match confidence $C_{match}$
- Triangle geometric invariants
- False positive probability $P_{FP}$

### Mathematical Derivation

Pattern recognition uses rotation-invariant features like angular separations to match observed triplets to catalog.

#### 1. Inter-Star Angular Distance

For unit vectors $\mathbf{v}_i, \mathbf{v}_j$:

$$
\theta_{ij} = \cos^{-1}(\mathbf{v}_i \cdot \mathbf{v}_j)
$$

On sphere, use spherical trigonometry for accuracy, but dot product suffices for small angles.

**Code Connection**: calculate_vector_angle in match.py uses np.arccos(np.clip(dot, -1,1)).

#### 2. Triangle Geometric Invariants

For triplet angles $a,b,c$ (sorted a ≤ b ≤ c), invariant is tuple (a,b,c), tolerant to rotation.

Catalog precomputes for each star its triplets with neighbors within FOV.

**Code Connection**: triplet_worker in catalog.py computes separations using astropy SkyCoord.separation.rad for all pairs.

#### 3. Matching Triplets

For observed triplet angles $\theta_1, \theta_2, \theta_3$, search catalog for matches where |θ - cat_θ| < Δθ_tol for all.

Confidence $C = 1 - \frac{\sum |\delta \theta|}{\Delta \theta_{tol}}$ or average.

**Code Connection**: find_triplet_match in match.py computes diffs and checks if all < tolerance, then scores.

#### 4. Probabilistic Matching and False Alarms

Probability of random match ~ number of catalog triplets × (Δθ_tol / π)^3.

False positive P_FP minimized by tight tolerance and verification.

**Code Connection**: Not explicit, but min_confidence threshold in match.

#### 5. Combinatorial Optimization: Hungarian Algorithm

For multiple possible assignments, use Hungarian for optimal one-to-one matching minimizing cost (e.g., angle diff).

But code uses greedy: find matches sequentially avoiding used stars.

**Code Connection**: match function iterates combinations, adds matches if found, tracks used indices to ensure bijective.

For full optimization, bijective_matching.py uses Hungarian via scipy.optimize.linear_sum_assignment.

#### 6. Confidence Metrics and Rejection

Post-match, verify geometric consistency, e.g., quaternion from match aligns all stars.

Reject if confidence < threshold.

**Code Connection**: score_match in match.py averages normalized diffs.

Error analysis: Propagate bearing errors to angle uncertainty, impact on P_FP.

### Physical Insights and Limitations

- **Invariance**: Angles preserved under rotation, key for attitude-independence.
- **Robustness**: Tolerates errors via Δθ_tol; multi-match improves reliability.
- **Limitations**: Combinatorial explosion for many stars; assumes no ambiguities.
- **Scalability**: Precomputed catalog essential for speed.

This enables identification for attitude estimation.

## Section 5: Bearing Vector Geometry

### Problem Statement

Converting pixel centroids to 3D unit vectors requires precise camera calibration and coordinate transformations. Bearing vector accuracy directly determines attitude solution precision, making this a critical error propagation node.

### Critical Parameters

- Focal length $f$
- Principal point $(c_x, c_y)$
- Pixel pitch $p$
- Bearing vector components $[\hat{v}_x, \hat{v}_y, \hat{v}_z]$
- Angular error $\Delta \theta_{bearing}$

### Mathematical Derivation

Bearing vectors are unit directions from camera to stars in camera frame.

#### 1. Pixel to Focal Plane Coordinates

From pixel (u,v) to focal plane (x_f, y_f) in microns:

$x_f = (u - c_x) * p$

$y_f = (v - c_y) * p$   (may flip y for convention)

**Code Connection**: In star_tracker_pipeline.calculate_bearing_vectors, uses pixel_to_physical_conversion with pixel_pitch.

#### 2. Pinhole Projection to 3D Vector

Bearing vector:

$$
\hat{v} = \frac{ [x_f, y_f, f] }{ \sqrt{x_f^2 + y_f^2 + f^2} }
$$

f in same units (microns).

**Code Connection**: calculate_bearing_vectors normalizes $[x_um/f_um, y_um/f_um, 1]$.

#### 3. Distortion Correction (if applicable)

For radial distortion, correct $(x_f, y_f)$ before vector.

Simple model: $x_{correct} = x (1 + k r^2)$, etc.

**Code Connection**: Not in base, but identify.py has apply_distortion_correction placeholder.

#### 4. Error Propagation

Centroid error $\delta$ u propagates to angular error:

$\delta \theta \approx \frac{p}{f} \delta u$   (radians)

For $f=40mm$, $p=5.5µm$, $\delta u=0.2$ px → ~1 arcsec.

**Code Connection**: In monte_carlo, errors from centroid std.

#### 5. Calibration Matrix

General form uses intrinsic matrix $\mathbf{K}$, but code assumes ideal pinhole.

### Physical Insights and Limitations

- **Accuracy**: Directly ties to focal length; longer $f$ better precision but smaller FOV.
- **Assumptions**: Ideal pinhole; real systems need distortion model.
- **Limitations**: Ignores temperature effects on $f, p$.
- **Optimization**: Choose $f$ to balance accuracy and star count.

This links detection to matching.

## Section 7: Optimal Attitude Estimation

### Problem Statement

Given matched star pairs, we must find the optimal rotation matrix relating observed and catalog reference frames. The QUEST algorithm solves Wahba's problem to minimize attitude errors while providing uncertainty bounds.

### Critical Parameters

- Davenport K-matrix eigenvalues $\lambda_i$
- Optimal quaternion $\hat{q}$
- Attitude uncertainty $\sigma_q$
- Residual errors $\epsilon_i$
- Confidence bounds

### Mathematical Derivation

Wahba's problem minimizes loss for rotation R: $L(R) = \sum w_i \| b_i - R r_i \|^2$, where $b_i$ observed, $r_i$ reference vectors, $w_i$ weights.

#### 1. Attitude Profile Matrix

Build $B = \sum w_i r_i b_i^T$

Then $S = B + B^T - trace(B) I$, $Z = [B_{23}-B_{32}, B_{31}-B_{13}, B_{12}-B_{21}]$

$K = [ trace(B), Z^T ; Z, S ]$

**Code Connection**: build_davenport_matrix in resolve.py computes $B$, then $K$.

#### 2. QUEST Solution

Optimal q is eigenvector of $K$ with max eigenvalue $λ_max$.

Solve $det(K - λ I) = 0$ for $\lambda$, but in practice compute $eig(K)$.

**Code Connection**: quest_algorithm in resolve.py uses np.linalg.eig(K), takes max real eigenvalue's vector, normalizes.

#### 3. Quaternion to Matrix

R from $q = [w,x,y,z]$:

$$
\begin{align}
R = [[1-2y^2-2z^2, 2xy-2wz, 2xz+2wy],\\
     [2xy+2wz, 1-2x^2-2z^2, 2yz-2wx],\\
     [2xz-2wy, 2yz+2wx, 1-2x^2-2y^2]]
\end{align}     
$$
**Code Connection**: quaternion_to_matrix in resolve.py.

#### 4. Monte Carlo Uncertainty

Perturb bearings with noise ~ centroid error, run QUEST many times, compute $q$ mean/std/cov.

Angular uncertainty ~ $2 * \sigma(q_{vector})$ radians.

Converge when rolling $\sigma$ < tolerance.

**Code Connection**: determine_attitude in monte_carlo_quest.py adds noise, batch computes eig, checks convergence, stats.

#### 5. Confidence and Residuals

Residuals $\epsilon_i = || b_i - R r_i ||$

Confidence from $λ_{max}$ or Monte Carlo spread.

**Code Connection**: eigenvalue in results; validate vs truth.

### Physical Insights and Limitations

- **Optimality**: QUEST near-optimal for Wahba, fast.
- **Uncertainty**: Monte Carlo captures real errors.
- **Limitations**: Assumes equal weights; needs ≥2 non-collinear pairs.
- **Extensions**: Weights by magnitude/SNR improve.

This completes the pipeline to attitude.

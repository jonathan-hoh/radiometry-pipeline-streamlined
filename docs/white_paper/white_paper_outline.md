# Section Breakdown

## **Section 1: Scene Generation Mathematics**

*Problem*: To simulate realistic star tracker scenarios, we must transform arbitrary spacecraft attitudes into precise star positions on the detector. This requires converting celestial coordinates through multiple reference frames while preserving angular relationships for pattern matching.

*Critical Parameters*: Rotation matrices $R(\phi,\theta,\psi)$, quaternion components $[q_0,q_1,q_2,q_3]$, focal plane coordinates $(x_f,y_f)$, detector pixel positions $(u,v)$, angular preservation error $\Delta\theta$.

We'll derive the complete coordinate transformation chain starting with spherical celestial coordinates (RA, Dec) and ending with pixel positions. The mathematics begins with converting spherical coordinates to Cartesian inertial vectors, then applying rotation matrices for arbitrary spacecraft attitudes. We'll establish the relationship between Euler angles and rotation matrices, derive quaternion-to-matrix conversions, and prove angular preservation through orthogonal transformations. The focal plane projection follows pinhole camera geometry, relating 3D camera-frame vectors to 2D focal plane coordinates via perspective division. Finally, we'll derive pixel coordinate transformations incorporating detector geometry and principal point offsets.

## **Section 2: Radiometric Chain Physics**

*Problem*: To model light interaction with our camera sensor, we need physical constraints linking stellar magnitude to detector response. This establishes the signal strength available for detection and determines noise-limited performance bounds.

*Critical Parameters*: Photon flux $\Phi$ (photons/s), quantum efficiency $QE$, optical transmission $\tau$, detector full well capacity $N_{FW}$, dark current $I_d$, signal-to-noise ratio $SNR$.

The mathematical framework starts with stellar flux calculations from magnitude-to-photon-rate conversions using the Pogson magnitude scale. We'll derive the complete signal chain: stellar flux → telescope collection → optical transmission → detector quantum efficiency → electron generation. Key derivations include the fundamental radiometry equation linking apparent magnitude to photon flux, integration over CMV4000 spectral response curves, and signal accumulation over exposure time. We'll establish noise source contributions including shot noise (Poisson statistics), thermal noise (temperature-dependent dark current), and read noise (electronic system limitations). The mathematics culminates in signal-to-noise ratio expressions that predict detection thresholds.

## **Section 3: Statistical Image Formation**

*Problem*: Real detectors exhibit shot noise and readout noise that fundamentally limit centroiding precision. We must model Poisson photon statistics and detector noise to predict achievable accuracy and validate sub-pixel PSF placement methods.

*Critical Parameters*: Poisson parameter $\lambda$, read noise variance $\sigma_r^2$, pixel response non-uniformity $\epsilon_{PRNU}$, PSF sampling error $\Delta x_{sample}$, total noise equivalent electrons $N_{noise}$.

We'll develop the mathematical framework for realistic image synthesis starting with Poisson shot noise theory. The derivations begin with probability distributions for photon arrival statistics, then extend to detector response modeling including gain, offset, and saturation effects. Critical mathematics includes PSF convolution theory for optical blur, numerical interpolation methods for sub-pixel PSF placement, and error propagation through the sampling process. We'll derive expressions for total image noise combining shot noise, dark current, and read noise contributions. The mathematical treatment addresses pixel integration effects and establishes theoretical limits for PSF reconstruction accuracy.

## **Section 4: Detection and Centroiding Theory**

*Problem*: Extracting sub-pixel star positions from noisy images requires optimal thresholding and moment calculations. The centroiding algorithm must achieve 0.1-0.3 pixel accuracy while rejecting noise and handling varying PSF shapes.

*Critical Parameters*: Detection threshold $T_{detect}$, centroid coordinates $(x_c,y_c)$, centroiding uncertainty $\sigma_{centroid}$, moment calculation weights $w_i$, region selection criteria.

The mathematical development covers optimal detection theory starting with likelihood ratio tests for star detection. We'll derive adaptive thresholding algorithms that account for local background variations and establish theoretical detection performance bounds. The centroiding mathematics centers on moment-based estimators, deriving intensity-weighted centroid expressions and their associated uncertainty bounds. Key derivations include the relationship between SNR and centroiding precision, bias analysis for finite aperture effects, and optimal window sizing criteria. We'll establish the mathematical connection between photon noise and ultimate centroiding accuracy limits, including Cramér-Rao bounds for unbiased estimators.

## **Section 5: Bearing Vector Geometry**

*Problem*: Converting pixel centroids to 3D unit vectors requires precise camera calibration and coordinate transformations. Bearing vector accuracy directly determines attitude solution precision, making this a critical error propagation node.

*Critical Parameters*: Focal length $f$, principal point $(c_x,c_y)$, pixel pitch $p$, bearing vector components $[\hat{v}_x,\hat{v}_y,\hat{v}z]$*,* angular error $*\Delta\theta_{bearing}*$.

We'll derive the complete mathematical framework for converting pixel coordinates to 3D bearing vectors using pinhole camera geometry. The mathematics begins with projective geometry relationships linking pixel positions to focal plane coordinates through affine transformations. Central derivations include camera calibration matrix formulations, lens distortion corrections, and coordinate system transformation matrices. We'll establish bearing vector normalization procedures and derive error propagation expressions relating pixel uncertainties to angular errors. The mathematical framework addresses numerical precision considerations and optimal focal length selection for attitude determination accuracy.

## **Section 6: Pattern Recognition Mathematics**

*Problem*: Identifying which observed stars correspond to catalog entries requires geometric pattern matching using inter-star angles. The BAST algorithm must reliably match triangular patterns despite measurement errors and false detections.

*Critical Parameters*: Inter-star angles $\theta_{ij}$, angle tolerance $\Delta\theta_{tol}$, match confidence $C_{match}$, triangle geometric invariants, false positive probability $P_{FP}$.

The mathematical foundation covers geometric invariant theory for triangle-based pattern matching. We'll derive inter-star angular distance calculations using spherical trigonometry and establish the mathematical basis for BAST triangle matching. Key derivations include rotation-invariant geometric features, probabilistic matching criteria, and false alarm rate calculations. The mathematics addresses combinatorial optimization for star assignment problems using Hungarian algorithm theory. We'll develop confidence metrics based on geometric consistency checks and establish mathematical criteria for rejecting ambiguous matches. Error analysis covers angular measurement uncertainties and their impact on matching reliability.

## **Section 7: Optimal Attitude Estimation**

*Problem*: Given matched star pairs, we must find the optimal rotation matrix relating observed and catalog reference frames. The QUEST algorithm solves Wahba's problem to minimize attitude errors while providing uncertainty bounds.

*Critical Parameters*: Davenport K-matrix eigenvalues $\lambda_i$, optimal quaternion $\hat{q}$, attitude uncertainty $\sigma_q$, residual errors $\epsilon_i$, confidence bounds.

We'll develop the complete mathematical framework for Wahba's problem and the QUEST algorithm solution. The derivation starts with least-squares formulations for attitude estimation, leading to the characteristic polynomial eigenvalue problem. Central mathematics includes Davenport K-matrix construction, quaternion algebra for rotation representations, and singular value decomposition methods for matrix solutions. We'll derive uncertainty propagation through the attitude estimation process, establishing covariance matrix expressions for attitude errors. The mathematical treatment covers Monte Carlo methods for robust error characterization and establishes convergence criteria for iterative solutions. Performance bounds derivations connect measurement noise to final attitude accuracy limits.
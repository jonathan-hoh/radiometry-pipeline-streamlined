# Star Tracker Radiometry Simulation: Mathematical Foundations

## Introduction

Imagine you're navigating a spacecraft hurtling through the vast emptiness of space. Without reliable landmarks, how do you know which way you're pointing? This is where star trackers come in—specialized cameras that use the fixed patterns of stars in the night sky as a cosmic compass. By capturing images of stars and comparing them to known catalogs, these devices calculate a spacecraft's precise orientation, or "attitude," in three-dimensional space. This attitude information is crucial for everything from pointing solar panels at the Sun to aligning scientific instruments with distant galaxies. However, designing and testing star trackers is challenging: real-world conditions like varying light levels, sensor noise, and spacecraft motion are hard to replicate on Earth without expensive hardware prototypes.  

That's why we've developed a Python-based simulation that acts as a "digital twin" for star tracker systems. A digital twin is essentially a virtual replica of a physical system, allowing us to model and predict performance under countless scenarios without ever leaving the computer. Our simulation focuses on the Basic Astronomical Star Tracker (BAST), a conceptual system designed for accurate attitude determination. By simulating the entire process—from generating realistic star scenes to estimating the final spacecraft orientation—we can predict how well a star tracker will perform in space, identify potential limitations, and optimize designs before building hardware. This not only saves time and resources but also helps engineers understand the underlying physics and algorithms at a deeper level.  

This white paper dives into the mathematical foundations that power our simulation. We'll break down the key components step by step, starting from the raw physics of light interacting with stars and sensors, all the way to advanced algorithms for matching stars and computing attitudes. The goal is to make these concepts accessible, even if you're not a PhD in math or physics. We'll explain every equation in plain words, define technical terms as they appear, and provide intuitive motivations for why each piece matters. Along the way, we'll connect the math to our Python codebase, highlighting how these ideas are implemented in practice, and discuss real-world insights and limitations.  

The paper is organized into seven core sections, each building on the last to form a complete simulation pipeline:  

1. **Scene Generation Mathematics**: How we create virtual star images based on spacecraft orientation.  
2. **Radiometric Chain Physics**: Modeling the flow of light from stars to sensor signals.  
3. **Statistical Image Formation**: Adding realistic noise to simulate actual camera outputs.  
4. **Detection and Centroiding Theory**: Finding and precisely locating stars in noisy images.  
5. **Bearing Vector Geometry**: Converting 2D image positions into 3D directions.  
6. **Pattern Recognition Mathematics**: Matching observed stars to known catalogs.  
7. **Optimal Attitude Estimation**: Calculating the final spacecraft orientation.  

To ensure clarity and consistency, each section follows a structured format:  

- **Problem Statement**: A clear description of the challenge we're addressing.  
- **Critical Parameters**: Key variables and terms, defined upfront.  
- **Step-by-Step Mathematical Derivation**: Detailed equations with word explanations before and after, so nothing feels like a black box.  
- **Connection to Codebase Implementation**: How this math translates to our Python code.  
- **Physical Insights and Limitations**: Real-world takeaways, including what works well and where assumptions might break down.  

By the end of this paper, you'll have a solid grasp of how star tracker simulations work, from the photons hitting a sensor to the quaternions describing a spacecraft's twist in space. We'll also include a glossary of all technical terms and variables for quick reference. If you're ready, let's start with Section 1, where we lay the groundwork by generating realistic star scenes— the essential first step before we can simulate light, noise, and everything that follows.  

*[Suggestion for Image]: To enhance motivation and intuition, consider adding a simple diagram here showing a spacecraft in orbit with a star tracker camera pointing at the stars, overlaid with arrows indicating the flow from starlight to attitude output. This would visually tie together the simulation pipeline and make the abstract concept more tangible for readers.*

## Section 1: Scene Generation Mathematics

Imagine you're setting the stage for a virtual space mission: before you can simulate how a star tracker "sees" the stars, you need to create a realistic snapshot of the night sky as viewed from a spacecraft. This involves taking the known positions of stars in the universe and transforming them into where they'd appear on the camera's sensor, based on the spacecraft's orientation (or "attitude"). Why is this important? Without accurately generating these star positions, all the downstream steps—like modeling light intensity, adding noise, detecting stars, and estimating attitude—would be based on flawed inputs, leading to unreliable simulation results. This section lays the foundation by ensuring the simulated scene preserves the true geometric relationships between stars, which is crucial for the pattern-matching algorithms that come later in the pipeline (explored in Section 6).  

In essence, we're bridging the gap between celestial maps and pixel coordinates. We'll start with stars' positions in a fixed, Earth-centered reference frame and apply rotations to mimic the spacecraft's viewpoint, projecting them onto a virtual camera detector. This process uses concepts from coordinate geometry and linear algebra, but we'll break it down intuitively: think of it as rotating a globe (the celestial sphere) and then flattening a portion onto a 2D photo.  

### Problem Statement

To create realistic star tracker images, we need to convert the positions of stars from their standard celestial coordinates into precise locations on the camera's detector, while accounting for the spacecraft's arbitrary orientation. This transformation must preserve the angular distances between stars, as these are key for later identifying and matching them against star catalogs.  

### Critical Parameters

Before diving into the math, let's define the key variables we'll use:  

- **Rotation matrices** $ R(\phi, \theta, \psi) $: 3x3 matrices that describe how to rotate coordinates from one frame to another, based on Euler angles (roll $\phi$, pitch $\theta$, yaw $\psi$).  
- **Quaternion components** $ [q_0, q_1, q_2, q_3] $: A compact, four-element representation of 3D rotations that avoids issues like gimbal lock in Euler angles.  
- **Focal plane coordinates** $ (x_f, y_f) $: Positions on an imaginary 2D plane inside the camera, in physical units like microns.  
- **Detector pixel positions** $ (u, v) $: The final grid locations on the sensor, measured in pixels.  
- **Angular preservation error** $ \Delta \theta $: A measure of how much angular distances might deviate due to approximations (ideally close to zero for accuracy).  

These parameters ensure the transformations are both mathematically sound and practically implementable.  

### Step-by-Step Mathematical Derivation

We'll derive the full transformation chain from celestial coordinates to pixel positions, explaining each step in words to build intuition. The process involves converting spherical positions to 3D vectors, applying rotations for attitude, projecting onto a plane, and mapping to pixels—all while keeping angles between stars unchanged.  

#### 1. Spherical to Cartesian Inertial Vectors

Stars are typically listed in catalogs using spherical coordinates: right ascension $\alpha$ (like longitude) and declination $\delta$ (like latitude) on the celestial sphere. To work with rotations, we first convert these to 3D Cartesian unit vectors in an inertial (fixed, Earth-centered) frame. This is like pointing from the center of a globe to a point on its surface.  

The conversion formula is:  

$$
\mathbf{v}_I = \begin{pmatrix}\cos \delta \cos \alpha \\\cos \delta \sin \alpha \\\sin \delta\end{pmatrix}
$$



This results in a unit vector (length 1), ensuring all stars are treated as if they're at the same infinite distance. The camera's boresight (its pointingdirection) is converted similarly to a vector $\mathbf{v}_B$. This step normalizes everything to directions rather than distances, which is perfect for attitude determination since stars are effectively points at infinity.  

**Connection to Codebase Implementation**: This is handled in the function `radec_to_inertial_vector` within `attitude_transform.py`, directly translating catalog data into usable vectors.  

#### 2. Attitude Representation and Rotation Matrices

Next, we account for the spacecraft's orientation using either Euler angles or quaternions to build a rotation matrix $R$. This matrix transforms star vectors from the inertial frame to the camera's frame, simulating how the view changes as the spacecraft rotates. Intuitively, it's like tilting your head while looking at the sky—the stars' relative positions stay the same, but their locations in your field of view shift.  

For Euler angles (in ZYX convention, meaning yaw around Z, then pitch around Y, then roll around X), we construct individual rotation matrices and multiply them:  

- Yaw (Z-axis): $ R_z(\psi) = \begin{pmatrix} \cos \psi & -\sin \psi & 0 \\ \sin \psi & \cos \psi & 0 \\ 0 & 0 & 1 \end{pmatrix} $  
- Pitch (Y-axis): $ R_y(\theta) = \begin{pmatrix} \cos \theta & 0 & \sin \theta \\ 0 & 1 & 0 \\ -\sin \theta & 0 & \cos \theta \end{pmatrix} $  
- Roll (X-axis): $ R_x(\phi) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos \phi & -\sin \phi \\ 0 & \sin \phi & \cos \phi \end{pmatrix} $  

The full rotation matrix is $ R = R_z(\psi) R_y(\theta) R_x(\phi) $.  

Alternatively, for quaternions (which are more efficient and avoid singularities), the rotation matrix is derived as:  

$$
R = \begin{pmatrix}q_0^2 + q_1^2 - q_2^2 - q_3^2 & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\2(q_1 q_2 + q_0 q_3) & q_0^2 - q_1^2 + q_2^2 - q_3^2 & 2(q_2 q_3 - q_0 q_1) \\2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & q_0^2 - q_1^2 - q_2^2 + q_3^2\end{pmatrix}
$$

We then apply the transformation: $\mathbf{v}_C = R \mathbf{v}_I$, where $\mathbf{v}_C$ is the star vector in the camera frame. Stars with a negative z-component ($ v_{C,z} \leq 0 $) are behind the camera and filtered out.  

A key property here is angular preservation: since rotation matrices are orthogonal ($ R^T R = I $, where $I$ is the identity matrix), the dot product (and thus angles) between vectors remains the same: $\mathbf{v}_{I1}^T \mathbf{v}_{I2} = \mathbf{v}_{C1}^T \mathbf{v}_{C2}$, so $\theta = \cos^{-1}(\mathbf{v}_1^T \mathbf{v}_2)$ is invariant. This ensures patterns look consistent regardless of rotation.  

**Connection to Codebase Implementation**: Functions like `euler_to_rotation_matrix` and `quaternion_to_rotation_matrix` in `attitude_transform.py` build these matrices, applied in `transform_to_camera_frame`.  

#### 3. Pinhole Camera Projection to Focal Plane

Now, we project the 3D camera-frame vectors onto a 2D focal plane using the pinhole camera model, which assumes light rays pass through a tiny aperture (like a pinhole) to form an image. This is a perspective projection, similar to how your eye sees depth compressed onto a flat retina.  

The coordinates are:  

$$
x_f = f \frac{v_{C,x}}{v_{C,z}}, \quad y_f = f \frac{v_{C,y}}{v_{C,z}}
$$

Here, $f$ is the focal length (distance from lens to sensor, in microns), acting as a scaling factor for how "zoomed in" the view is.  

**Connection to Codebase Implementation**: This is implemented in `project_to_focal_plane` within `attitude_transform.py`.  

#### 4. Focal Plane to Pixel Coordinates

Finally, convert physical focal plane positions (in microns) to pixel grid positions on the detector:  

$  u = u_0 + \frac{x_f}{p}, \quad v = v_0 + \frac{y_f}{p} $  

Where $(u_0, v_0)$ is the principal point (usually the detector's center), and $p$ is the pixel pitch (e.g., 5.5 µm for a CMV4000 sensor). We filter out positions outside the detector bounds (e.g., 0 to 2048 pixels) with a small margin to avoid edge effects.  

**Connection to Codebase Implementation**: Handled by `focal_plane_to_pixels` and `filter_detector_bounds` in `attitude_transform.py`.  

#### 5. Integrated Gnomonic Projection (Legacy Mode)

For simpler cases without full attitude rotations (e.g., boresight-centered views), we use a gnomonic projection, which maps the celestial sphere onto a tangent plane. This involves defining basis vectors (east $\mathbf{e}$ and north $\mathbf{n}'$) relative to the boresight and projecting stars onto this plane before converting to pixels. It's like unfolding a portion of the sphere onto a flat map, useful for quick approximations but less flexible than the full rotation method.  

**Connection to Codebase Implementation**: Found in `_sky_to_detector` within `scene_generator.py`.  

### Physical Insights and Limitations

This transformation chain allows us to generate multi-star scenes for any spacecraft attitude, preserving the geometry needed for accurate pattern matching in later sections. Physically, it highlights how sub-pixel precision is tied to hardware: angular error per pixel is roughly $\Delta \theta \approx \frac{p}{f}$ radians, so a longer focal length improves resolution but narrows the field of view (potentially seeing fewer stars).  

However, limitations include assuming an ideal pinhole model (no lens distortions, which could skew positions in real cameras) and validity only for small fields-of-view (under 20° to avoid projection inaccuracies). We validate by checking that pairwise angles remain consistent after transformations. Overall, this step ensures the simulation starts with a faithful representation of the sky, setting up the radiometric modeling in Section 2, where we'll add light and intensity to these positions.  

*[Suggestion for Image]: A flowchart diagram would be helpful here, illustrating the transformation chain from celestial coordinates to pixel positions, with icons for each step (e.g., a globe for spherical to Cartesian, arrows for rotations, and a camera icon for projection). This visual would make the sequence more intuitive and easier to follow for readers.*

## Section 2: Radiometric Chain Physics

Once we've generated a virtual star scene with accurate positions on the detector (as covered in Section 1), the next challenge is simulating how much light from those stars actually reaches and registers on the camera sensor. This is the realm of radiometry—the study of measuring electromagnetic radiation, like visible light from stars. Why does this matter? In a real star tracker, the brightness of stars determines the strength of the signal we can detect amidst noise; too faint, and stars might get lost in the background, while too bright could saturate the sensor. By modeling this "radiometric chain," we predict realistic signal levels, which directly feed into the noisy image formation in Section 3. Think of it as calculating the "fuel" (photons) that powers the rest of the simulation: get this wrong, and your digital twin won't match real-world performance.  

We'll trace the journey of starlight from its source to electron signals in the detector, using physics principles like flux and quantum efficiency. This involves converting abstract concepts like stellar magnitude into concrete photon counts, all while incorporating camera properties. The math draws from optics and astrophysics, but we'll explain it intuitively: imagine starlight as a stream of particles (photons) that get filtered and collected, much like rainwater funneled into a bucket.  

### Problem Statement

To create a faithful simulation, we must link a star's apparent brightness (its magnitude) to the actual number of photons hitting the detector, factoring in the camera's optics and sensor characteristics. This establishes the baseline signal strength, which sets performance limits for detection and helps predict when noise might overwhelm the signal.  

### Critical Parameters

Let's define the essential variables upfront, so the derivations feel grounded:  

- **Photon flux** $ \Phi $: The rate at which photons from a star arrive at Earth, in photons per second per square meter.  
- **Quantum efficiency** $ QE $: The fraction of incoming photons that generate electrons in the sensor (e.g., 0.8 means 80% efficiency).  
- **Optical transmission** $ \tau $: The fraction of light that passes through the lens without being absorbed or scattered (e.g., 0.9 for high-quality optics).  
- **Detector full well capacity** $ N_{FW} $: The maximum number of electrons a pixel can hold before saturating (e.g., 20,000 electrons).  
- **Dark current** $ I_d $: Unwanted electron generation in the sensor due to heat, in electrons per second per pixel.  
- **Signal-to-noise ratio** $ SNR $: A measure of signal quality, calculated as signal divided by noise; higher values mean clearer detections.  

These parameters bridge astronomy (star brightness) and engineering (camera specs), ensuring the model is realistic.  

### Step-by-Step Mathematical Derivation

We'll build the radiometric chain from stellar magnitude to electron count, explaining each step with words to reveal the physics. This is like following a photon from the star to the sensor, accounting for losses and conversions along the way.  

#### 1. Stellar Flux from Apparent Magnitude

Stars' brightness is measured on the apparent magnitude scale, a logarithmic system where lower numbers mean brighter objects (e.g., the Sun is -26.74, while faint stars might be +6 or higher). This scale is based on human eye perception, but we need to convert it to physical units like energy flux.  

The Pogson relation says that a magnitude difference $ \Delta m $ corresponds to a flux ratio of $ 10^{-0.4 \Delta m} $, which simplifies to approximately $ 2.512^{-\Delta m} $ (each magnitude step dims by a factor of about 2.512). We reference the Sun: its apparent magnitude $ m_\odot = -26.74 $ and solar constant $ S = 1366 $ W/m² (energy per second per square meter at Earth).  

To work with photons, calculate the average photon energy for visible light using a central wavelength $ \lambda_c $ (midpoint of the camera's bandpass, say 550 nm for green light):   

$E_{ph} = \frac{h c}{\lambda_c}$

Here, $ h $ is Planck's constant (6.626 × 10^{-34} J s), and $ c $ is the speed of light (3 × 10^8 m/s). This gives the energy per photon in joules.  

The Sun's photon flux is then $ \Phi_\odot = S / E_{ph} $ (photons/s/m²). For a zero-magnitude star (a standard reference), scale up from the Sun:  

$\Phi_0 = \Phi_\odot \times 2.512^{m_\odot}$  

Finally, for a star of magnitude $ m $:  

$$\Phi = \Phi_0 \times 2.512^{-m}  $

This $ \Phi $ is the photon arrival rate at Earth. Note that the code approximates 2.512 as 2.5 for simplicity, introducing a small ~1.7% error, but it's close enough for most simulations.  

**Connection to Codebase Implementation**: In `star.calculate_flux` from the codebase, it computes a constant based on solar values (with a 1e-6 unit adjustment) and scales by $ 2.5^{-m} $, then divides by photon energy to get flux.  

#### 2. Collected Photons at Aperture

The camera's lens acts like a collector: its aperture area $ A = \pi (D/2)^2 $ (where $ D $ is the diameter) determines how many photons are gathered. The incident photon rate is $ \Phi \times A $.  

Not all light makes it through—some is lost in the optics—so multiply by transmission $ \tau $:  

$\Phi_{opt} = \Phi \times A \times \tau$

Over an integration time $ t $ (exposure duration in seconds), the total photons collected are:  

$N_{ph} = \Phi_{opt} \times t$

This assumes a point source like a star, where all light focuses onto a small area on the sensor.  

**Connection to Codebase Implementation**: The `calculate_optical_signal` function computes the signal as star flux times aperture area, times transmission, times integration time (adjusted from ms to s).  

#### 3. Detector Response and Electron Generation

The sensor converts photons to electrons via the photoelectric effect, with efficiency $ QE $:  

$N_e = N_{ph} \times QE$

This $ N_e $ is the raw signal. However, the detector has limits: if $ N_e > N_{FW} $, the pixel saturates and loses information. Additionally, dark current adds spurious electrons: $ I_d \times t $, even without light.  

We'll address noise in detail next (Section 3), but for now, note that these effects set practical bounds on detectable magnitudes.  

**Connection to Codebase Implementation**: QE is a parameter in the focal plane array (FPA) class, applied to photon counts, though noise is handled separately in image simulations.  

#### 4. Noise Contributions

Real signals are noisy. Key sources include:  

- **Shot noise**: From the random arrival of photons, following a Poisson distribution with variance $ N_{ph} $.  
- **Dark noise**: Variance $ I_d t $ from dark current.  
- **Read noise**: Gaussian noise during readout, with variance $ \sigma_r^2 $.  

The total noise variance is approximately $ \sigma^2 = N_e + I_d t + \sigma_r^2 $ (after applying QE). The SNR is then:  

$SNR = \frac{N_e}{\sigma}$

For reliable detection, SNR typically needs to exceed 5–10. This helps set thresholds for what's visible in the simulated image.  

**Connection to Codebase Implementation**: While not in the core signal calculation, these are used in Poisson simulations elsewhere, like adding noise layers.  

#### 5. Spectral Integration

For precision, we could integrate flux over the full wavelength spectrum:  

$\Phi = \int_{\lambda_{min}}^{\lambda_{max}} F(\lambda) \frac{\lambda}{h c} \, d\lambda$

But the code simplifies to a central wavelength approximation, which works well for broadband visible light but might need refinement for narrow filters.  

**Connection to Codebase Implementation**: The codebase sticks to the central wavelength method for efficiency.  

### Physical Insights and Limitations

This radiometric model reveals how star brightness translates to usable signals: brighter stars (lower magnitudes) yield more photons and higher SNR, enabling better precision, while faint ones risk being drowned out by noise. Physically, it underscores trade-offs like aperture size (larger collects more light but increases weight) and exposure time (longer improves signal but risks motion blur).  

Limitations include assumptions of blackbody-like star spectra and uniform QE (real stars vary in color, and sensors in sensitivity), plus ignoring atmospheric effects (fine for space but not ground tests). Bright stars can saturate ($ N_e > N_{FW} $), capping dynamic range, and approximations like 2.5 vs. 2.512 introduce minor errors. Overall, this chain predicts detectability, paving the way for Section 3, where we add statistical noise to form realistic images—turning these photon counts into pixelated, imperfect snapshots.  

*[Suggestion for Image]: A flowchart or infographic tracing the radiometric chain would be valuable here, starting with a star icon emitting light, arrows showing flux calculation, aperture collection, optical losses, and ending at electron generation on a sensor grid. Include labels for key equations and parameters to visually reinforce the step-by-step process and make the abstract photon journey more concrete.]*

## Section 3: Statistical Image Formation

With the positions and brightness of stars established in the virtual scene (from Section 1) and their photon signals calculated (from Section 2), we now need to simulate what a real camera image would look like—complete with the imperfections of noise and randomness. This step is crucial because actual star tracker images are fuzzy due to the point spread function (PSF, which describes how light from a point source spreads out on the sensor) and riddled with statistical fluctuations from photon arrivals and electronic noise. Why bother modeling this? Without realistic noise, our simulation couldn't predict how well detection algorithms (coming up in Section 4) will perform in spotting stars amid the chaos, or how accurately we can pinpoint their centers. Think of it as sprinkling unpredictability onto an otherwise ideal picture: photons don't arrive in a steady stream but in random bursts, like raindrops on a window, and the sensor adds its own electronic "hiss" that can obscure faint signals.  

In this section, we'll model image formation using probability and statistics, focusing on Poisson processes for photon counts and Gaussian approximations for other noises. This creates a probabilistic "noisy" image from the deterministic signals, allowing us to test the limits of star detection under real-world conditions. The math draws from stochastic processes, but we'll unpack it intuitively through thousands of photon "dice rolls" to build up the image pixel by pixel.  

### Problem Statement

Real camera sensors introduce shot noise (from random photon arrivals) and readout noise (from electronics), which limit how precisely we can locate stars. To make our digital twin accurate, we must incorporate these effects using Poisson statistics for photons and additional models for detector imperfections, enabling predictions of detection reliability and validation of sub-pixel positioning techniques.  

### Critical Parameters

Before the derivations, let's define the key variables to ground the concepts:  

- **Poisson parameter** $ \lambda $: The expected number of photons (or electrons) in a pixel, which also equals the variance for Poisson-distributed counts.  
- **Read noise variance** $ \sigma_r^2 $: The spread of random electronic noise added during image readout, typically in electrons squared (e.g., 10 e⁻ RMS).  
- **Pixel response non-uniformity** $ \epsilon_{PRNU} $: Small variations in how pixels respond to the same light level, expressed as a percentage (e.g., 1% means pixels vary by ±0.01 in sensitivity).  
- **PSF sampling error** $ \Delta x_{sample} $: The positional inaccuracy introduced when shifting the PSF to sub-pixel locations, in pixels (e.g., about 1/12 pixel for basic interpolation).  
- **Total noise equivalent electrons** $ N_{noise} $: A summary metric of all noise sources, in electrons, helping estimate the faintest detectable signal.  

These parameters capture the stochastic nature of imaging, linking hardware specs to simulation outcomes.  

### Step-by-Step Mathematical Derivation

We'll derive the process of building a noisy image from the PSF and photon counts, explaining each part in words. This involves treating the PSF as a probability map, simulating random photon arrivals, adding detector effects, and handling sub-pixel precision—all to mimic how real images form.  

#### 1. PSF as Probability Distribution

The point spread function (PSF) describes how light from a star blurs across pixels due to optics and diffraction—it's essentially a 2D map of light intensity. To model randomness, we treat the PSF as a probability distribution: for each photon, it dictates the chance of landing in a specific pixel (i,j). First, normalize it so the total adds up to 1:  

$  \sum_{i,j} P(i,j) = 1$    

Here, $ P(i,j) $ is the normalized intensity at pixel (i,j), often derived from optical simulations like Zemax software. This normalization turns the PSF into a proper probability density, ensuring that if you "throw" many photons at it, their distribution matches the expected blur pattern.  

**Connection to Codebase Implementation**: The `normalize_psf` function in `psf_photon_simulation.py` divides the PSF matrix by its sum (if non-zero), creating this probability map directly from input data.  

#### 2. Poisson Photon Arrival

Photons don't arrive uniformly; their counts follow a Poisson distribution, reflecting the quantum randomness of light. For a total number of photons $ N_{ph} $ from a star (calculated in Section 2), the expected count in pixel (i,j) is:  

$\lambda_{i,j} = N_{ph} P(i,j)$

The actual count $ k_{i,j} $ is then drawn from a Poisson random variable:  

$Pr(k_{i,j} = k) = \frac{\lambda_{i,j}^k e^{-\lambda_{i,j}}}{k!}$

This means the variance equals the mean ($ \lambda_{i,j} $), creating "shot noise" that's more pronounced for fewer photons. To simulate an image, you can either sample per pixel using this distribution or, for Monte Carlo accuracy, simulate individual photon locations by repeatedly sampling from the PSF probabilities.  

**Connection to Codebase Implementation**: The `simulate_psf_with_poisson_noise` function computes expected_counts as psf_normalized * num_photons, then uses np.random.poisson(expected_counts) to generate noisy counts for each pixel across multiple trials. An alternative, `simulate_photon_distribution`, flattens the PSF into probabilities and samples photon locations directly.  

#### 3. Detector Response Modeling

Once we have photon counts, the sensor converts them to digital values, but not perfectly. We add effects like gain (amplification), offset (baseline signal), read noise (random fluctuations during readout), and dark current (thermal electrons, also Poisson-distributed). The digital number (DN) for pixel (i,j) is:  

$DN_{i,j} = g (k_{i,j} + d_{i,j}) + o + r_{i,j}$

Here, $ g $ is the gain (electrons per photon, though QE from Section 2 is often folded in), $ d_{i,j} \sim Poisson(I_d t) $ is dark current electrons over exposure time t, $ o $ is a fixed offset, and $ r_{i,j} \sim \mathcal{N}(0, \sigma_r^2) $ is Gaussian read noise. Finally, clamp DN to the sensor's bit depth (e.g., 0 to 4095 for 12 bits) to simulate saturation.  

This equation builds a complete pixel value, incorporating both quantum and electronic effects for realism.  

**Connection to Codebase Implementation**: While the core Poisson simulation outputs the noisy image, comments in the code mention adding these layers; they're not always explicit in the base functions but are implied for full detector modeling, with poisson_image as the primary output.  

#### 4. Sub-Pixel PSF Placement and Interpolation

Stars don't always land exactly on pixel centers, so we shift the PSF by fractional amounts to place it at precise (x,y) positions from Section 1. This requires interpolation, like bilinear (linear blending between neighbors) or spline methods, to create a shifted PSF grid. The sampling error for bilinear interpolation is approximately:  

$\Delta x_{sample} \approx \frac{p}{12}$

Where p is pixel size—this represents the average positional error from approximating the continuous PSF on a discrete grid.  

**Connection to Codebase Implementation**: In `multi_star_radiometry.py`, the `_place_psf_at_position_subpixel` function uses scipy.ndimage.shift with order=1 (linear interpolation) to handle these fractional shifts accurately.  

#### 5. Total Noise and Error Propagation

Combining all sources, the variance for each pixel's electron count is:  

$\sigma_{i,j}^2 = g^2 (\lambda_{i,j} + I_d t) + \sigma_r^2$

This includes scaled shot and dark noise plus read noise. For the whole star signal, SNR is the total electrons divided by the square root of the total variance. Errors propagate: for example, PSF sampling affects centroiding (previewed here, detailed in Section 4 via the Cramér-Rao bound, which sets a theoretical minimum uncertainty for position estimates).  

**Connection to Codebase Implementation**: Simulations in the code compute mean_image and std_image over multiple Monte Carlo trials, effectively capturing this total noise and its propagation.  

### Physical Insights and Limitations

This statistical modeling shows how noise sets fundamental limits: for bright stars with many photons, shot noise (sqrt of signal) dominates, allowing precise measurements; for faint ones, fixed read noise takes over, potentially hiding them entirely. Sampling and interpolation highlight the need for high-resolution PSFs to avoid aliasing artifacts, like jagged edges in the simulated blur.  

Limitations include assuming pixels are independent (ignoring real effects like crosstalk, where light leaks between pixels, or PRNU variations), and the central wavelength approximation from Section 2 might not capture color-dependent noise. The model is validated by checking that averaged Monte Carlo images match the expected PSF shape. Overall, this step produces the "raw" noisy images essential for testing detection in Section 4, where we'll extract star positions from this simulated mess.  

*[Suggestion for Image]: An illustrative figure here could show side-by-side comparisons: an ideal PSF, a noisy Poisson-simulated version, and one with added detector noise. Include overlays of probability distributions (e.g., a Poisson curve) to visually explain randomness, helping readers grasp how clean signals become imperfect images.]*

## Section 4: Detection and Centroiding Theory

Now that we've simulated a noisy, realistic image of stars on the detector (from Section 3), the next step is to extract useful information from it: specifically, identifying where the stars are and pinpointing their exact positions with sub-pixel accuracy. This process, known as detection and centroiding, is like finding needles in a haystack—scanning the image for bright spots that stand out from the noise, then calculating their "center of mass" to get precise coordinates. Why is this vital? These positions are the raw data for converting 2D pixels into 3D directions (bearing vectors, covered in Section 5), and any errors here propagate through the entire pipeline, potentially throwing off star matching and attitude estimation. In low-SNR scenarios (e.g., faint stars or high noise), poor detection could miss stars altogether, while inaccurate centroiding might lead to mismatched patterns.  

We'll approach this using image processing techniques grounded in statistics and moments (a mathematical way to describe shapes). Intuitively, detection sets a "brightness cutoff" to flag potential stars, then centroiding weighs the pixel intensities to find the average position, much like balancing a unevenly loaded tray to find its center. The math involves adaptive thresholding and weighted sums, but we'll explain it step by step to show how it handles real-world messiness like varying backgrounds or overlapping blurs.  

### Problem Statement

From a noisy image, we need to robustly detect regions corresponding to stars and compute their sub-pixel centroids (positions) with 0.1–0.3 pixel accuracy. The algorithms must adapt to non-uniform noise, reject false positives (like hot pixels), and work for both isolated and clustered stars, all while minimizing biases from finite image windows or PSF shapes.  

### Critical Parameters

Let's define the key terms before proceeding, to make the math more approachable:  

- **Detection threshold** $ T_{detect} $: The minimum intensity value (in digital counts) a pixel must exceed to be considered part of a star; often set dynamically based on local noise.  
- **Centroid coordinates** $ (x_c, y_c) $: The calculated sub-pixel position of a star's center, in pixel units.  
- **Centroiding uncertainty** $ \sigma_{centroid} $: The estimated error in the centroid position, typically in fractions of a pixel (e.g., 0.2 pixels).  
- **Moment calculation weights** $ w_i $: Intensity values used to "weight" each pixel's contribution to the centroid, emphasizing brighter areas.  
- **Region selection criteria**: Rules like minimum/maximum area (in pixels) or total intensity to filter valid star blobs from noise or artifacts.  

These parameters ensure the process is tunable to different camera setups and noise levels.  

### Step-by-Step Mathematical Derivation

We'll derive the detection and centroiding process from basic image statistics to precise position estimates, with word explanations for each step. This builds from identifying candidate regions to refining their centers, incorporating noise considerations to quantify reliability.  

#### 1. Adaptive Thresholding for Detection

Images have varying backgrounds (e.g., due to dark current or stray light), so a fixed threshold might miss faint stars in noisy areas or flag too many false positives in quiet ones. To adapt, divide the image into blocks (e.g., B×B pixels, like 32×32) and compute local statistics: the mean intensity $ \mu_b $ and standard deviation $ \sigma_b $ for each block.  

The local threshold is then:  

$T_b = \mu_b + k \sigma_b$

Here, $ k $ is a multiplier (typically 3–5) chosen to balance sensitivity and false alarms—higher $ k $ is more conservative. Apply this to create a binary mask: pixels above $ T_b $ are marked as "1" (potential star), others as "0." This adaptive approach ensures detection works across the entire image, even if noise levels vary.  

**Connection to Codebase Implementation**: In `detect_stars_and_calculate_centroids` (likely in a file like identify.py), it uses cv2.resize for multi-scale processing, then computes local mean and std to set the threshold dynamically.  

#### 2. Connected Components Labeling

With the binary mask, group adjacent "1" pixels into "blobs" using 8-connectivity (pixels touching horizontally, vertically, or diagonally are connected). This identifies distinct regions that might be stars.  

Filter these blobs to reject noise: require a minimum area (e.g., 3 pixels to avoid single hot pixels) and maximum area (e.g., 50 pixels to exclude large artifacts or merged stars). For single-star images, select the brightest blob by summing its intensities; for multi-star scenes, keep all that pass the filters. This step refines candidates, reducing computational load for centroiding.  

**Connection to Codebase Implementation**: Handled by `cv2.connectedComponentsWithStats` in a function like `group_pixels` (in identify.py), which filters regions by size and other stats.  

#### 3. Moment-Based Centroiding

For each valid region, compute the centroid as the intensity-weighted average position—like finding the balance point of a shape where brighter pixels "pull" harder. For pixels at positions (x, y) with intensities I(x, y):  



$x_c = \frac{\sum x \, I(x,y)}{\sum I(x,y)}, \quad y_c = \frac{\sum y \, I(x,y)}{\sum I(x,y)}$



This is the first-order moment (center of mass). The sums are over the region's pixels, and weights $ w_i = I(x,y) $ emphasize the core of the PSF where signal is strongest.  

To estimate uncertainty due to noise, approximate the variance in $ x_c $:  



$\sigma_{x_c}^2 \approx \frac{\sum (x - x_c)^2 I(x,y)}{\left( \sum I(x,y) \right)^2 } \times \sum I(x,y)$



This accounts for Poisson noise (variance equals mean intensity). A theoretical lower limit is the Cramér-Rao bound:  

$\sigma_{centroid} \geq \frac{\sigma_{psf}}{\sqrt{N_{ph}}}$

Where $ \sigma_{psf} $ is the PSF width (e.g., in pixels) and $ N_{ph} $ is the total photons—more photons mean tighter bounds, like averaging more data points for better precision.  

**Connection to Codebase Implementation**: The `calculate_centroid` function in identify.py computes these weighted moments directly from the region's pixel data.  

#### 4. Bias and Window Effects

Centroids can be biased if the calculation window cuts off the PSF's tails (faint outer light). For a Gaussian-like PSF, the bias grows exponentially with distance from the center: roughly $ \exp(-r^2 / 2\sigma^2) / r $, where r is the window radius. To minimize this, use an optimal window size of about 3–5 times the PSF's full width at half maximum (FWHM)—large enough to capture most light but small enough to exclude noise.  

**Connection to Codebase Implementation**: Size filtering in the labeling step helps mitigate this; adaptive blocks from thresholding also aid in selecting appropriate regions.  

#### 5. Alternative: Peak Detection

For crowded fields or overlapping PSFs, moment-based methods might merge stars, so an alternative is peak detection: find local maxima (brightest pixels) above the threshold, then filter by minimum separation (e.g., 5 pixels) to avoid duplicates. This is simpler but less precise for sub-pixel work, often used as a starting point for refinement.  

**Connection to Codebase Implementation**: Implemented in `detect_stars_peak_method` in peak_detection.py, using a max_filter to identify peaks.  

### Physical Insights and Limitations

This detection and centroiding process highlights how SNR drives accuracy: high-SNR stars (bright, many photons) yield centroids with low uncertainty (e.g., 0.1 pixels), enabling arcsecond-level pointing; low-SNR cases degrade to 0.5 pixels or more, risking mismatches. Physically, it's tied to the PSF—sharper PSFs (from better optics) allow tighter centroids, but real sensors add biases like pixel non-uniformity.  

Limitations include assumptions of isolated stars (crowding requires deblending techniques not covered here) and sensitivity to background variations (adaptive methods help but aren't perfect). For very faint stars, false positives rise, and finite windows introduce biases (up to 0.05 pixels for typical PSFs). Validation comes from comparing simulated centroids to known positions. Overall, this step transforms noisy pixels into precise locations, setting up the geometric conversion to bearing vectors in Section 5—where we'll turn these 2D spots into 3D directions for matching and attitude calculation.  

*[Suggestion for Image]: A before-and-after diagram would be beneficial here, showing a noisy star image on the left, with overlaid detection masks and centroids (e.g., red crosses on blobs), and a zoomed-in view on the right illustrating the weighted moment calculation with arrows from pixels to the center. This would visually demonstrate how raw noise becomes precise positions, making the abstract process more intuitive.]*

## Section 5: Bearing Vector Geometry

With precise star positions extracted from the noisy image via detection and centroiding (as detailed in Section 4), we now transform these 2D pixel coordinates into 3D directions—known as bearing vectors—that point from the camera toward the stars in space. This step is essentially reversing the projection process from Section 1, but starting from the detector end: it turns flat image spots into unit vectors in the camera's reference frame, which can then be compared to known star directions in a celestial catalog. Why is this crucial? Bearing vectors form the bridge between the observed image and the star matching algorithms in Section 6; any inaccuracies here, like those from camera calibration errors, could lead to mismatched stars and faulty attitude estimates. Intuitively, think of it as "unprojecting" a photo back into the 3D world—like estimating directions to landmarks in a picture based on your camera's lens specs.  

We'll use geometry and camera models to derive these vectors, incorporating factors like focal length and distortions. The math relies on inverse perspective projection, but we'll explain it accessibly: it's like tracing rays backward from the sensor through the lens to infinity, normalizing them to unit length for consistency with celestial maps.  

### Problem Statement

To enable star pattern matching, we must convert 2D pixel centroids into accurate 3D unit bearing vectors in the camera frame, accounting for intrinsic camera parameters and potential distortions. This transformation must propagate uncertainties from centroiding to predict overall angular errors, ensuring the vectors are reliable for downstream attitude determination.  

### Critical Parameters

Before the derivations, let's clearly define the main variables involved:  

- **Focal length** $ f $: The distance from the camera's optical center to the sensor plane, in physical units like microns or millimeters (e.g., 40 mm for a typical star tracker lens).  
- **Principal point** $ (c_x, c_y) $: The pixel coordinates where the optical axis intersects the detector, often near the center (e.g., (1024, 1024) for a 2048×2048 sensor).  
- **Pixel pitch** $ p $: The physical size of each pixel, in microns (e.g., 5.5 µm), which scales pixel positions to real-world distances.  
- **Bearing vector components** $ [\hat{v}_x, \hat{v}_y, \hat{v}_z] $: The normalized 3D direction vector, with length 1, pointing toward the star.  
- **Angular error** $ \Delta \theta_{bearing} $: The uncertainty in the vector's direction, in radians or arcseconds, arising from centroid errors and calibration inaccuracies.  

These parameters tie hardware specifications to geometric accuracy, making the model adaptable to different camera designs.  

### Step-by-Step Mathematical Derivation

We'll derive the bearing vector from pixel coordinates, explaining each transformation in words. This process inverts the forward projection from Section 1, starting with physical coordinates and ending with normalized 3D vectors, while incorporating error analysis for realism.  

#### 1. Pixel to Focal Plane Coordinates

Pixel positions $ (u, v) $ from centroiding are in discrete grid units, but we need physical distances on the focal plane. Convert them using the pixel pitch and principal point, which accounts for any offset in the optical center.  

The formulas are:  

$x_f = (u - c_x) \times p, \quad y_f = (v - c_y) \times p$

Here, $ x_f $ and $ y_f $ are in microns (or the same units as $ f $). Note that the y-coordinate might be flipped (e.g., $ y_f = (c_y - v) \times p $) depending on the image coordinate convention—top-left vs. bottom-left origin—to match the camera's orientation.  

This step translates abstract pixels into measurable distances, like marking points on a physical grid.  

**Connection to Codebase Implementation**: In `star_tracker_pipeline.calculate_bearing_vectors`, this is handled by a `pixel_to_physical_conversion` function that multiplies by pixel_pitch after subtracting the principal point.  

#### 2. Pinhole Projection to 3D Vector

Using the pinhole model (assuming light rays converge at a point), we extend the focal plane coordinates into 3D by adding the focal length as the z-component. This creates a vector from the optical center to the point on the plane, which we then normalize to unit length for consistency (since stars are at infinite distance).  

The bearing vector is:  

$\mathbf{\hat{v}} = \frac{ [x_f, y_f, f] }{ \sqrt{x_f^2 + y_f^2 + f^2} }$

The components are $ \hat{v}_x = x_f / d $, $ \hat{v}_y = y_f / d $, $ \hat{v}_z = f / d $, where $ d = \sqrt{x_f^2 + y_f^2 + f^2} $ is the normalization denominator. This ensures $ \|\mathbf{\hat{v}}\| = 1 $, representing a pure direction.  

Intuitively, longer focal lengths make the vector more "forward-pointing" (larger $ \hat{v}_z $), corresponding to narrower fields of view.  

**Connection to Codebase Implementation**: The `calculate_bearing_vectors` function computes normalized vectors as [x_um / f_um, y_um / f_um, 1] and then normalizes the array, effectively scaling by the focal length in microns.  

#### 3. Distortion Correction (if Applicable)

Real lenses introduce distortions (e.g., barrel or pincushion effects) that warp positions, so we correct $ (x_f, y_f) $ before vector calculation. A simple radial distortion model uses a polynomial:  

$x_{corr} = x_f (1 + k_1 r^2 + k_2 r^4), \quad y_{corr} = y_f (1 + k_1 r^2 + k_2 r^4)$

Where $ r^2 = x_f^2 + y_f^2 $, and $ k_1, k_2 $ are calibration coefficients (e.g., from lab measurements). Higher-order terms can be added for precision. After correction, use $ (x_{corr}, y_{corr}) $ in the vector formula.  

This step compensates for optical imperfections, ensuring vectors align with the true sky geometry.  

**Connection to Codebase Implementation**: While not in the base pipeline, `identify.py` includes a placeholder `apply_distortion_correction` function, indicating it's optional but can be integrated for more accurate simulations.  

#### 4. Error Propagation

Uncertainties in centroids $ \delta u $ and $ \delta v $ (from Section 4) propagate to angular errors in the bearing vector. Approximating for small errors, the directional uncertainty is:  

$\Delta \theta \approx \frac{p}{f} \sqrt{ (\delta u)^2 + (\delta v)^2 }$

In radians (convert to arcseconds by multiplying by 206265). For example, with $ f = 40 $ mm (40,000 µm), $ p = 5.5 $ µm, and $ \delta u = 0.2 $ pixels, $ \Delta \theta \approx 1 $ arcsecond—critical for high-precision attitude.  

Calibration errors in $ f $ or $ (c_x, c_y) $ add systematic biases, which can be modeled via sensitivity analysis (e.g., partial derivatives of $ \mathbf{\hat{v}} $ with respect to parameters).  

**Connection to Codebase Implementation**: In Monte Carlo simulations (e.g., `monte_carlo.py`), centroid standard deviations are used to perturb positions, and the resulting bearing vector errors are propagated through to attitude uncertainty.  

#### 5. Calibration Matrix (General Form)

For more advanced cameras, use an intrinsic matrix $ K $ (3×3) that encapsulates $ f $, $ p $, and $ (c_x, c_y) $. The bearing vector can be derived from the inverse projection, but the code assumes a simplified ideal pinhole model without skew or non-square pixels.  

**Connection to Codebase Implementation**: The base implementation sticks to the ideal model for efficiency, but extensions could incorporate full $ K $-matrix inversion.  

### Physical Insights and Limitations

This geometric transformation reveals key trade-offs in star tracker design: a longer focal length $ f $ improves angular resolution (smaller $ \Delta \theta $ per pixel error), but it narrows the field of view, potentially capturing fewer stars for matching. Physically, it emphasizes calibration's importance—temperature changes can expand/contract the lens, shifting $ f $ by 0.1% and introducing errors up to arcminutes if unaccounted for.  

Limitations include the ideal pinhole assumption (real systems often need distortion models, especially for wide fields >10°), and ignoring effects like thermal variations or mechanical misalignment. The model assumes square pixels and no skew, which holds for most CMOS sensors but not all. Validation involves comparing simulated vectors to known star positions, with errors typically under 1 arcsecond for well-calibrated systems. Overall, these bearing vectors provide the 3D foundation for pattern recognition in Section 6, where we'll match them to catalog stars to identify patterns and unlock the spacecraft's orientation.  

*[Suggestion for Image]: A ray-tracing diagram would enhance understanding here, depicting light rays from stars passing through the pinhole, hitting the focal plane at pixel positions, and then reversed arrows showing the bearing vectors extending outward. Label key parameters like $ f $, $ p $, and $ (x_f, y_f) $, with a zoomed inset for distortion correction to illustrate warping effects. This visual would make the inverse projection intuitive and highlight error sources.]*

## Section 6: Pattern Recognition Mathematics

Armed with 3D bearing vectors from the observed stars (derived in Section 5), we now face the task of identifying which stars they correspond to in a known celestial catalog. This is pattern recognition: matching the geometric arrangement of observed directions to precomputed star patterns, using features like inter-star angles that remain invariant under rotation. Why is this a linchpin in the pipeline? Without correct identifications, the attitude estimation in Section 7 would use mismatched data, leading to wildly inaccurate orientations—like confusing the Big Dipper for Orion. In challenging conditions (e.g., noisy vectors or stray light creating false stars), the algorithm must be robust, rejecting ambiguities while confirming matches with high confidence. Think of it as solving a cosmic puzzle: we look for unique "signatures" like triangular shapes formed by star triplets, which are as distinctive as fingerprints for identification.  

We'll derive the math behind this matching process, drawing from spherical geometry and combinatorial search. Intuitively, it's like comparing distances on a map: we calculate angles between vectors, form triplets, and search for near-identical ones in a database, scoring them to pick the best fit. The approach minimizes false positives through tolerances and verification, ensuring reliability for spacecraft navigation.  

### Problem Statement

To determine spacecraft attitude, we must associate observed bearing vectors with catalog stars using rotation-invariant metrics like angular separations. The algorithm needs to handle measurement errors, potential false detections, and computational efficiency, while providing match confidence to flag ambiguities.  

### Critical Parameters

Before delving into the derivations, let's outline the key variables that shape the matching process:  

- **Inter-star angles** $ \theta_{ij} $: The angular separation between two stars i and j, in radians (typically 0.1° to 10° for field-of-view limits).  
- **Angle tolerance** $ \Delta \theta_{tol} $: The maximum allowable difference for a match, in radians (e.g., 0.01° to account for bearing errors from Section 5).  
- **Match confidence** $ C_{match} $: A score (0–1) indicating how well the patterns align, based on angular residuals.  
- **Triangle geometric invariants**: Sorted angles (a ≤ b ≤ c) of star triplets, used as unique identifiers.  
- **False positive probability** $ P_{FP} $: The likelihood of erroneous matches, minimized by tight tolerances and verification steps.  

These parameters balance sensitivity (catching true matches) with specificity (avoiding fakes), tunable via simulations.  

### Step-by-Step Mathematical Derivation

We'll build the pattern matching from basic angular calculations to full triplet searches and scoring, explaining each step intuitively. This process leverages the angular preservation from Section 1, ensuring patterns are viewpoint-independent.  

#### 1. Inter-Star Angular Distance

The foundation is computing the angle between two unit bearing vectors $ \mathbf{v}_i $ and $ \mathbf{v}_j $ (from observations or catalog). Since they're on the unit sphere, the great-circle distance is:  

$\theta_{ij} = \cos^{-1}(\mathbf{v}_i \cdot \mathbf{v}_j)$

The dot product $ \mathbf{v}_i \cdot \mathbf{v}_j $ ranges from -1 to 1; we clip it to avoid numerical issues. For small angles (common in narrow fields), approximate $ \theta \approx \sqrt{2(1 - \cos \theta)} $, but the full arccos is used for precision. This preserves the "shape" of constellations regardless of rotation.  

**Connection to Codebase Implementation**: The `calculate_vector_angle` function in `match.py` computes np.arccos(np.clip(dot, -1, 1)), returning radians for pairwise comparisons.  

#### 2. Triangle Geometric Invariants

To make matching robust, we form triplets of stars (three vectors) and compute their three inter-star angles $ a, b, c $, sorted so a ≤ b ≤ c. This sorted tuple is rotation-invariant and acts as a hash for quick lookups. In a catalog, precompute these for every possible triplet within a field-of-view radius (e.g., 10°), associating each with the involved star IDs.  

For efficiency, limit to nearby stars: for a central star, find neighbors within the max separation, compute angles, and store in a database. The number of triplets scales as O(n^3) but is pruned by angular constraints.  

**Connection to Codebase Implementation**: The `triplet_worker` in `catalog.py` uses astropy's SkyCoord.separation.rad to compute angles for all pairs, then forms and stores triplets for rapid querying.  

#### 3. Matching Triplets

For an observed triplet with angles $ \theta_1, \theta_2, \theta_3 $ (sorted), search the catalog for entries where each angle differs by less than $ \Delta \theta_{tol} $:  

$|\theta_k - \theta_{cat,k}| < \Delta \theta_{tol} \quad \forall k = 1,2,3$  

If multiple candidates, score them with confidence:  



$C_{match} = 1 - \frac{\sum |\delta \theta_k|}{3 \Delta \theta_{tol}}$



Higher C means a better fit. Select the highest-confidence match, or verify by checking if the implied rotation aligns additional stars.  

**Connection to Codebase Implementation**: `find_triplet_match` in `match.py` calculates differences and checks if all are below tolerance, then computes a score based on averages.  

#### 4. Probabilistic Matching and False Alarms

To quantify reliability, estimate the probability of a random match. The "volume" of tolerance in angle space is roughly $ (\Delta \theta_{tol})^3 $, and with N catalog triplets, $ P_{FP} \approx N \times (\Delta \theta_{tol} / \pi)^3 $ (normalized to the sphere's range). Tighten $ \Delta \theta_{tol} $ based on bearing errors from Section 5 (e.g., set to 3σ of angular uncertainty) to keep $ P_{FP} < 10^{-6} $.  

For verification, compute residuals after a tentative match and reject if they exceed a threshold.  

**Connection to Codebase Implementation**: While not directly computed, `match.py` uses a min_confidence threshold to filter, implicitly controlling false positives.  

#### 5. Combinatorial Optimization for Multi-Star Matching

With multiple observed stars, generate all possible triplets and match them combinatorially. To assign unique catalog IDs without duplicates, use a greedy approach: match the most confident triplet first, mark those stars as used, and repeat. For optimal assignment, employ the Hungarian algorithm on a cost matrix where costs are angular differences, minimizing the total mismatch.  

The cost for assigning observed star i to catalog star j could be $ c_{ij} = |\theta_{obs} - \theta_{cat}| $ for pairwise, but extended to full patterns.  

**Connection to Codebase Implementation**: The main `match` function iterates combinations greedily, tracking used indices for bijectivity. For advanced cases, `bijective_matching.py` uses scipy.optimize.linear_sum_assignment (Hungarian) on a cost matrix.  

#### 6. Confidence Metrics and Rejection

After matching, compute overall confidence from averaged residuals or by fitting a rotation (preview of Section 7) and checking alignment. Propagate errors: bearing uncertainty $ \Delta \theta_{bearing} $ from Section 5 adds to $ \Delta \theta_{tol} $, increasing $ P_{FP} $ in noisy conditions.  

**Connection to Codebase Implementation**: `score_match` in `match.py` averages normalized differences; matches are rejected if below a threshold.  

### Physical Insights and Limitations

This pattern recognition leverages the spherical geometry of the sky: angles are invariant, making matches independent of the spacecraft's unknown attitude—a key insight for "lost-in-space" scenarios where no prior orientation is assumed. Physically, it highlights catalog density trade-offs: denser catalogs (more faint stars) provide more unique patterns but increase computational load and ambiguity risk.  

Limitations include combinatorial explosion for many stars (e.g., >20 requires efficient pruning), assumptions of no identical patterns (rare but possible in sparse fields), and sensitivity to false stars (mitigated by confidence scoring). For wide fields, spherical distortions might need haversine formulas over dot products. Validation involves simulating random views and measuring match success rates. Overall, successful matching unlocks the final step in Section 7, where identified stars yield the precise attitude quaternion—turning pattern IDs into navigational truth.  

*[Suggestion for Image]: A visual representation of triplet matching would be ideal here, showing observed bearing vectors on the left forming a triangle with labeled angles, a catalog database in the middle with similar triangles, and a matched pair on the right with confidence scores and residual arrows. Include error bubbles around angles to illustrate tolerance, helping readers visualize the search and scoring process in an intuitive way.]*

## Section 7: Optimal Attitude Estimation

With matched star identifications from pattern recognition (as established in Section 6), we can finally determine the spacecraft's precise orientation—its attitude—by finding the best rotation that aligns the observed bearing vectors with their catalog counterparts. This is the culmination of the pipeline: solving Wahba's problem, which seeks the optimal rotation matrix minimizing the misalignment between two sets of vectors. Why is this essential? Accurate attitude is the end goal of any star tracker, enabling navigation and control; errors here could lead to pointing inaccuracies of arcminutes or more, jeopardizing missions. In noisy or uncertain matches, the algorithm must provide not just the estimate but also confidence bounds, like a statistical "error bar" on the orientation. Intuitively, it's like fitting puzzle pieces: we compute a quaternion (a compact way to represent rotations) that twists the observed stars to overlap the known ones, using eigenvalues to find the sweet spot.  

We'll derive the QUEST (QUaternion ESTimator) algorithm, a efficient solution to Wahba's problem, incorporating weights and Monte Carlo methods for uncertainty. The math involves linear algebra and optimization, but we'll break it down: start with building a matrix from vector pairs, then extract the eigenvector for the quaternion, and simulate perturbations to gauge reliability.  

### Problem Statement

Given pairs of observed bearing vectors and matched catalog vectors, compute the optimal rotation (as a quaternion or matrix) that aligns them, while quantifying uncertainty from upstream errors like centroiding noise. The solution must be computationally efficient for real-time use and robust to outliers or mismatched pairs.  

### Critical Parameters

Let's define the pivotal variables upfront to anchor the derivations:  

- **Davenport K-matrix eigenvalues** $ \lambda_i $: The roots of the characteristic equation, where the maximum $ \lambda_{max} $ indicates the optimal solution's quality.  
- **Optimal quaternion** $ \hat{q} = [q_0, q_1, q_2, q_3] $: A unit vector representing the rotation, with $ q_0 $ as the scalar part.  
- **Attitude uncertainty** $ \sigma_q $: The standard deviation of quaternion components, in dimensionless units (convertible to angular errors).  
- **Residual errors** $ \epsilon_i $: The angular misalignments after rotation, in radians, used for validation.  
- **Confidence bounds**: Statistical intervals (e.g., 95%) on the attitude, derived from Monte Carlo simulations.  

These parameters ensure the estimation is not just a point value but a probabilistic assessment, critical for mission risk analysis.  

### Step-by-Step Mathematical Derivation

We'll derive the QUEST method from Wahba's loss function to quaternion extraction, with explanations to highlight the optimization. This builds on the matched vectors from Section 6, treating them as noisy measurements to solve for the attitude rotation.  

#### 1. Wahba's Loss Function and Attitude Profile Matrix

Wahba's problem minimizes the loss: $ L(R) = \sum_{i=1}^n w_i \| \mathbf{b}_i - R \mathbf{r}_i \|^2 $, where $ \mathbf{b}_i $ are observed unit vectors, $ \mathbf{r}_i $ are catalog references, $ w_i $ are weights (e.g., inversely proportional to uncertainty), and R is the rotation matrix (orthogonal, det(R)=1).  

Expanding, this simplifies to maximizing $ G(R) = \sum w_i (\mathbf{b}_i \cdot R \mathbf{r}_i) $, since minimizing L is equivalent.  

The attitude profile matrix is $ B = \sum w_i \mathbf{b}_i \mathbf{r}_i^T $—a 3×3 matrix capturing pairwise alignments. Then, decompose: S = B + B^T - trace(B) I (symmetric part), and Z = [B_{23} - B_{32}, B_{31} - B_{13}, B_{12} - B_{21}]^T (skew-symmetric vector).  

The Davenport K-matrix is:  

$K = \begin{bmatrix} \mathrm{tr}(B) & Z^T \\ Z & S \end{bmatrix}$

This 4×4 matrix encodes the problem, turning the optimization into an eigenvalue hunt.  

**Connection to Codebase Implementation**: In `build_davenport_matrix` from resolve.py, it computes B as the sum of weighted outer products (np.outer(b, r) * w), then constructs S, Z, and K as described.  

#### 2. QUEST Eigenvalue Solution

The optimal quaternion $ \hat{q} $ is the eigenvector of K corresponding to its maximum eigenvalue $ \lambda_{max} $. This comes from rewriting G(R) in quaternion form and maximizing under the unit norm constraint.  

To find $ \lambda_{max} $, solve the characteristic equation det(K - λ I) = 0, but numerically, compute all eigenvalues and pick the largest real one. The associated eigenvector is $ \hat{q} $, normalized to unit length.  

For efficiency, QUEST approximates $ \lambda_{max} $ iteratively, but the code uses direct eigendecomposition for precision.  

**Connection to Codebase Implementation**: The `quest_algorithm` in resolve.py calls np.linalg.eig(K), selects the index of the maximum real eigenvalue, extracts the eigenvector, and normalizes it to get the quaternion.  

#### 3. Quaternion to Rotation Matrix

Convert the quaternion to a matrix for applying rotations: For $ \mathbf{q} = [w, x, y, z] $ (with w = q_0),  

$$
R = \begin{bmatrix}1 - 2y^2 - 2z^2 & 2xy - 2wz & 2xz + 2wy \\2xy + 2wz & 1 - 2x^2 - 2z^2 & 2yz - 2wx \\2xz - 2wy & 2yz + 2wx & 1 - 2x^2 - 2y^2\end{bmatrix}
$$

This R transforms reference vectors to the observed frame (or vice versa, depending on convention).  

**Connection to Codebase Implementation**: `quaternion_to_matrix` in resolve.py implements this exact formula, handling the components directly.  

#### 4. Monte Carlo Uncertainty Estimation

To quantify errors, perturb the bearing vectors with Gaussian noise reflecting centroid uncertainty (from Section 4, propagated via Section 5). For each trial: add noise ~ N(0, σ_bearing), recompute matches if needed, run QUEST, and collect quaternions.  

Compute the mean quaternion (via averaging or more advanced methods like averaging on the SO(3) manifold) and covariance. The angular uncertainty is approximately $ \Delta \theta \approx 2 \|\sigma_{q_{vec}}\| $ radians, where $ \sigma_{q_{vec}} $ is the std of the vector part [q1,q2,q3].  

Converge by monitoring rolling std until below a tolerance (e.g., 1e-6), ensuring stable estimates.  

**Connection to Codebase Implementation**: In `determine_attitude` from monte_carlo_quest.py, it adds noise to bearings, batches QUEST calls, computes stats like mean and std of quaternions, and checks convergence on rolling statistics. 

#### 5. Residuals and Confidence Bounds

After estimation, compute residuals: $ \epsilon_i = \|\mathbf{b}_i - R \mathbf{r}_i\| $ (or angular: arccos(dot product)). High residuals indicate poor fits or outliers.  

Confidence from $ \lambda_{max} $ (closer to sum(w_i) means better alignment) or Monte Carlo: e.g., 95% bounds from the quaternion distribution. Reject if residuals exceed a threshold (e.g., 3σ).  

**Connection to Codebase Implementation**: The code validates by applying R to references and comparing to observations, with eigenvalue and residuals in output results.  

### Physical Insights and Limitations

The QUEST algorithm reveals the information-theoretic limits of attitude determination: with more matched stars (higher n), uncertainty drops as 1/sqrt(n), like averaging measurements; weights allow prioritizing brighter, less noisy stars for better precision. Physically, it underscores the need for diverse star patterns—collinear stars provide poor constraints on all axes, leading to degenerate solutions.  

Limitations include assumptions of Gaussian errors (real mismatches might be non-Gaussian) and equal handling of all pairs (advanced versions incorporate adaptive weights based on SNR from Section 2). For few pairs (n<3), solutions are underconstrained; Monte Carlo adds computational overhead but is vital for realism. Validation compares estimated attitudes to ground truth in simulations, often achieving arcsecond accuracy. As the pipeline's finale, this step delivers the actionable orientation, closing the loop from photons to navigation—enabling everything from satellite pointing to deep-space autonomy.  

*[Suggestion for Image]: A schematic illustration would be helpful here, depicting observed vectors on one side, catalog vectors on the other, with curved arrows showing the optimal rotation alignment via QUEST. Include eigenvalue spectra, a quaternion icon, and error ellipses from Monte Carlo trials to visualize uncertainty, making the optimization and probabilistic aspects more tangible for readers.]*

## Glossary

To complement the mathematical foundations detailed in this white paper, we provide a comprehensive glossary of key terms, concepts, and abbreviations used throughout the document. Terms are listed alphabetically, with definitions derived directly from the context of the star tracker radiometry simulation pipeline. Where applicable, cross-references to relevant sections are included for deeper exploration. This glossary aims to clarify technical jargon, ensuring accessibility for readers from diverse backgrounds in aerospace engineering, optics, and computational simulation.  

- **Adaptive Thresholding**: A detection technique that dynamically sets intensity cutoffs based on local image statistics (mean and standard deviation) to account for varying backgrounds and noise levels. Used in star detection to create binary masks. (See Section 4)  
- **Apparent Magnitude (m)**: A logarithmic measure of a star's brightness as observed from Earth, where lower values indicate brighter stars. It relates to photon flux via the Pogson scale, forming the basis for radiometric calculations. (See Section 2)  
- **Attitude Estimation**: The process of determining a spacecraft's orientation in space by finding the optimal rotation that aligns observed star vectors with catalog references. Solved using algorithms like QUEST. (See Section 7)  
- **Attitude Profile Matrix (B)**: A 3×3 matrix constructed from weighted outer products of observed and reference vectors, used in Wahba's problem to encode alignment information for attitude computation. (See Section 7)  
- **Bearing Vector**: A normalized 3D unit vector representing the direction from the camera to a star in the camera's reference frame, derived from pixel coordinates via inverse projection. Essential for pattern matching. (See Section 5)  
- **Centroiding**: The calculation of a star's sub-pixel position in an image using weighted moments (center of mass) of pixel intensities, accounting for noise and PSF shape. Achieves high precision like 0.1–0.3 pixels. (See Section 4)  
- **Connected Components Labeling**: An image processing step that groups adjacent pixels in a binary mask into distinct regions (blobs), filtered by criteria like area to identify potential stars. (See Section 4)  
- **Cramér-Rao Bound**: A theoretical lower limit on the uncertainty of position estimates (e.g., centroids), given by $ \sigma_{centroid} \geq \frac{\sigma_{psf}}{\sqrt{N_{ph}}} $, where $ \sigma_{psf} $ is PSF width and $ N_{ph} $ is photon count. (See Sections 3 and 4)  
- **Dark Current (I_d)**: Thermal generation of electrons in the detector, contributing Poisson-distributed noise (variance $ I_d t $) over exposure time t. A key factor in total noise variance. (See Sections 2 and 3)  
- **Davenport K-Matrix**: A 4×4 matrix derived from the attitude profile matrix, used in the QUEST algorithm. Its maximum eigenvalue and corresponding eigenvector yield the optimal quaternion. (See Section 7)  
- **Detection Threshold (T_detect)**: The minimum intensity value for flagging pixels as part of a star, often set adaptively (e.g., $ T_b = \mu_b + k \sigma_b $) to balance sensitivity and false positives. (See Section 4)  
- **Euler Angles (φ, θ, ψ)**: A set of three angles (roll, pitch, yaw) used to represent spacecraft attitude via sequential rotations around axes. Converted to rotation matrices for coordinate transformations. (See Section 1)  
- **False Positive Probability (P_FP)**: The likelihood of erroneous pattern matches, estimated as approximately the number of catalog triplets times $ (\Delta \theta_{tol} / \pi)^3 $. Minimized through tight tolerances. (See Section 6)  
- **Focal Length (f)**: The distance from the camera's optical center to the sensor plane, crucial for projecting 3D vectors to 2D coordinates and vice versa. Affects angular resolution. (See Sections 1 and 5)  
- **Full Well Capacity (N_FW)**: The maximum number of electrons a pixel can hold before saturation, limiting the signal for bright stars. (See Section 2)  
- **Gnomonic Projection**: A tangent-plane projection used for mapping celestial coordinates to a flat detector, assuming a pinhole model. Legacy mode for boresight-centered scenes. (See Section 1)  
- **Hungarian Algorithm**: An optimization method for solving assignment problems, such as bijective matching of observed stars to catalog entries by minimizing a cost matrix of angular differences. (See Section 6)  
- **Inter-Star Angle (θ_ij)**: The angular separation between two stars, computed as $ \theta_{ij} = \cos^{-1}(\mathbf{v}_i \cdot \mathbf{v}_j) $ for unit vectors. Rotation-invariant and key for pattern matching. (See Section 6)  
- **Match Confidence (C_match)**: A score (typically 0–1) quantifying how well observed and catalog patterns align, based on angular residuals (e.g., $ C = 1 - \frac{\sum |\delta \theta|}{\Delta \theta_{tol}} $). (See Section 6)  
- **Moment-Based Centroiding**: A method to compute star positions using intensity-weighted averages, akin to finding the center of mass. Includes uncertainty estimates from noise variance. (See Section 4)  
- **Monte Carlo Simulation**: A statistical technique involving repeated trials with random perturbations (e.g., noise on bearings) to estimate attitude uncertainty and convergence of quaternion distributions. (See Sections 3 and 7)  
- **Pattern Recognition**: The process of matching observed star patterns (e.g., triplets) to a celestial catalog using invariant features like sorted inter-star angles, enabling star identification. (See Section 6)  
- **Photon Flux (Φ)**: The rate of photons arriving from a star per unit area, derived from apparent magnitude and scaled by aperture and transmission. Basis for signal calculation. (See Section 2)  
- **Pinhole Camera Model**: An ideal projection model assuming light rays pass through a single point, used to map 3D directions to 2D focal plane coordinates (and inversely for bearing vectors). (See Sections 1 and 5)  
- **Pixel Pitch (p)**: The physical size of a detector pixel (e.g., 5.5 µm), used to convert focal plane distances to pixel coordinates. Influences angular error propagation. (See Sections 1 and 5)  
- **Point Spread Function (PSF)**: A 2D intensity distribution describing how light from a point source (star) spreads on the detector due to optics. Normalized as a probability map for photon simulation. (See Section 3)  
- **Poisson Distribution**: A statistical model for random photon arrivals, where the count k in a pixel follows $ Pr(k) = \frac{\lambda^k e^{-\lambda}}{k!} $, with mean and variance λ. Models shot noise. (See Section 3)  
- **Principal Point (c_x, c_y)**: The pixel coordinates where the optical axis intersects the detector, used in converting pixel positions to physical focal plane coordinates. (See Section 5)  
- **Quantum Efficiency (QE)**: The fraction of incident photons converted to photoelectrons in the detector, typically expressed as electrons per photon. Affects signal strength. (See Section 2)  
- **Quaternion (q)**: A four-component vector [q_0, q_1, q_2, q_3] representing 3D rotations compactly, avoiding gimbal lock. Used for attitude and converted to rotation matrices. (See Sections 1 and 7)  
- **QUEST (QUaternion ESTimator)**: An algorithm solving Wahba's problem by finding the eigenvector of the Davenport K-matrix corresponding to the maximum eigenvalue, yielding the optimal attitude quaternion. (See Section 7)  
- **Read Noise (σ_r^2)**: Gaussian-distributed electronic noise introduced during image readout, contributing to total variance independent of signal. (See Sections 2 and 3)  
- **Rotation Matrix (R)**: A 3×3 orthogonal matrix representing transformations between reference frames (e.g., inertial to camera). Preserves angles and derived from Euler angles or quaternions. (See Section 1)  
- **Shot Noise**: Random fluctuations in photon counts following Poisson statistics, with variance equal to the mean number of photons. Dominant for bright sources. (See Sections 2 and 3)  
- **Signal-to-Noise Ratio (SNR)**: The ratio of signal electrons to the square root of total noise variance, used to predict detection reliability (e.g., SNR > 5 for thresholds). (See Section 2)  
- **Triangle Geometric Invariants**: Sorted inter-star angles (a ≤ b ≤ c) of a star triplet, used as rotation-invariant identifiers for efficient catalog matching. (See Section 6)  
- **Wahba's Problem**: An optimization problem to find the rotation matrix minimizing the weighted sum of squared differences between observed and rotated reference vectors. (See Section 7)  

This glossary is compiled from the core concepts in Sections 1–7. For terms not listed here or requiring more scholarly depth (e.g., advanced derivations of the Cramér-Rao bound in imaging contexts), further academic resources can be explored. 
*"Ludwig Boltzmann, who spent much of his life studying statistical mechanics, died in 1906, by his own hand. Paul Ehrenfest, carrying on the work, died similarly in 1933. Now it is our turn to study statistical mechanics. Perhaps it will be wise to approach the subject cautiously".*

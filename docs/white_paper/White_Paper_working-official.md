# Star Tracker Radiometry Simulation: Mathematical Foundations

Dr. Jonathan Ryan (JR) Hoh, Lead Engineering Physicist -- Second Order Effects -- 9/18/2025

## Introduction

In order to understand the non-linear relationships between the parameters of a star tracker and its final attitude accuracy, the Engineering Physics Department at SOE has developed a Python-based simulation that acts as a "digital twin" for star tracker systems. This digital twin is essentially a digital replica of out physical system, allowing us to model and predict performance under countless scenarios without ever leaving the computer.  The simulation is built around the Basic Astronomical Star Tracker (BAST), an in-house custom star tracking algorithm designed for accurate attitude determination based on images of space. By simulating the entire process (from generating realistic star scenes to estimating the final spacecraft orientation) we can predict how well a star tracker will perform in space, identify potential limitations, and optimize designs before building hardware. For the task of creating low-cost COTS star trackers, this simulation provides the elusive superpower of determining which mechanical, optical, electrical, and algorithmic parameters are the easiest to cost-down. 

Here we present an investigative white paper which dives into the mathematical foundations that power our simulation. Key components will be broken down step-by-step, starting from the raw physics of light interacting with stars and sensors, all the way to advanced algorithms for matching stars and computing attitudes. The goal is to make these concepts accessible to anyone with the math and physics background from an engineering bachelor's degree. This will be accomplished by explaining all major equations in plain prose, defining technical terms as they appear, and providing intuitive motivations for why each piece matters. Along the way, the mathematics derived will be connected to the existing Python codebase, highlighting how these ideas are implemented in practice, and discuss real-world insights and limitations.

The paper is organized into seven core sections, each building on the last to form a complete simulation pipeline:

1. **Scene Generation Mathematics**: How we create virtual star images based on spacecraft orientation.
2. **Radiometric Chain Physics**: Modeling the flow of light from stars to sensor signals.
3. **Statistical Image Formation**: Adding realistic noise to simulate actual camera outputs.
4. **Detection and Centroiding Theory**: Finding and precisely locating stars in noisy images.
5. **Bearing Vector Geometry**: Converting 2D image positions into 3D directions.
6. **Pattern Recognition Mathematics**: Matching observed stars to known catalogs.
7. **Optimal Attitude Estimation**: Calculating the final spacecraft orientation.

To ensure clarity and consistency, each section follows a rigidly structured format consisting of:

* **Problem Statement**: A clear description of the challenge we're addressing.
* **Critical Parameters**: Key variables and terms, defined upfront.
* **Step-by-Step Mathematical Derivation**: Detailed equations with word explanations before and after, so nothing feels like a black box.
* **Connection to Codebase Implementation**: How this math translates to our Python code.
* **Physical Insights and Limitations**: Real-world takeaways, including what works well and where assumptions might break down.

We'll also include a glossary of all technical terms and variables for quick reference. For the sake of those reading sections individually or out of order, a glossary can be found at the end of the document which contains definitions of all technical terms and major variables seen.
Although there are a number of places where this analysis could begin, we start in Section 1 by laying the groundwork for generating realistic star scenes from synthetic catalogs of rest-frame observation vectors. This allows us to set the baseline scenes that we are attempting to solve for attitude.

<img title="Figure 1: Cartoon of Star Tracker and Bearing vectors to detected stars" src="file:///H:/Star%20Tracker/White%20Paper/Images/Created%20Images/Intro_Image.png" alt="loading-ag-1573" data-align="inline" style="zoom:33%;">

## Section 1: Scene Generation Mathematics

Before understanding how a star tracker "sees" stars, a realistic snapshot of the night sky as viewed from a spacecraft is required. This involves taking the known positions of stars in the universe and transforming them into where they'd appear on the camera's sensor, based on the spacecraft's orientation (or "attitude"). Why is this important? Without accurately generating these star positions, all the downstream steps (like modeling light intensity, adding noise, detecting stars, and estimating attitude) would be based on flawed inputs, leading to unreliable simulation results. This section lays the foundation by ensuring the simulated scene preserves the true geometric relationships between stars, which is crucial for the pattern-matching algorithms that come later in the pipeline (explored in Section 6).

In essence, this section will bridge the gap between celestial maps and pixel coordinates. Starting with stars' positions in a fixed, Earth-centered reference frame and we apply rotations to mimic the spacecraft's viewpoint, projecting them onto a virtual camera detector. This process uses concepts from coordinate geometry and linear algebra, but can be broken down intuitively: imagine sitting in a planetarium theater as a dome of stars orbits the audience and someone snaps a photo from their seat. The mathematics derived in this section connects the locations of the stars seen in the resulting 2-dimensional photo their locations on the projected celestial sphere of the planetarium.

### Problem Statement

To create realistic star tracker images, we need to convert the positions of stars from their standard celestial coordinates into precise locations on the camera's detector, while accounting for the spacecraft's arbitrary orientation. This transformation must preserve the angular distances (inner-angles) between stars, as these are key for later identifying and matching them against star catalogs.

<img title="Figure 2: Simple example of inner-angles between 3 observed points relative to observer frame" src="file:///H:/Star%20Tracker/White%20Paper/Images/Created%20Images/inner_angles.png" alt="Figure 2: Simple example of inner-angles between 3 observed points relative to observer frame" style="zoom:25%;" data-align="inline">

### Critical Parameters

Before diving into the math for each section, we will establish and define all of the main components that will be seen in the formulae:

* **Rotation matrices** $R(\phi, \theta, \psi)$: 3x3 matrices that describe how to rotate coordinates from one frame to another, based on Euler angles (roll $\phi$, pitch $\theta$, yaw $\psi$).
* **Quaternion components** $[q_0, q_1, q_2, q_3]$: A compact, four-element representation of 3D rotations that avoids issues like gimbal lock in Euler angles (deeper explanation below).
* **Focal plane coordinates** $(x_f, y_f)$: Positions on an imaginary 2D plane inside the camera, in physical units like microns.
* **Detector pixel positions** $(u, v)$: The final grid locations on the sensor, measured in pixels.
* **Angular preservation error** $\Delta \theta$: A measure of how much angular distances might deviate due to approximations (ideally close to zero for accuracy).

These parameters ensure the transformations are both mathematically sound and practically implementable.

### Step-by-Step Mathematical Derivation

Here we derive the full transformation chain from celestial coordinates to pixel positions, with the purpose of intuitively describing each step in prose as well as mathematics. The process at its core involves converting spherical positions to 3D vectors, applying rotations for attitude, projecting onto a plane, and mapping to pixels... all while keeping angles between stars unchanged.

#### 1. Spherical to Cartesian Inertial Vectors

Stars are typically listed in catalogs using spherical coordinates: right ascension $\alpha$ (like longitude) and declination $\delta$ (like latitude) on the celestial sphere. To work with rotations, we first convert these to 3D Cartesian unit vectors in an inertial (fixed, Earth-centered) frame which we call $\mathbf{v}_I$. This is like pointing from the center of a globe to a point on its surface, except here the globe is the entire celestial sphere.

The conversion formula is:

$$
\mathbf{v}_I = \begin{pmatrix}\cos \delta \cos \alpha \\\cos \delta \sin \alpha \\\sin \delta\end{pmatrix}
$$

This results in a unit vector (length 1), **ensuring all stars are treated as if they're at the same infinite distance**. The camera's boresight (its pointing direction) is converted similarly to a vector $\mathbf{v}_B$. This step normalizes everything to directions rather than distances, which is perfect for attitude determination since stars are effectively points at infinity.

**Connection to Codebase Implementation**: This is handled in the function `radec_to_inertial_vector` within `attitude_transform.py`, directly translating catalog data into usable vectors.

![Figure 3: Illustration of the components of the celestial sphere](https://www.lpi.usra.edu/education/skytellers/constellations/images/celest_sphere.jpg "Figure 3: Illustration of the components of the celestial sphere")

*image credit: J.A. Hester AST 111 course lecture, Maricopa Community College*

#### 2. Attitude Representation and Rotation Matrices

Next, we account for the spacecraft's orientation using either Euler angles or quaternions to build a rotation matrix $R$. This matrix transforms star vectors from the inertial frame to the camera's frame, simulating how the view changes as the spacecraft rotates. Intuitively, it's like tilting your head while looking at the sky, where the stars' relative positions stay the same, but their locations in your field of view shift.

For Euler angles (in ZYX convention, meaning yaw around Z, then pitch around Y, then roll around X), we construct individual rotation matrices and multiply them:

* Yaw (Z-axis): $R_z(\psi) = \begin{pmatrix} \cos \psi & -\sin \psi & 0 \\ \sin \psi & \cos \psi & 0 \\ 0 & 0 & 1 \end{pmatrix}$

* Pitch (Y-axis): $R_y(\theta) = \begin{pmatrix} \cos \theta & 0 & \sin \theta \\ 0 & 1 & 0 \\ -\sin \theta & 0 & \cos \theta \end{pmatrix}$

* Roll (X-axis): $R_x(\phi) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos \phi & -\sin \phi \\ 0 & \sin \phi & \cos \phi \end{pmatrix}$

The full rotation matrix is $R = R_z(\psi) R_y(\theta) R_x(\phi)$.

Alternatively, for quaternions (which are more efficient and avoid singularities), the rotation matrix is derived as:

$$
R = \begin{pmatrix}q_0^2 + q_1^2 - q_2^2 - q_3^2 & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\2(q_1 q_2 + q_0 q_3) & q_0^2 - q_1^2 + q_2^2 - q_3^2 & 2(q_2 q_3 - q_0 q_1) \\2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & q_0^2 - q_1^2 - q_2^2 + q_3^2\end{pmatrix}
$$

We then apply the transformation: $\mathbf{v}_C = R \mathbf{v}_I$, where $\mathbf{v}_C$ is the star vector in the camera frame. Stars with a negative z-component ($v_{C,z} \leq 0$) are behind the camera and filtered out.

A key property here is angular preservation: since rotation matrices are orthogonal ($R^T R = I$, where $I$ is the identity matrix), the dot product (and thus angles) between vectors remains the same: $\mathbf{v}_{I1}^T \mathbf{v}_{I2} = \mathbf{v}_{C1}^T \mathbf{v}_{C2}$, so $\theta = \cos^{-1}(\mathbf{v}_1^T \mathbf{v}_2)$ is invariant. This ensures patterns look consistent regardless of rotation.

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

$u = u_0 + \frac{x_f}{p}, \quad v = v_0 + \frac{y_f}{p}$

Where $(u_0, v_0)$ is the principal point (usually the detector's center), and $p$ is the pixel pitch (e.g., 5.5 µm for a CMV4000 sensor). We filter out positions outside the detector bounds (e.g., 0 to 2048 pixels) with a small margin to avoid edge effects.

**Connection to Codebase Implementation**: Handled by `focal_plane_to_pixels` and `filter_detector_bounds` in `attitude_transform.py`.

![eea35bed-8b1b-423c-92af-af43da464406](file:///C:/Users/Dr.%20J/OneDrive/Pictures/Typedown/eea35bed-8b1b-423c-92af-af43da464406.png)

#### 5. Integrated Gnomonic Projection (Legacy Mode)

For simpler cases without full attitude rotations (e.g., boresight-centered views, a.k.a. the "trivial" rotation), we use a gnomonic projection, which maps the celestial sphere onto a tangent plane. This involves defining basis vectors (east $\mathbf{e}$ and north $\mathbf{n}'$) relative to the boresight and projecting stars onto this plane before converting to pixels. It's like unfolding a portion of the sphere onto a flat map, useful for quick approximations but less flexible than the full rotation method.

**Connection to Codebase Implementation**: Found in `_sky_to_detector` within `scene_generator.py`.

### Physical Insights and Limitations

This transformation chain allows us to generate multi-star scenes for any spacecraft attitude, preserving the geometry needed for accurate pattern matching in later sections. Physically, it highlights how sub-pixel precision is tied to hardware: angular error per pixel is roughly $\Delta \theta \approx \frac{p}{f}$ radians, so a longer focal length improves resolution but narrows the field of view (potentially seeing fewer stars).

However, limitations include assuming an ideal pinhole model (no lens distortions, which could skew positions in real cameras) and validity only for small fields-of-view (under 20° to avoid projection inaccuracies). We validate by checking that pairwise angles remain consistent after transformations. Overall, this step ensures the simulation starts with a faithful representation of the sky, setting up the radiometric modeling in Section 2, where we'll add light and intensity to these positions.

<img title="" src="file:///H:/Star%20Tracker/White%20Paper/Images/diagram-export-9-17-2025-4_11_25-PM.png" alt="" style="zoom:67%;">

## Section 2: Radiometric Chain Physics

Once we've generated a virtual star scene with accurate positions on the detector (as covered in Section 1), the next challenge is simulating how much light from those stars actually reaches and registers on the camera sensor. This is the realm of radiometry, the study of measuring electromagnetic radiation like visible light from stars. In a real star tracker, the brightness of stars determines the strength of the signal we can detect amidst noise; too faint, and stars might get lost in the background, while too bright could saturate the sensor. By modeling this "radiometric chain," we predict realistic signal levels, which directly feed into the noisy image formation in Section 3. Think of it as calculating the intensity of light that powers the rest of the simulation: get this wrong, and the digital twin won't match real-world performance.

Here we trace the journey of starlight from its source to electron signals in the detector, using physics principles like flux and quantum efficiency. This involves converting abstract concepts like stellar magnitude into concrete photon counts, all while incorporating camera properties. The math draws from optics and astrophysics, but can still be explained intuitively: imagine starlight as a stream of particles (photons) that get filtered and collected, much like rainwater funneled into a bucket.

### Problem Statement

To create a faithful simulation, we must link a star's apparent brightness (its magnitude) to the actual number of photons hitting the detector, factoring in the camera's optics and sensor characteristics. This establishes the baseline signal strength, which sets performance limits for detection and helps predict when noise might overwhelm the signal.

### Critical Parameters

* **Photon flux** $\Phi$: The rate at which photons from a star arrive at Earth, in photons per second per square meter.
* **Quantum efficiency** $QE$: The fraction of incoming photons that generate electrons in the sensor (e.g., 0.8 means 80% efficiency).
* **Optical transmission** $\tau$: The fraction of light that passes through the lens without being absorbed or scattered (e.g., 0.9 for high-quality optics).
* **Detector full well capacity** $N_{FW}$: The maximum number of electrons a pixel can hold before saturating (e.g., 20,000 electrons).
* **Dark current** $I_d$: Unwanted electron generation in the sensor due to heat, in electrons per second per pixel.
* **Signal-to-noise ratio** $SNR$: A measure of signal quality, calculated as signal divided by noise; higher values mean clearer detections.

### Step-by-Step Mathematical Derivation

#### 1. Stellar Flux from Apparent Magnitude

Stars' brightness is measured on the apparent magnitude scale, a logarithmic system where lower numbers mean brighter objects (e.g., the Sun is -26.74, while faint stars might be +6 or higher; this system is a relic from a time before logic was invented). This scale is based on human eye perception and requires conversion to physical units like energy flux in order to be computationally useful.

The Pogson relation says that a magnitude difference $\Delta m$ corresponds to a flux ratio of $10^{-0.4 \Delta m}$, which simplifies to approximately $2.512^{-\Delta m}$ (each magnitude step dims by a factor of about 2.512). As a reference, astronomers commonly make use of the Sun: its apparent magnitude $m_\odot = -26.74$ and solar constant $S = 1366$ W/m² (energy per second per square meter at Earth).

To work with photons, calculate the average photon energy for visible light using a central wavelength $\lambda_c$ (midpoint of the camera's bandpass, say 550 nm for green light):

$E_{ph} = \frac{h c}{\lambda_c}$

Here, $h$ is Planck's constant (6.626 × 10^{-34} J s), and $c$ is the speed of light (3 × 10^8 m/s). This gives the energy per photon in joules.

The Sun's photon flux is then $\Phi_\odot = S / E_{ph}$ (photons/s/m²), whos dimensions refer to the flux over a unit area. For a zero-magnitude star (a standard reference), scale up from the Sun:

$\Phi_0 = \Phi_\odot \times 2.512^{m_\odot}$

Finally, for a star of magnitude $m$:

$\Phi = \Phi_0 \times 2.512^{-m}$

This $\Phi$ is the photon arrival rate at Earth. **Note that the code approximates 2.512 as 2.5 for simplicity**, introducing a small ~1.7% error, but it's close enough for most simulations purposes and is kept steady across all runs.

**Connection to Codebase Implementation**: The class function `star.calculate_flux` from the codebase computes a constant based on solar values (with a 1e-6 unit adjustment) and scales by $2.5^{-m}$, then divides by photon energy to get flux.

#### 2. Collected Photons at Aperture

The camera's lens acts like a collector: its aperture area $A = \pi (D/2)^2$ (where $D$ is the diameter) determines how many photons are gathered. The incident photon rate is $\Phi \times A$.

Not all light makes it through (some is lost in the optics) so multiply by transmission $\tau$:

$\Phi_{opt} = \Phi \times A \times \tau$

Over an integration time $t$ (exposure duration in seconds), the total photons collected are:

$N_{ph} = \Phi_{opt} \times t$

This assumes a point source like a star, where all light focuses onto a small area on the sensor. Since the purpose of the simulation is star tracking, this assumption will always hold true.

**Connection to Codebase Implementation**: The `calculate_optical_signal` function computes the signal as star flux times aperture area, times transmission, times integration time (adjusted from ms to s).

#### 3. Detector Response and Electron Generation

The sensor converts photons to electrons via the photoelectric effect, with efficiency $QE$:

$N_e = N_{ph} \times QE$

This $N_e$ is the raw signal. However, the detector has limits: if $N_e > N_{FW}$, the pixel saturates and loses information. Additionally, dark current adds spurious electrons: $I_d \times t$, even without light. Note that these effects set practical bounds on detectable magnitudes.

**Connection to Codebase Implementation**: QE is a parameter in the focal plane array (FPA) class, applied to photon counts, though noise is handled separately in image simulations.

#### 4. Noise Contributions

Real signals are noisy. Key sources include:

* **Shot noise**: From the random arrival of photons, following a Poisson distribution with variance $N_{ph}$.

* **Dark noise**: Variance $I_d t$ from dark current.

* **Read noise**: Gaussian noise during readout, with variance $\sigma_r^2$. 
  *Read noise is an often misunderstood noise source which deserves slightly more explanation. It arises from the analog-to-digital conversion process when the sensor reads out pixel values. Unlike shot noise (which depends on signal strength), read noise is **signal-independent** meaning it's the same whether the pixel received many photons or none. It stems from thermal fluctuations in the readout electronics and follows a Gaussian distribution with zero mean and variance σ²_r. Typical values are 5-15 electrons RMS for modern CMOS sensors. This noise floor limits the detection of faint stars, as it sets a baseline "hiss" that can't be reduced by longer exposures.*

The total noise variance is approximately $\sigma^2 = N_e + I_d t + \sigma_r^2$ (after applying QE). The SNR is then:

$SNR = \frac{N_e}{\sigma}$

For reliable detection, SNR typically needs to exceed 5–10. This helps set thresholds for what's visible in the simulated image.

**Connection to Codebase Implementation**: These primary noise sources are modelled to first order throughout the radiometry chain with values determined by the CMV4000 datasheet. This is one element of the current simulation which could use further investigation and improvement.

#### 5. Spectral Integration

For precision, flux integration over the full wavelength spectrum can be carried out:

$\Phi = \int_{\lambda_{min}}^{\lambda_{max}} F(\lambda) \frac{\lambda}{h c} \, d\lambda$

However, to save computation, the code simplifies to a central wavelength approximation, which works well for broadband visible light but might need refinement for narrow filters.

#### 6. Consolidated Radiometric Equation

For a star of apparent magnitude $m$, the expected photon flux $\lambda$ incident on the detector follows:

$\lambda = F_0 \cdot 10^{-0.4m} \cdot A_{\text{eff}} \cdot \Delta t \cdot \eta(\lambda_0)$

where $F_0$ is the zero-magnitude flux density, $A_{\text{eff}}$ is the effective collection area, $\Delta t$ is the integration time, and $\eta(\lambda_0)$ represents the quantum efficiency at the effective wavelength. This is a key equation for us since **the primary simulation output of the radiometry chain is the expected photon flux** (the number of photons per second per unit area expected to hit our detector).

This unified expression consolidates the entire radiometric chain derived in the preceding steps, providing a direct relationship between stellar brightness and detector signal for star tracker applications. The equation elegantly captures the complete photon collection process: stellar magnitude conversion ($10^{-0.4m}$), optical gathering power ($A_{\text{eff}}$), temporal integration ($\Delta t$), and detector efficiency ($\eta(\lambda_0)$). 

### Physical Insights and Limitations

This radiometric model reveals how star brightness translates to usable signals: brighter stars (lower magnitudes) yield more photons and higher SNR, enabling better precision, while faint ones risk being drowned out by noise. Physically, it underscores trade-offs like aperture size (larger collects more light but increases weight) and exposure time (longer improves signal but risks motion blur).

Limitations include assumptions of blackbody-like star spectra and uniform QE (real stars vary in color, and sensors in sensitivity), plus ignoring atmospheric effects (fine for space but not ground tests which will likely be used as early-stage verification opportunities). Bright stars can saturate ($N_e > N_{FW}$), capping dynamic range, and approximations like 2.5 vs. 2.512 introduce minor errors. Overall, this chain predicts detectability, paving the way for Section 3, where we add statistical noise to form realistic images, turning these photon counts into pixelated, imperfect snapshots.



![radiometrc_flow_chart](file:///H:/Star%20Tracker/White%20Paper/Images/radiometrc_flow_chart.png)

## Section 3: Statistical Image Formation

With the positions and brightness of stars established in the virtual scene and their photon signals calculated, we now need to simulate what a real camera image would look like, complete with the imperfections of noise and randomness. This step is crucial because actual star tracker images are fuzzy due to the point spread function (PSF, which describes how light from a point source spreads out on the sensor) and riddled with statistical fluctuations from photon arrivals and electronic noise. Properly modelling these complexities is one of the primary obstacles to developing a full star-tracking pipeline simulation as the mathematics are complex and require external simulations for reliable PSF distributions. Think of it as sprinkling unpredictability onto an otherwise ideal picture: photons don't arrive in a steady stream but in random bursts, like raindrops on a window, and the sensor adds its own electronic "hiss" that can obscure faint signals.

Here, we model image formation using probability and statistics, focusing on Poisson processes for photon counts and Gaussian approximations for other noises. This creates a probabilistic "noisy" image from the deterministic signals, allowing us to test the limits of star detection under real-world conditions. The math draws from stochastic processes, but we'll frame it intuitively through thousands of photon "dice rolls" to build up the image pixel by pixel.

### Problem Statement

Real camera sensors introduce shot noise (from random photon arrivals) and readout noise (from electronics), which limit how precisely we can locate stars. To make our digital twin accurate, we must incorporate these effects using Poisson statistics for photons and additional models for detector imperfections, enabling predictions of detection reliability and validation of sub-pixel positioning techniques.

### Critical Parameters

Once again, we define the primary characters at play before starting the mathematics:

* **Poisson parameter** $\lambda$: The expected number of photons (or electrons) in a pixel, which also equals the variance for Poisson-distributed counts.
* **Read noise variance** $\sigma_r^2$: The spread of random electronic noise added during image readout, typically in electrons squared (e.g., 10 e⁻ RMS).
* **Pixel response non-uniformity** $\epsilon_{PRNU}$: Small variations in how pixels respond to the same light level, expressed as a percentage (e.g., 1% means pixels vary by ±0.01 in sensitivity).
* **PSF sampling error** $\Delta x_{sample}$: The positional inaccuracy introduced when shifting the PSF to sub-pixel locations, in pixels (e.g., about 1/12 pixel for basic interpolation).
* **Total noise equivalent electrons** $N_{noise}$: A summary metric of all noise sources, in electrons, helping estimate the faintest detectable signal.

These parameters capture the stochastic nature of imaging, linking hardware specs to simulation outcomes.

### Step-by-Step Mathematical Derivation

Next, we derive the process of building a noisy image from two critical inputs: **PSFs and photon counts**.  This involves treating the PSF as a probability map, simulating random photon arrivals, adding detector effects, and handling sub-pixel precision, all to mimic how real images form.

#### 1. PSF as Probability Distribution

The point spread function (PSF) describes how light from a star blurs across pixels due to optics and diffraction; in essence a 2D map of light intensity. To model randomness, we treat the PSF as a probability distribution: for each photon, it dictates the chance of landing in a specific pixel (i,j). First, normalize it so the total adds up to 1:

$\sum_{i,j} P(i,j) = 1$

Here, $P(i,j)$ is the normalized intensity at pixel (i,j), often derived from optical simulations like Zemax software. This normalization turns the PSF into a proper probability density, ensuring that if you "throw" many photons at it, their distribution matches the expected blur pattern.

**Connection to Codebase Implementation**: The `normalize_psf` function in `psf_photon_simulation.py` divides the PSF matrix by its sum (if non-zero), creating this probability map directly from input data. As mentioned, realistic PSF models have to be simulated by powerful outside programs. The PSFs used in this simulation are Zemax simulation outputs from simulating the PSF of starlight seen by a satellite in LEO.

<img title="Zemax PSF Simulation" src="file:///H:/Star%20Tracker/White%20Paper/Images/simple_PSF.png" alt="" style="zoom:67%;">

#### 2. Poisson Photon Arrival

Photons don't arrive uniformly; their counts follow a Poisson distribution, reflecting the quantum randomness of light. For a total number of photons $N_{ph}$ from a star (calculated in Section 2), the expected count in pixel (i,j) is:

$\lambda_{i,j} = N_{ph} P(i,j)$

The actual count $k_{i,j}$ is then drawn from a Poisson random variable:

$Pr(k_{i,j} = k) = \frac{\lambda_{i,j}^k e^{-\lambda_{i,j}}}{k!}$

This means the variance equals the mean ($\lambda_{i,j}$), creating "shot noise" that's more pronounced for fewer photons.

**In Layman's Terms:** Think of the PSF as a stencil with holes of different sizes. The bigger holes represent areas where photons are more likely to land, smaller holes where they're less likely. When we "spray paint" (photons) through this stencil onto paper (the detector), some paint gets through the big holes, some through the small holes, and the pattern on the paper follows the stencil's shape. But here's the key: **each spray is random**. Even using the exact same stencil and the same amount of paint, every time you spray, you get a slightly different pattern. Sometimes more paint lands in one spot, sometimes less. That's the Poisson randomness. The stencil (PSF) determines where paint is *likely* to go, but chance determines where it *actually* goes in each spray attempt.

This is exactly what happens when starlight hits our detector - the PSF acts as nature's stencil, and photons are the random spray paint creating slightly different patterns each time, even from the same star.

For implementation, we can either sample per pixel using this distribution or, for Monte Carlo accuracy, simulate individual photon locations by repeatedly sampling from the PSF probabilities. In the case of the full pipeline simulation, **everything from this point forward is simulated with Monte Carlo statistics**. This is what allows for the simulation to numerically compute relationships between variables which would be impossible to solve analytically. This analogy is shown visually in the following figure.

<img src="file:///H:/Star%20Tracker/White%20Paper/Images/Monte_Carlo_PSF.png" title="" alt="" style="zoom:80%;">

The normalized PSF $\psi(x,y)$ acts as a probability density function satisfying:

$\iint \psi(x,y) , dx , dy = 1$

Each of the $N$ photons is assigned a detector position $(x_i, y_i)$ by sampling from the PSF distribution. The discrete photon image $I(x,y)$ is constructed by accumulating individual photon events:

$I(x,y) = \sum_{i=1}^{N} \delta(x - x_i, y - y_i)$

where $\delta$ represents the Dirac delta function.

The simulation process consists of:

1. **Temporal Sampling**: Draw photon count $N \sim \text{Poisson}(\lambda)$
2. **Spatial Sampling**: For each photon $i$, sample position $(x_i, y_i)$ from $\psi(x,y)$
3. **Discrete Realization**: Construct detector image $I(x,y)$ through photon accumulation

This methodology preserves both the Poisson photon noise characteristics and the optical system's spatial response, enabling realistic performance predictions for centroiding algorithms and detection thresholds under shot noise-limited conditions.

**Connection to Codebase Implementation**: The `simulate_psf_with_poisson_noise` function computes expected_counts as psf_normalized * num_photons, then uses np.random.poisson(expected_counts) to generate noisy counts for each pixel across multiple trials. An alternative, `simulate_photon_distribution`, flattens the PSF into probabilities and samples photon locations directly.

#### 3. Detector Response Modeling

Once we have photon counts, the sensor converts them to digital values, but not perfectly. We add effects like gain (amplification), offset (baseline signal), read noise (random fluctuations during readout), and dark current (thermal electrons, also Poisson-distributed). The digital number (DN) for pixel (i,j) is:

$DN_{i,j} = g (k_{i,j} + d_{i,j}) + o + r_{i,j}$

Here, $g$ is the gain (meaning the number of electrons generated per photon, though QE from Section 2 is often folded in), $d_{i,j} \sim Poisson(I_d t)$ is dark current electrons over exposure time t, $o$ is a fixed offset, and $r_{i,j} \sim \mathcal{N}(0, \sigma_r^2)$ is Gaussian read noise. Finally, clamp DN to the sensor's bit depth (e.g., 0 to 4095 for 12 bits) to simulate saturation.

This equation builds a complete pixel value, incorporating both quantum and electronic effects for realism.

#### 4. Sub-Pixel PSF Placement and Interpolation

Stars don't always land exactly on pixel centers, so we shift the PSF by fractional amounts to place it at precise (x,y) positions from Section 1. This requires interpolation, like bilinear (linear blending between neighbors) or spline methods, to create a shifted PSF grid. The sampling error for bilinear interpolation is approximately:

$\Delta x_{sample} \approx \frac{p}{12}$

Where p is pixel size. This represents the average positional error from approximating the continuous PSF on a discrete grid.

**Connection to Codebase Implementation**: In `multi_star_radiometry.py`, the `_place_psf_at_position_subpixel` function uses scipy.ndimage.shift with order=1 (linear interpolation) to handle these fractional shifts accurately.

#### 5. Total Noise and Error Propagation

Combining all sources, the variance for each pixel's electron count is:

$\sigma_{i,j}^2 = g^2 (\lambda_{i,j} + I_d t) + \sigma_r^2$

This includes scaled shot and dark noise plus read noise. For the whole star signal, SNR is the total electrons divided by the square root of the total variance. Errors propagate: for example, PSF sampling affects centroiding (previewed here, detailed in Section 4 via the Cramér-Rao bound, which sets a theoretical minimum uncertainty for position estimates).

**Connection to Codebase Implementation**: Simulations in the code compute mean_image and std_image over multiple Monte Carlo trials, effectively capturing this total noise and its propagation.

### Physical Insights and Limitations

This statistical modeling shows how noise sets fundamental limits: for bright stars with many photons, shot noise (sqrt of signal) dominates, allowing precise measurements; for faint ones, fixed read noise takes over, potentially hiding them entirely. Sampling and interpolation highlight the need for high-resolution PSFs to avoid aliasing artifacts, like jagged edges in the simulated blur.

Limitations include assuming pixels are independent (ignoring real effects like crosstalk, where light leaks between pixels, or PRNU variations), and the central wavelength approximation from Section 2 might not capture color-dependent noise. The model is validated by checking that averaged Monte Carlo images match the expected PSF shape. Overall, this step produces the "raw" noisy images essential for testing detection in Section 4, where we'll extract star positions from this simulated mess.

## Section 4: Detection and Centroiding Theory

Now that we've simulated a noisy, realistic image of stars on the detector, the next step is to extract useful information from it: specifically, identifying where the stars are and pinpointing their exact positions with sub-pixel accuracy. This process, known as **detection and centroiding**, is like finding needles in a haystack. However, instead of sifting through hay to find needles, we sift through the image for bright spots that stand out from the noise, then calculate their "brightness center" (equivalent to center of mass) to get precise coordinates. These positions are the raw data for converting 2D pixels into 3D directions (bearing vectors, covered in Section 5), and any errors here propagate through the entire pipeline, potentially throwing off star matching and attitude estimation. In low-SNR scenarios (e.g., faint stars or high noise), poor detection could miss stars altogether, while inaccurate centroiding might lead to mismatched patterns.

We'll approach this using image processing techniques grounded in statistics and moments (a mathematical way to describe shapes). Intuitively, detection sets a "brightness cutoff" to flag potential stars, then centroiding weighs the pixel intensities to find the average position, much like balancing a unevenly loaded tray to find its center. The math involves adaptive thresholding and weighted sums, but we'll explain how it handles real-world unpredictabilities like varying backgrounds or overlapping blurs.

### Problem Statement

From a noisy image, we need to robustly detect regions corresponding to stars and compute their sub-pixel centroids (positions) with 0.1–0.3 pixel accuracy. The algorithms must adapt to non-uniform noise, reject false positives (like hot pixels), and work for both isolated and clustered stars, all while minimizing biases from finite image windows or PSF shapes.

### Critical Parameters

Some more key terms:

* **Detection threshold** $T_{detect}$: The minimum intensity value (in digital counts) a pixel must exceed to be considered part of a star; often set dynamically based on local noise.
* **Centroid coordinates** $(x_c, y_c)$: The calculated sub-pixel position of a star's center, in pixel units.
* **Centroiding uncertainty** $\sigma_{centroid}$: The estimated error in the centroid position, typically in fractions of a pixel (e.g., 0.2 pixels).
* **Moment calculation weights** $w_i$: Intensity values used to "weight" each pixel's contribution to the centroid, emphasizing brighter areas.
* **Region selection criteria**: Rules like minimum/maximum area (in pixels) or total intensity to filter valid star blobs from noise or artifacts.

These parameters ensure the process is tunable to different camera setups and noise levels.

### Step-by-Step Mathematical Derivation

We'll derive the detection and centroiding process from basic image statistics to precise position estimates. This builds from identifying candidate regions to refining their centers, incorporating noise considerations to quantify reliability.

#### 1. Adaptive Thresholding for Detection

Images have varying backgrounds (e.g., due to dark current or stray light), so a fixed threshold might miss faint stars in noisy areas or flag too many false positives in quiet ones. To adapt, divide the image into blocks (e.g., B×B pixels, like 32×32) and compute local statistics: the mean intensity $\mu_b$ and standard deviation $\sigma_b$ for each block.

The local threshold is then:

$T_b = \mu_b + k \sigma_b$

Here, $k$ is a multiplier (typically 3–5) chosen to balance sensitivity and false alarms; higher $k$ is more conservative. Different $k$ values allow for testing different extremes, but a "chosen" value has yet to be decided. Apply this to create a "binary mask": pixels above $T_b$ are marked as "1" (potential star), others as "0". This adaptive approach ensures detection works across the entire image, even if noise levels vary.

**Connection to Codebase Implementation**: In `detect_stars_and_calculate_centroids` (likely in a file like identify.py), it uses cv2.resize for multi-scale processing, then computes local mean and std to set the threshold dynamically. In the following figure, we see an example of this thresholding stage with the threshold manually set to $k=4$.<img title="" src="file:///C:/Users/Dr.%20J/iCloudDrive/SOE_Cloud/Star%20Tracker/Phase%201/BAST%20animation/output_5_0.png" alt="output_5_0" style="zoom:100%;" data-align="left">

#### 2. Connected Components Labeling

With the binary mask, group adjacent "1" pixels into "blobs" using 8-connectivity (pixels touching horizontally, vertically, or diagonally are connected). This identifies distinct regions that might be stars.

Filter these blobs to reject noise: require a minimum area (e.g., 3 pixels to avoid single hot pixels) and maximum area (e.g., 50 pixels to exclude large artifacts or merged stars). For single-star images, select the brightest blob by summing its intensities; for multi-star scenes, keep all that pass the filters. This step refines candidates, reducing computational load for centroiding.

**Connection to Codebase Implementation**: Handled by `cv2.connectedComponentsWithStats` in a function like `group_pixels` (in identify.py), which filters regions by size and other stats.



<img title="" src="file:///C:/Users/Dr.%20J/iCloudDrive/SOE_Cloud/Star%20Tracker/Phase%201/BAST%20animation/output_7_0.png" alt="output_7_0" style="zoom:100%;">

#### 3. Moment-Based Centroiding

For each valid region, compute the centroid as the intensity-weighted average position, like finding the balance point of a shape where brighter pixels "pull" harder. For pixels at positions (x, y) with intensities I(x, y):

$x_c = \frac{\sum x \, I(x,y)}{\sum I(x,y)}, \quad y_c = \frac{\sum y \, I(x,y)}{\sum I(x,y)}$

This is the first-order moment (center of mass). The sums are over the region's pixels, and weights $w_i = I(x,y)$ emphasize the core of the PSF where signal is strongest.

To estimate uncertainty due to noise, approximate the variance in $x_c$:

$\sigma_{x_c}^2 \approx \frac{\sum (x - x_c)^2 I(x,y)}{\left( \sum I(x,y) \right)^2 } \times \sum I(x,y)$

This accounts for Poisson noise (variance equals mean intensity). A theoretical lower limit is the Cramér-Rao bound:

$\sigma_{centroid} \geq \frac{\sigma_{psf}}{\sqrt{N_{ph}}}$

Where $\sigma_{psf}$ is the PSF width (e.g., in pixels) and $N_{ph}$ is the total photons. More photons mean tighter bounds, like averaging more data points for better precision.

**Connection to Codebase Implementation**: The `calculate_centroid` function in identify.py computes these weighted moments directly from the region's pixel data.

<img title="" src="file:///H:/Star%20Tracker/White%20Paper/Images/detection_and_centroiding/output_9_0.png" alt="output_9_0" style="zoom:67%;" data-align="left">

#### 4. Bias and Window Effects

Centroids can be biased if the calculation window cuts off the PSF's tails (faint outer light). For a Gaussian-like PSF, the bias grows exponentially with distance from the center: roughly $\exp(-r^2 / 2\sigma^2) / r$, where r is the window radius. To minimize this, use an optimal window size of about 3–5 times the PSF's full width at half maximum (FWHM); large enough to capture most light but small enough to exclude noise.

**Connection to Codebase Implementation**: Size filtering in the labeling step helps mitigate this; adaptive blocks from thresholding also aid in selecting appropriate regions.

#### 5. Calibration Alternative: Peak Detection

For crowded fields or overlapping PSFs, moment-based methods might merge stars, so an alternative is peak detection: find local maxima (brightest pixels) above the threshold, then filter by minimum separation (e.g., 5 pixels) to avoid duplicates. This is simpler but less precise for sub-pixel work, and is used within the simulation only as starting point for refinement when attempting to debug elements of the pipeline. 

**Connection to Codebase Implementation**: Implemented in `detect_stars_peak_method` in peak_detection.py, using a max_filter to identify peaks. This alternate option is primarily used in debugging methods where simplified peak detection is preferred for its reliability.

### Physical Insights and Limitations

This detection and centroiding process highlights how SNR drives accuracy: high-SNR stars (bright, many photons) yield centroids with low uncertainty (e.g., 0.1 pixels), enabling arcsecond-level pointing; low-SNR cases degrade to 0.5 pixels or more, risking mismatches. This is tied physically to PSFs, where sharper PSFs (from better optics) allow tighter centroids, but real sensors add biases like pixel non-uniformity.

Limitations include assumptions of isolated stars (crowding requires deblending techniques not covered here) and sensitivity to background variations (adaptive methods help but aren't perfect). For very faint stars, false positives rise, and finite windows introduce biases (up to 0.05 pixels for typical PSFs). Validation comes from comparing simulated centroids to known positions. Overall, this step transforms noisy pixels into precise locations, setting up the geometric conversion to bearing vectors in Section 5, where we'll turn these 2D spots into 3D vectors which house inner-angles: the essential invariant in our attitude-determination control system.

## Section 5: Bearing Vector Geometry

With precise star positions extracted from the noisy image via detection and centroiding, we now transform these 2D pixel coordinates into 3D directions known as bearing vectors that point from the camera toward the stars in space. Bearing vectors form the bridge between the observed image and the star matching algorithms in Section 6; any inaccuracies here, like those from camera calibration errors, could lead to mismatched stars and faulty attitude estimate. The extrusion of bearing vectors is essentially reversing the projection process from Section 1, but starting from the detector end; it turns flat image spots into unit vectors in the camera's reference frame, which can then be compared to known star directions in a celestial catalog. Intuitively, think of it as "unprojecting" a photo back into the 3D world; like estimating directions to landmarks in a picture based on your camera's lens specs.

We'll use geometry and camera models to derive these vectors, incorporating factors like focal length and distortions. The math relies on inverse perspective projection, but is a reasonably simple concept to imagine graphically; it's like tracing rays backward from the sensor through the lens to infinity, and then normalizing them to unit length. Note that this step is the only other apart from the radiometry chain in Section 2 which considers the actual camera parameters of the spacecraft. Therefore, by generalizing the Python modules for radiometry and bearing vector geometry to accept an arbitrary set of input camera parameters, we have expanded the simulation to work for all possible camera systems in existence (assuming the datasheet is known).

### Problem Statement

To enable star pattern matching, we must convert pixel centroids into accurate unit bearing vectors in the camera frame, accounting for intrinsic camera parameters and potential distortions. This transformation must propagate uncertainties from centroiding to predict overall angular errors, ensuring the vectors are reliable for downstream attitude determination.

### Critical Parameters

* **Focal length** $f$: The distance from the camera's optical center to the sensor plane, in physical units like microns or millimeters (e.g., 40 mm for a typical star tracker lens).
* **Principal point** $(c_x, c_y)$: The pixel coordinates where the optical axis intersects the detector, often near the center (e.g., (1024, 1024) for a 2048×2048 sensor).
* **Pixel pitch** $p$: The physical size of each pixel, in microns (e.g., 5.5 µm), which scales pixel positions to real-world distances.
* **Bearing vector components** $[\hat{v}_x, \hat{v}_y, \hat{v}_z]$: The normalized 3D direction vector, with length 1, pointing toward the star.
* **Angular error** $\Delta \theta_{bearing}$: The uncertainty in the vector's direction, in radians or arcseconds, arising from centroid errors and calibration inaccuracies.

These parameters tie hardware specifications to geometric accuracy, making the model adaptable to different camera designs.

### Step-by-Step Mathematical Derivation

Here we derive the bearing vector from pixel coordinates. This process inverts the forward projection from Section 1, starting with physical coordinates and ending with normalized 3D vectors, while incorporating error analysis for realism.

#### 1. Pixel to Focal Plane Coordinates

Pixel positions $(u, v)$ from centroiding are in discrete grid units, but we need physical distances on the focal plane. Convert them using the pixel pitch and principal point, which accounts for any offset in the optical center.

The formulas are:

$x_f = (u - c_x) \times p, \quad y_f = (v - c_y) \times p$

Here, $x_f$ and $y_f$ are in microns (or the same units as $f$). Note that the y-coordinate might be flipped (e.g., $y_f = (c_y - v) \times p$) depending on the image coordinate convention (top-left vs. bottom-left origin) to match the camera's orientation.

This step translates abstract pixels into measurable distances, like marking points on a physical grid.

**Connection to Codebase Implementation**: In `star_tracker_pipeline.calculate_bearing_vectors`, this is handled by a `pixel_to_physical_conversion` function that multiplies by pixel_pitch after subtracting the principal point.

#### 2. Pinhole Projection to 3D Vector

Using the pinhole model (assuming light rays converge at a point), we extend the focal plane coordinates into 3D by adding the focal length as the z-component. This creates a vector from the optical center to the point on the plane, which we then normalize to unit length for consistency (since stars are at infinite distance).

The bearing vector is:

$\mathbf{\hat{v}} = \frac{ [x_f, y_f, f] }{ \sqrt{x_f^2 + y_f^2 + f^2} }$

The components are $\hat{v}_x = x_f / d$, $\hat{v}_y = y_f / d$, $\hat{v}_z = f / d$, where $d = \sqrt{x_f^2 + y_f^2 + f^2}$ is the normalization denominator. This ensures $\|\mathbf{\hat{v}}\| = 1$, representing a pure direction.

Intuitively, longer focal lengths make the vector more "forward-pointing" (larger $\hat{v}_z$), corresponding to narrower fields of view.

**Connection to Codebase Implementation**: The `calculate_bearing_vectors` function computes normalized vectors as `[x_um / f_um, y_um / f_um, 1]` and then normalizes the array, effectively scaling by the focal length in microns.

<img title="" src="file:///H:/Star%20Tracker/White%20Paper/Images/bearing_vec.png" alt="bearing_vec" style="zoom:50%;" data-align="inline">

#### 3. Distortion Correction (if Applicable)

Real lenses introduce distortions (e.g., barrel or pincshion effects) that warp positions, so we correct $(x_f, y_f)$ before vector calculation. A simple radial distortion model uses a polynomial:

$x_{corr} = x_f (1 + k_1 r^2 + k_2 r^4), \quad y_{corr} = y_f (1 + k_1 r^2 + k_2 r^4)$

Where $r^2 = x_f^2 + y_f^2$, and $k_1, k_2$ are calibration coefficients (e.g., from lab measurements). Higher-order terms can be added for precision. After correction, use $(x_{corr}, y_{corr})$ in the vector formula.

This step compensates for optical imperfections, ensuring vectors align with the true sky geometry.

**Connection to Codebase Implementation**: While not in the base pipeline, `identify.py` includes a placeholder `apply_distortion_correction` function, indicating it's optional but can be integrated for more accurate simulations.

#### 4. Error Propagation

Uncertainties in centroids $\delta u$ and $\delta v$ (from Section 4) propagate to angular errors in the bearing vector. Approximating for small errors, the directional uncertainty is:

$\Delta \theta \approx \frac{p}{f} \sqrt{ (\delta u)^2 + (\delta v)^2 } \;\;\textrm{radians (convert to arcseconds by multiplying by 206265)}$ 

For example, with $f = 40$ mm (40,000 µm), $p = 5.5$ µm, and $\delta u = 0.2$ pixels, $\Delta \theta \approx 1$ arcsecond

Calibration errors in $f$ or $(c_x, c_y)$ add systematic biases, which can be modeled via sensitivity analysis (e.g., partial derivatives of $\mathbf{\hat{v}}$ with respect to parameters).

#### 5. Calibration Matrix (General Form)

Looking forward to cases using more advanced cameras, use an intrinsic matrix $K$ (3×3) that encapsulates $f$, $p$, and $(c_x, c_y)$. The bearing vector can be derived from the inverse projection, but the code assumes a simplified ideal pinhole model without skew or non-square pixels. The base implementation sticks to the ideal model for efficiency, but extensions could incorporate full $K$-matrix inversion.

### Physical Insights and Limitations

This geometric transformation reveals key trade-offs in star tracker design: a longer focal length $f$ improves angular resolution (smaller $\Delta \theta$ per pixel error), but it narrows the field of view, potentially capturing fewer stars for matching. Physically, it emphasizes calibration's importance. Temperature changes can expand/contract the lens, shifting $f$ by 0.1% and introducing errors up to arcminutes if unaccounted for.

Limitations include the ideal pinhole assumption (real systems often need distortion models, especially for wide fields >10°), and ignoring effects like thermal variations or mechanical misalignment. The model assumes square pixels and no skew, which holds for most CMOS sensors but not all. Validation involves comparing simulated vectors to known star positions, with errors typically under 1 arcsecond for well-calibrated systems. Overall, these bearing vectors provide the 3D foundation for pattern recognition in Section 6, where we'll match them to catalog stars to identify patterns and unlock the spacecraft's orientation.

<img title="" src="file:///C:/Users/Dr.%20J/iCloudDrive/SOE_Cloud/Star%20Tracker/Phase%201/BAST%20animation/output_12_0.png" alt="output_12_0" style="zoom:50%;">

## Section 6: Pattern Recognition Mathematics

Armed with 3D bearing vectors from the observed stars (derived in Section 5), we now face the task of identifying which stars they correspond to in a known celestial catalog. This is pattern recognition: matching the geometric arrangement of observed directions to precomputed star patterns by use of inner-angles that remain invariant under rotation. This is a critical lynchpin in the pipeline. Without correct identifications, the attitude estimation in Section 7 would use mismatched data, leading to wildly inaccurate orientations (like confusing the Big Dipper for Orion). In challenging conditions (e.g., noisy vectors or stray light creating false stars), the algorithm must be robust, rejecting ambiguities while confirming matches with high confidence. This stage is the first of the two canonical steps which define the cosmic puzzle of star tracking. Here we look for unique "signatures" through triangular shapes formed by star triplets, which house inner-angle triplets that are as distinctive as fingerprints for identification. By setting a threshold for how closely these inner-angle triplets must be to catalog values in order to be a "match", we must delicately balance wanting strict false-match rejection with the necessary tolerance demanded by the compounded error from previous stages.

Deriving the math behind this matching process draws from spherical geometry and combinatorial search. We calculate angles between vectors, form triplets, and search for near-identical ones in a database, scoring them to pick the best fit. The approach minimizes false positives through tolerances and verification, ensuring reliability for spacecraft navigation.

### Problem Statement

To determine spacecraft attitude, we must associate observed bearing vectors with catalog stars using rotation-invariant metrics like angular separations. The algorithm needs to handle measurement errors, potential false detections, and computational efficiency, while providing match confidence to flag ambiguities.

### Critical Parameters

Triplet matching key ideas:

* **Inter-star angles** $\theta_{ij}$: The angular separation between two stars i and j, in radians (typically 0.1° to 10° for field-of-view limits).
* **Angle tolerance** $\Delta \theta_{tol}$: The maximum allowable difference for a match, in radians (e.g., 0.01° to account for bearing errors from Section 5).
* **Match confidence** $C_{match}$: A score (0–1) indicating how well the patterns align, based on angular residuals.
* **Triangle geometric invariants**: Sorted angles (a ≤ b ≤ c) of star triplets, used as unique identifiers.
* **False positive probability** $P_{FP}$: The likelihood of erroneous matches, minimized by tight tolerances and verification steps.

These parameters balance sensitivity (catching true matches) with specificity (avoiding fakes), tunable via simulations.

### Step-by-Step Mathematical Derivation

This process leverages the angular preservation from Section 1, ensuring patterns are viewpoint-independent.

#### 1. Inter-Star Angular Distance

The foundation is computing the angle between two unit bearing vectors $\mathbf{v}_i$ and $\mathbf{v}_j$ (from observations or catalog). Since they're on the unit sphere, the great-circle distance is:

$\theta_{ij} = \cos^{-1}(\mathbf{v}_i \cdot \mathbf{v}_j)$

The dot product $\mathbf{v}_i \cdot \mathbf{v}_j$ ranges from -1 to 1; we clip it to avoid numerical issues. While it is not essential, understanding how dot products relate vectors to one-another in 3D space is very helpful for understanding how this equation leads to an inner-angle. The dot product $\mathbf{v}_i \cdot \mathbf{v}_j$ between two unit vectors (length = 1) gives a number between -1 and 1 that represents how *aligned* they are:

- If pointing in the exact same direction: dot product = 1
- If pointing in opposite directions: dot product = -1  
- If perpendicular (90°): dot product = 0

For two vectors at angle θ, the dot product equals $\cos(\theta)$ . So $\cos^{-1}(\mathbf{v}_i \cdot \mathbf{v}_j)$ simply undoes the cosine to give us the actual angle θ between the vectors.

This works because **unit vectors are just directions** - the dot product measures how much their directions overlap, and that overlap amount directly corresponds to the angle between them. For small angles (common in narrow fields), approximate $\theta \approx \sqrt{2(1 - \cos \theta)}$, but the full arccos is used for precision. This preserves the "shape" of constellations regardless of rotation.

**Connection to Codebase Implementation**: The `calculate_vector_angle` function in `match.py` computes `np.arccos(np.clip(dot, -1, 1))`, returning radians for pairwise comparisons.

#### 2. Triangle Geometric Invariants

To make matching robust, we form triplets of stars (three vectors) and compute their three inter-star angles $a, b, c$, sorted so a ≤ b ≤ c. This sorted tuple is rotation-invariant and acts as a hash for quick lookups. In a catalog, precompute these for every possible triplet within a field-of-view radius (e.g., 10°), associating each with the involved star IDs.

For efficiency, limit to nearby stars: for a central star, find neighbors within the max separation, compute angles, and store in a database. The number of triplets scales as $O(n^3)$ but is pruned by angular constraints.

**Connection to Codebase Implementation**: The `triplet_worker` in `catalog.py` uses astropy's SkyCoord.separation.rad to compute angles for all pairs, then forms and stores triplets for rapid querying.

#### 3. Matching Triplets

For an observed triplet with angles $\theta_1, \theta_2, \theta_3$ (sorted), search the catalog for entries where each angle differs by less than $\Delta \theta_{tol}$:

$|\theta_k - \theta_{cat,k}| < \Delta \theta_{tol} \quad \forall k = 1,2,3$

If multiple candidates, score them with confidence:

$C_{match} = 1 - \frac{\sum |\delta \theta_k|}{3 \Delta \theta_{tol}}$

Higher C means a better fit. Select the highest-confidence match, or verify by checking if the implied rotation aligns additional stars.

**Connection to Codebase Implementation**: `find_triplet_match` in `match.py` calculates differences and checks if all are below tolerance, then computes a score based on averages.

<img src="file:///C:/Users/Dr.%20J/iCloudDrive/SOE_Cloud/Star%20Tracker/Phase%201/BAST%20animation/output_16_0.png" title="" alt="output_16_0" style="zoom:67%;">

#### 4. Pyramid Verification

After finding initial triplet matches, pyramid verification provides a crucial confirmation step that significantly reduces false positives and increases matching reliability. This technique takes the attitude solution implied by a matched triplet and predicts where additional stars should appear in the observed field-of-view. If these predictions match actual detected stars within tolerance, it confirms the match is genuine rather than coincidental.

The mathematical framework is as follows: Given a matched triplet that implies rotation matrix $R_{match}$, we can predict the expected positions of all other catalog stars in the field:

$\mathbf{v}_{pred,i} = R_{match} \mathbf{v}_{catalog,i}$

For each predicted star position, we find the closest observed star and check if the angular separation falls within a verification tolerance (typically 1-2 arcseconds). Represented with set-theory notation:

$|\theta_{pred,i} - \theta_{obs,j}| < \Delta \theta_{verif} \quad \forall i \in \text{additional stars}$

**Verification Metrics**:

- **Match Percentage**: Fraction of predicted stars that have a corresponding observed star within tolerance
- **Verification Score**: Weighted average of angular residuals for matched predictions
- **Confidence Boost**: Matches that pass verification (e.g., >70% match percentage) receive significantly higher confidence scores

**Benefits and Implementation**:

1. **False Positive Rejection**: Eliminates coincidental triplet matches by requiring consistency across the entire star field
2. **Ambiguity Resolution**: When multiple triplets suggest different attitudes, verification scores help identify the correct one
3. **Robustness**: Works even with moderate centroiding errors, as long as the overall pattern is consistent
4. **Computational Efficiency**: Can be parallelized across multiple candidate matches for rapid verification

This verification step transforms star tracking from a two-step process (find + match) to a three-step process (find + match + verify), dramatically improving reliability in real-world applications with noise, false detections, and crowded fields.

**Connection to Codebase Implementation**: While not currently in the baseline simulation for the full star tracker pipeline (it is virtually useless when working with synthetic catalogs as the inner-angle triplets are guaranteed to match accurately), `pyramid_verify` is an extension of the `match` function to include a verification phase after initial triplet matching that predicts additional star positions and compute verification metrics. This module has been used successfully in previous models of the simulation. Future versions of the codebase will include this verification step as a default step.

![output_15_0](C:\Users\Dr.%20J\iCloudDrive\SOE_Cloud\Star%20Tracker\Phase%201\BAST%20animation\output_15_0.png)

#### 5. Probabilistic Matching and False Alarms

To quantify reliability, estimate the probability of a random match. The "volume" of tolerance in angle space is roughly $(\Delta \theta_{tol})^3$, and with N catalog triplets, $P_{FP} \approx N \times (\Delta \theta_{tol} / \pi)^3$ (normalized to the sphere's range). Tighten $\Delta \theta_{tol}$ based on bearing errors from Section 5 (e.g., set to 3σ of angular uncertainty) to keep $P_{FP} < 10^{-6}$.

For verification, compute residuals after a tentative match and reject if they exceed a threshold.

**Connection to Codebase Implementation**: While not directly computed, `match.py` uses a min_confidence threshold to filter, implicitly controlling false positives.

#### 6. Combinatorial Optimization for Multi-Star Matching

With multiple observed stars, generate all possible triplets and match them combinatorially. To assign unique catalog IDs without duplicates, use a greedy approach: match the most confident triplet first, mark those stars as used, and repeat. For optimal assignment, employ the Hungarian algorithm (explained shortly) on a cost matrix where costs are angular differences, minimizing the total mismatch.

The cost for assigning observed star i to catalog star j could be $c_{ij} = |\theta_{obs} - \theta_{cat}|$ for pairwise, but extended to full patterns.

The Hungarian algorithm solves the "assignment problem": imagine a group of workers and a group of tasks, where each worker needs to be assigned to exactly one task to minimize total cost.

In star tracking context:

- Workers = observed stars from your image
- Tasks = catalog stars from your database  
- Cost= how poorly each observed star matches each catalog star (angular difference)
- Goal = Find the assignment that minimizes total mismatch across all pairings

Apart from raw computation requirements, the Hungarian algorithm is a universally superior approach. The greedy algorithm at its core is just, "match the best pair first, then the next best from remaining stars, and repeat". This method can quite easily get "trapped" if the first good match prevents a globally superior matching arrangement. The Hungarian approach considers **all** possible assignments simultaneously and finds the one with the lowest total cost. It's essentially **global optimization** rather than **local optimization**, ensuring the best possible star matching arrangement rather than just a "good enough" one that happened to be found first.

**Connection to Codebase Implementation**: The main `match` function iterates combinations greedily, tracking used indices for bijectivity. For advanced cases, `bijective_matching.py` uses `scipy.optimize.linear_sum_assignment` (Hungarian) on a cost matrix.

#### 7. Confidence Metrics and Rejection

After matching, compute overall confidence from averaged residuals or by fitting a rotation (preview of Section 7) and checking alignment. Propagate errors: bearing uncertainty $\Delta \theta_{bearing}$ from Section 5 adds to $\Delta \theta_{tol}$, increasing $P_{FP}$ in noisy conditions.

**Connection to Codebase Implementation**: `score_match` in `match.py` averages normalized differences; matches are rejected if below a threshold.

### Physical Insights and Limitations

This pattern recognition leverages the spherical geometry of the sky: angles are invariant, making matches independent of the spacecraft's unknown attitude: a key insight for "lost-in-space" scenarios where no prior orientation is assumed. Physically, it highlights catalog density trade-offs: denser catalogs (more faint stars) provide more unique patterns but increase computational load and ambiguity risk.

Limitations include combinatorial explosion for many stars (e.g., >20 requires efficient pruning), assumptions of no identical patterns (rare but possible in sparse fields), and sensitivity to false stars (mitigated by confidence scoring). For wide fields, spherical distortions might need haversine formulas over dot products. Validation involves simulating random views and measuring match success rates. Overall, successful matching unlocks the final step in Section 7, where identified stars yield the precise attitude quaternion.



## Section 7: Optimal Attitude Estimation

With matched star identifications from pattern recognition (Section 6), we can finally determine the spacecraft's precise attitude by finding the best rotation that aligns the observed bearing vectors with their catalog counterparts. Solving Wahba's problem, which seeks the optimal rotation matrix minimizing the misalignment between two sets of vectors, is the culmination of all of the work simulating the star tracker pipeline up to now. In noisy or uncertain matches, the algorithm must provide not just the estimate but also confidence bounds, like a statistical "error bar" on the orientation. It's like fitting pieces to a puzzle; we compute a quaternion (a compact way to represent rotations, discussed in Appendix B) that twists the observed stars to overlap the known ones, using eigenvalues to find the sweet spot.

Here we derive the QUEST (QUaternion ESTimator) algorithm, a efficient solution to Wahba's problem, incorporating weights and our Monte Carlo methods for uncertainty. The math involves linear algebra and optimization, starting with building a matrix from vector pairs, then extracting the eigenvector for the quaternion, and finally creating perturbations to gauge reliability.

### Problem Statement

Given pairs of observed bearing vectors and matched catalog vectors, compute the optimal rotation (as a quaternion or matrix) that aligns them, while quantifying uncertainty from upstream errors like centroiding noise. The solution must be computationally efficient for real-time use and robust to outliers or mismatched pairs.

### Critical Parameters

First, define the pivotal variables upfront to anchor the derivations:

* **Davenport K-matrix eigenvalues** $\lambda_i$: The roots of the characteristic equation, where the maximum $\lambda_{max}$ indicates the optimal solution's quality.
* **Optimal quaternion** $\hat{q} = [q_0, q_1, q_2, q_3]$: A unit vector representing the rotation, with $q_0$ as the scalar part.
* **Attitude uncertainty** $\sigma_q$: The standard deviation of quaternion components, in dimensionless units (convertible to angular errors).
* **Residual errors** $\epsilon_i$: The angular misalignments after rotation, in radians, used for validation.
* **Confidence bounds**: Statistical intervals (e.g., 95%) on the attitude, derived from Monte Carlo simulations.

These parameters ensure the estimation is not just a point value but a probabilistic assessment, critical for mission risk analysis.

### Step-by-Step Mathematical Derivation

This process stars with deriving the QUEST method from Wahba's loss function to quaternion extraction, highlighting the optimization elements used in the software implementation. This builds on the matched vectors from Section 6, treating them as noisy measurements to solve for the attitude rotation.

#### 1. Wahba's Loss Function and Attitude Profile Matrix

Wahba's problem minimizes the loss: $L(R) = \sum_{i=1}^n w_i \| \mathbf{b}_i - R \mathbf{r}_i \|^2$, where $\mathbf{b}_i$ are observed unit vectors, $\mathbf{r}_i$ are catalog references, $w_i$ are weights (e.g., inversely proportional to uncertainty), and R is the rotation matrix (orthogonal, det(R)=1).

Expanding, this simplifies to maximizing $G(R) = \sum w_i (\mathbf{b}_i \cdot R \mathbf{r}_i)$, since minimizing L is equivalent.

The attitude profile matrix is $B = \sum w_i \mathbf{b}_i \mathbf{r}_i^T$—a 3×3 matrix capturing pairwise alignments. Then, decompose the symmetric portion of  $S = B + B^T - \textbf{tr}(B) I$, and $Z = [B_{23} - B_{32}, B_{31} - B_{13}, B_{12} - B_{21}]^T$ (skew-symmetric vector).

The Davenport K-matrix is:

$K = \begin{bmatrix} \mathbf{tr}(B) & Z^T \\ Z & S \end{bmatrix}$

This 4×4 matrix encodes the problem, turning the optimization into an eigenvalue hunt.

**Connection to Codebase Implementation**: In `build_davenport_matrix` from resolve.py, it computes $B$ as the sum of weighted outer products `(np.outer(b, r) * w)`, then constructs $S$, $Z$, and $K$ as described.

##### An Aside: Layman's Explanation of Wahba's Problem Setup:

Wahba's problem is arguably the single most critical element in the star tracking pipeline. It is also one of the most complicated to grasp intuitively. To bridge this gap, let's take a moment to understand what is actually happening here in plain-english.

**Setting up:** At this point, there are two sets of directions pointing towards the matched stars: where the camera observes the stars, and where the catalog says those same stars should be. The goal of Wahba's problem is to solve for the magic rotational quaternion which optimally links these two views.

**Step 1 - The "Badness" Measure (Loss Function):**
Think of $L(R)$ as a "total error score." For each matched star pair:

- Take a guess at a rotational quaternion R
- Rotate the catalog direction using  guess rotation R
- Measure how far off it is from where the camera actually observed it 
- Square that distance error (so big mistakes hurt more than small ones)
- Add them all up, giving more importance to reliable stars (the weights $w_i$)

**Step 2 - Mathematical Trick (L to G conversion):**
Instead of minimizing total badness, we can maximize total "goodness" as mathematically, the loss function $L(R)$ can be inverted this way. This is just the mathematical way of saying something like "minimizing your golf score is the same as maximizing your distance under par." (This of course assumes one to be an elite golfer capable of shooting regularly under par, but alas.) 

**Step 3 - The Summary Matrix (B):**
Rather than deal with all the individual star pairs, we create a 3×3 "summary matrix" B that captures all the alignment information in one compact form. Think of B as a "relationship summary." For each star pair, you multiply the observed direction by the catalog direction (as vectors). When you add all these up, B captures the overall "correlation pattern" between what you observed and what the catalog predicts.

* If your observations perfectly match the catalog (no rotation needed), B would be the identity matrix
* If there's a rotation, B encodes which way and how much everything needs to turn

**Step 4 - Matrix Decomposition (S and Z):**
Every matrix can be split into symmetric and skew-symmetric parts (like splitting a number into even and odd components). This isn't just mathematical housekeeping - it has physical meaning:

* **Symmetric part (S)**: Captures the "stretching and squashing" aspects of the transformation
* **Skew-symmetric part (Z)**: Captures the "pure rotation" aspects of the transformation

Think of it like analyzing dance moves: you can separate any complex motion into "size changes" (symmetric) and "spinning/rotating" (skew-symmetric) components. For our problem, we only want the rotation part.

**Step 5 - The Magic K-Matrix:**
We reorganize all our alignment information into a special 4×4 matrix K. Here's where it gets clever. Rotations can be represented by either:

* 3×3 rotation matrices (9 numbers with 6 constraints = 3 degrees of freedom)
* 4-element quaternions (4 numbers with 1 constraint = 3 degrees of freedom)

The specific 4×4 format of K is designed so that when you find its largest eigenvalue and corresponding eigenvector, that eigenvector IS the quaternion representing the optimal rotation! The K-matrix arrangement mathematically "encodes" the optimization problem in quaternion space. Instead of searching through all possible 3×3 rotation matrices, we can directly find the optimal 4-element quaternion using standard eigenvalue techniques. This specific format allows us to find the optimal rotation automatically, almost like having a calculator that solves the whole problem for us in one step.



#### 2. QUEST Eigenvalue Solution

The optimal quaternion $\hat{q}$ is the eigenvector of K corresponding to its maximum eigenvalue $\lambda_{max}$. This comes from rewriting G(R) in quaternion form and maximizing under the unit norm constraint.

To find $\lambda_{max}$, solve the characteristic equation det(K - λ I) = 0, but numerically, compute all eigenvalues and pick the largest real one. The associated eigenvector is $\hat{q}$, normalized to unit length.

For efficiency, QUEST approximates $\lambda_{max}$ iteratively, but the code uses direct eigendecomposition for precision. The code IS doing the QUEST algorithm, it's just using a more accurate implementation than the original.

**Some Historical Context:**
The QUEST algorithm was developed in the 1970s-80s when computers were much slower. Back then, computing all eigenvalues of even a 4×4 matrix was computationally expensive, so QUEST used clever iterative approximations to find just the largest eigenvalue without computing the others.

**What BAST Does:**
The simulation code uses a custom "QUEST with direct eigendecomposition" method where it builds the same Davenport K-matrix and extracts the same eigenvector, but uses modern linear algebra libraries to compute ALL eigenvalues at once, then picks the largest one. So is this really still QUEST? Yes! the mathematical framework is identical:

- Same K-matrix construction
- Same eigenvalue-based solution
- Same quaternion extraction

The only difference is *how* we find the largest eigenvalue (direct vs. iterative approximation). For simulation and capability research, direct eigendecomposition is just objectively better: more accurate, simpler code, negligible speed difference for 4×4 matrices. However, when the time comes for the real-time flight hardware, there should be some discussion as to the choice between iterative or direct eigenvalue calculation since iterative QUEST can be slightly faster for repeated computations, as well as having a more predictable computational load (not to mention the heritage and track record of the algorithm). The bottom line is that the code implements the core QUEST mathematics with higher precision than the original. It's absolutely usable in practice, and many modern star trackers use this approach. The choice between direct vs. iterative depends on the specific speed vs. accuracy requirements as we get further into the design timeline.

**Connection to Codebase Implementation**: The `quest_algorithm` in resolve.py calls np.linalg.eig(K), selects the index of the maximum real eigenvalue, extracts the eigenvector, and normalizes it to get the quaternion.

#### 3. Quaternion to Rotation Matrix

With the optimal rotation quaternion in-hand, all that is left for attitude determination is to convert the quaternion to a matrix for applying rotations.

For $\mathbf{q} = [w, x, y, z]$ (with w = q_0):

$$
R = \begin{bmatrix}1 - 2y^2 - 2z^2 & 2xy - 2wz & 2xz + 2wy \\2xy + 2wz & 1 - 2x^2 - 2z^2 & 2yz - 2wx \\2xz - 2wy & 2yz + 2wx & 1 - 2x^2 - 2y^2\end{bmatrix}
$$

This R transforms reference vectors to the observed frame (or vice versa, depending on convention).

**Connection to Codebase Implementation**: `quaternion_to_matrix` in resolve.py implements this exact formula, handling the components directly.

#### 4. Monte Carlo Uncertainty Estimation

To quantify errors, perturb the bearing vectors with Gaussian noise reflecting centroid uncertainty (from Section 4, propagated via Section 5). For each trial: add noise ~ $N(0, σ_{bearing})$, recompute matches if needed, run QUEST, and collect quaternions.

Compute the mean quaternion (via averaging or more advanced methods like averaging on the SO(3) manifold) and covariance. The angular uncertainty is approximately $\Delta \theta \approx 2 \|\sigma_{q_{vec}}\|$ radians, where $\sigma_{q_{vec}}$ is the std of the vector part [q1,q2,q3].

Converge by monitoring rolling std until below a tolerance (e.g., 1e-6), ensuring stable estimates.

**Connection to Codebase Implementation**: In `determine_attitude` from monte_carlo_quest.py, it adds noise to bearings, batches QUEST calls, computes stats like mean and std of quaternions, and checks convergence on rolling statistics.

#### 5. Residuals and Confidence Bounds

After estimation, compute residuals: $\epsilon_i = \|\mathbf{b}_i - R \mathbf{r}_i\|$ (or angular: arccos(dot product)). High residuals indicate poor fits or outliers.

Confidence from $\lambda_{max}$ (closer to sum($w_i$) means better alignment) or Monte Carlo: e.g., 95% bounds from the quaternion distribution. Reject if residuals exceed a threshold (e.g., 3σ).

### Physical Insights and Limitations

The QUEST algorithm reveals the information-theoretic limits of attitude determination: with more matched stars (higher n), uncertainty drops as $\frac{1}{\sqrt(n)}$, like averaging measurements; weights allow prioritizing brighter, less noisy stars for better precision. Physically, it underscores the need for diverse star patterns since collinear stars provide poor constraints on all axes, leading to degenerate solutions.

Limitations include assumptions of Gaussian errors (real mismatches might be non-Gaussian) and equal handling of all pairs (advanced versions incorporate adaptive weights based on SNR from Section 2). For few pairs (n<3), solutions are underconstrained; Monte Carlo adds computational overhead but is vital for realism. Validation compares estimated attitudes to ground truth in simulations, often achieving arcsecond accuracy. As the pipeline's finale, this step delivers the actionable orientation, closing the loop from photons to navigation.

*[Image in Development]: A schematic illustration would be helpful here, depicting observed vectors on one side, catalog vectors on the other, with curved arrows showing the optimal rotation alignment via QUEST. Include eigenvalue spectra, a quaternion icon, and error ellipses from Monte Carlo trials to visualize uncertainty, making the optimization and probabilistic aspects more tangible for readers.]*

# Appendices

## Appendix A: Glossary

To complement the mathematical foundations detailed in this white paper, we provide a comprehensive glossary of key terms, concepts, and abbreviations used throughout the document. Terms are listed alphabetically, with definitions derived directly from the context of the star tracker radiometry simulation pipeline. Where applicable, cross-references to relevant sections are included for deeper exploration. This glossary aims to clarify technical jargon, ensuring accessibility for readers from diverse backgrounds in aerospace engineering, optics, and computational simulation.

* **Adaptive Thresholding**: A detection technique that dynamically sets intensity cutoffs based on local image statistics (mean and standard deviation) to account for varying backgrounds and noise levels. Used in star detection to create binary masks. (See Section 4)
* **Apparent Magnitude (m)**: A logarithmic measure of a star's brightness as observed from Earth, where lower values indicate brighter stars. It relates to photon flux via the Pogson scale, forming the basis for radiometric calculations. (See Section 2)
* **Attitude Estimation**: The process of determining a spacecraft's orientation in space by finding the optimal rotation that aligns observed star vectors with catalog references. Solved using algorithms like QUEST. (See Section 7)
* **Attitude Profile Matrix (B)**: A 3×3 matrix constructed from weighted outer products of observed and reference vectors, used in Wahba's problem to encode alignment information for attitude computation. (See Section 7)
* **Bearing Vector**: A normalized 3D unit vector representing the direction from the camera to a star in the camera's reference frame, derived from pixel coordinates via inverse projection. Essential for pattern matching. (See Section 5)
* **Centroiding**: The calculation of a star's sub-pixel position in an image using weighted moments (center of mass) of pixel intensities, accounting for noise and PSF shape. Achieves high precision like 0.1–0.3 pixels. (See Section 4)
* **Connected Components Labeling**: An image processing step that groups adjacent pixels in a binary mask into distinct regions (blobs), filtered by criteria like area to identify potential stars. (See Section 4)
* **Cramér-Rao Bound**: A theoretical lower limit on the uncertainty of position estimates (e.g., centroids), given by $\sigma_{centroid} \geq \frac{\sigma_{psf}}{\sqrt{N_{ph}}}$, where $\sigma_{psf}$ is PSF width and $N_{ph}$ is photon count. (See Sections 3 and 4)
* **Dark Current (I_d)**: Thermal generation of electrons in the detector, contributing Poisson-distributed noise (variance $I_d t$) over exposure time t. A key factor in total noise variance. (See Sections 2 and 3)
* **Davenport K-Matrix**: A 4×4 matrix derived from the attitude profile matrix, used in the QUEST algorithm. Its maximum eigenvalue and corresponding eigenvector yield the optimal quaternion. (See Section 7)
* **Detection Threshold (T_detect)**: The minimum intensity value for flagging pixels as part of a star, often set adaptively (e.g., $T_b = \mu_b + k \sigma_b$) to balance sensitivity and false positives. (See Section 4)
* **Euler Angles (φ, θ, ψ)**: A set of three angles (roll, pitch, yaw) used to represent spacecraft attitude via sequential rotations around axes. Converted to rotation matrices for coordinate transformations. (See Section 1)
* **False Positive Probability (P_FP)**: The likelihood of erroneous pattern matches, estimated as approximately the number of catalog triplets times $(\Delta \theta_{tol} / \pi)^3$. Minimized through tight tolerances. (See Section 6)
* **Focal Length (f)**: The distance from the camera's optical center to the sensor plane, crucial for projecting 3D vectors to 2D coordinates and vice versa. Affects angular resolution. (See Sections 1 and 5)
* **Full Well Capacity (N_FW)**: The maximum number of electrons a pixel can hold before saturation, limiting the signal for bright stars. (See Section 2)
* **Gnomonic Projection**: A tangent-plane projection used for mapping celestial coordinates to a flat detector, assuming a pinhole model. Legacy mode for boresight-centered scenes. (See Section 1)
* **Hungarian Algorithm**: An optimization method for solving assignment problems, such as bijective matching of observed stars to catalog entries by minimizing a cost matrix of angular differences. (See Section 6)
* **Inter-Star Angle (θ_ij)**: The angular separation between two stars, computed as $\theta_{ij} = \cos^{-1}(\mathbf{v}_i \cdot \mathbf{v}_j)$ for unit vectors. Rotation-invariant and key for pattern matching. (See Section 6)
* **Match Confidence (C_match)**: A score (typically 0–1) quantifying how well observed and catalog patterns align, based on angular residuals (e.g., $C = 1 - \frac{\sum |\delta \theta|}{\Delta \theta_{tol}}$). (See Section 6)
* **Moment-Based Centroiding**: A method to compute star positions using intensity-weighted averages, akin to finding the center of mass. Includes uncertainty estimates from noise variance. (See Section 4)
* **Monte Carlo Simulation**: A statistical technique involving repeated trials with random perturbations (e.g., noise on bearings) to estimate attitude uncertainty and convergence of quaternion distributions. (See Sections 3 and 7)
* **Pattern Recognition**: The process of matching observed star patterns (e.g., triplets) to a celestial catalog using invariant features like sorted inter-star angles, enabling star identification. (See Section 6)
* **Photon Flux (Φ)**: The rate of photons arriving from a star per unit area, derived from apparent magnitude and scaled by aperture and transmission. Basis for signal calculation. (See Section 2)
* **Pinhole Camera Model**: An ideal projection model assuming light rays pass through a single point, used to map 3D directions to 2D focal plane coordinates (and inversely for bearing vectors). (See Sections 1 and 5)
* **Pixel Pitch (p)**: The physical size of a detector pixel (e.g., 5.5 µm), used to convert focal plane distances to pixel coordinates. Influences angular error propagation. (See Sections 1 and 5)
* **Point Spread Function (PSF)**: A 2D intensity distribution describing how light from a point source (star) spreads on the detector due to optics. Normalized as a probability map for photon simulation. (See Section 3)
* **Poisson Distribution**: A statistical model for random photon arrivals, where the count k in a pixel follows $Pr(k) = \frac{\lambda^k e^{-\lambda}}{k!}$, with mean and variance λ. Models shot noise. (See Section 3)
* **Principal Point (c_x, c_y)**: The pixel coordinates where the optical axis intersects the detector, used in converting pixel positions to physical focal plane coordinates. (See Section 5)
* **Quantum Efficiency (QE)**: The fraction of incident photons converted to photoelectrons in the detector, typically expressed as electrons per photon. Affects signal strength. (See Section 2)
* **Quaternion (q)**: A four-component vector [q_0, q_1, q_2, q_3] representing 3D rotations compactly, avoiding gimbal lock. Used for attitude and converted to rotation matrices. (See Sections 1 and 7)
* **QUEST (QUaternion ESTimator)**: An algorithm solving Wahba's problem by finding the eigenvector of the Davenport K-matrix corresponding to the maximum eigenvalue, yielding the optimal attitude quaternion. (See Section 7)
* **Read Noise (σ_r^2)**: Gaussian-distributed electronic noise introduced during image readout, contributing to total variance independent of signal. (See Sections 2 and 3)
* **Rotation Matrix (R)**: A 3×3 orthogonal matrix representing transformations between reference frames (e.g., inertial to camera). Preserves angles and derived from Euler angles or quaternions. (See Section 1)
* **Shot Noise**: Random fluctuations in photon counts following Poisson statistics, with variance equal to the mean number of photons. Dominant for bright sources. (See Sections 2 and 3)
* **Signal-to-Noise Ratio (SNR)**: The ratio of signal electrons to the square root of total noise variance, used to predict detection reliability (e.g., SNR > 5 for thresholds). (See Section 2)
* **Triangle Geometric Invariants**: Sorted inter-star angles (a ≤ b ≤ c) of a star triplet, used as rotation-invariant identifiers for efficient catalog matching. (See Section 6)
* **Wahba's Problem**: An optimization problem to find the rotation matrix minimizing the weighted sum of squared differences between observed and rotated reference vectors. (See Section 7)

This glossary is compiled from the core concepts in Sections 1–7. For terms not listed here or requiring more scholarly depth (e.g., advanced derivations of the Cramér-Rao bound in imaging contexts), ample academic resources are available for exploration. Inquire with Dr. J for more information. 



## Appendix B: Quaternions and the QUEST Algorithm

While most engineers are deeply familiar with the concepts of vectors and matrices, encountering quaternions for the first time can seem like an abstract mathematical curiosity. Developing intuition for a set of four mysterious numbers that somehow encode 3D rotations more elegantly than the familiar Euler angles will usually fall into the camp of "a job for someone else". Yet quaternions are fundamental to modern spacecraft attitude determination, offering computational efficiency and avoiding the singularities that plague other rotation representations. In the context of the discussed star tracker simulation pipeline, quaternions appear in two critical places: representing spacecraft attitude during scene generation (Section 1) and as the final output of the QUEST attitude estimation algorithm (Section 7). This appendix provides both the mathematical foundations and intuitive understanding necessary to work confidently with quaternions in star tracking applications.

Think of quaternions as a more sophisticated coordinate system for rotations; just as Cartesian coordinates elegantly describe positions in 3D space, quaternions provide a natural language for describing orientations.

### What Are Quaternions? The Geometric Intuition

A quaternion is fundamentally a mathematical object that encodes a rotation in 3D space using four components: one scalar and three vector components. While this might seem like overkill (why four numbers for three-dimensional rotations?), the extra dimension provides remarkable advantages for computation and avoids the singularities inherent in three-parameter representations.

The key insight is that quaternions represent rotations using **Euler's rotation theorem**: any rotation in 3D space can be described as a single rotation by some angle θ about some axis **n̂**. A quaternion encodes both this axis and angle in a compact, computationally friendly form.

#### Mathematical Definition

A quaternion **q** consists of four components, typically written as:

$\mathbf{q} = [q_0, q_1, q_2, q_3] = [w, x, y, z]$

where:

* $q_0$ (or $w$) is the **scalar part**: $q_0 = \cos(\theta/2)$
* $q_1, q_2, q_3$ (or $x, y, z$) form the **vector part**: $[q_1, q_2, q_3] = \sin(\theta/2) \hat{\mathbf{n}}$

Here, $\theta$ is the rotation angle and $\hat{\mathbf{n}} = [n_x, n_y, n_z]$ is the unit vector defining the rotation axis.

The crucial constraint is that quaternions representing physical rotations must be **unit quaternions**:

$|\mathbf{q}|^2 = q_0^2 + q_1^2 + q_2^2 + q_3^2 = 1$

This normalization ensures that the quaternion represents a pure rotation without scaling.

#### Intuitive Understanding Through Examples

Using concrete examples, we can see more clearly how these calcualtions apply to well-known rotations:

**Identity Rotation** (no rotation):

* $\mathbf{q} = [1, 0, 0, 0]$
* $\theta = 0°$, so $\cos(0°) = 1$ and $\sin(0°) = 0$

**90° Rotation about the Z-axis**:

* $\mathbf{q} = [0.707, 0, 0, 0.707]$
* $\theta = 90°$, $\hat{\mathbf{n}} = [0, 0, 1]$ (along z-axis)
* $q_0 = \cos(45°) = 0.707$, $q_3 = \sin(45°) \times 1 = 0.707$
* When rotation is about a single axis, only one of the **vector parts** of the quaternion will be non-zero

**180° Rotation about the X-axis**:

* $\mathbf{q} = [0, 1, 0, 0]$
* $\theta = 180°$, $\hat{\mathbf{n}} = [1, 0, 0]$
* $q_0 = \cos(90°) = 0$, $q_1 = \sin(90°) \times 1 = 1$
  
  

### Why Quaternions? Advantages Over Euler Angles

While Euler angles (roll, pitch, yaw) provide an intuitive way to think about rotations, they suffer from several limitations that make quaternions superior for computational applications:

#### 1. Gimbal Lock Avoidance

Euler angles suffer from **gimbal lock**: configurations where one degree of freedom is lost. This occurs when the middle rotation (typically pitch) approaches ±90°, causing the first and third rotations to become aligned. In this configuration, infinite combinations of roll and yaw angles can produce the same final orientation, leading to computational instabilities and discontinuities.

Quaternions, having four parameters for three degrees of freedom, provide redundancy that eliminates gimbal lock entirely. Every possible 3D rotation has a smooth, continuous quaternion representation.

#### 2. Computational Efficiency

Converting quaternions to rotation matrices requires only 12 multiplications and 12 additions, compared to the trigonometric function evaluations needed for Euler angles. For spacecraft applications processing attitude updates at high rates, this efficiency is crucial.

#### 3. Smooth Interpolation

Quaternions enable smooth interpolation between orientations using **Spherical Linear Interpolation (SLERP)**. This is essential for attitude control systems that need to compute intermediate orientations during maneuvers. Euler angle interpolation can produce unexpected rotational paths due to singularities.

#### 4. Composition Simplicity

Combining rotations with quaternions involves simple quaternion multiplication (detailed below), whereas combining Euler angle rotations requires matrix multiplications or complex trigonometric calculations.

### Quaternion Mathematics for Star Tracking

#### Quaternion Multiplication

The composition of two rotations represented by quaternions **p** and **q** is given by their product **pq**:

$\mathbf{p} \otimes \mathbf{q} = \begin{pmatrix} p_0 q_0 - p_1 q_1 - p_2 q_2 - p_3 q_3 \\p_0 q_1 + p_1 q_0 + p_2 q_3 - p_3 q_2 \\p_0 q_2 - p_1 q_3 + p_2 q_0 + p_3 q_1 \\p_0 q_3 + p_1 q_2 - p_2 q_1 + p_3 q_0 \end{pmatrix}$

This operation is **non-commutative**: $\mathbf{p} \otimes \mathbf{q} \neq \mathbf{q} \otimes \mathbf{p}$ in general, reflecting the fact that the order of rotations matters in 3D space.

#### Quaternion Conjugate and Inverse

The **conjugate** of a quaternion reverses the rotation:  $\mathbf{q}^* = [q_0, -q_1, -q_2, -q_3]$

For unit quaternions, the conjugate equals the **inverse**:  $\mathbf{q}^{-1} = \mathbf{q}^*  \iff |\mathbf{q}|=1 $

This property is computationally advantageous as computing the inverse rotation requires only negating three components.

#### Converting Quaternions to Rotation Matrices

The rotation matrix corresponding to quaternion $\mathbf{q} = [q_0, q_1, q_2, q_3]$ is:

$\begin{pmatrix} q_0^2 + q_1^2 - q_2^2 - q_3^2 & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\2(q_1 q_2 + q_0 q_3) & q_0^2 - q_1^2 + q_2^2 - q_3^2 & 2(q_2 q_3 - q_0 q_1) \\2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & q_0^2 - q_1^2 - q_2^2 + q_3^2 \end{pmatrix}$

This matrix is orthogonal ($\mathbf{R}^T \mathbf{R} = \mathbf{I}$) with determinant +1, ensuring it represents a proper rotation without scaling or reflection.

#### Rotating Vectors with Quaternions

To rotate a 3D vector **v** = $[v_x, v_y, v_z]$ by quaternion **q**, we use the quaternion rotation formula:

$\mathbf{v}' = \mathbf{q} \otimes [0, \mathbf{v}] \otimes \mathbf{q}^*$

Here, $[0, \mathbf{v}]$ represents the vector as a pure quaternion (zero scalar part). While this triple quaternion multiplication might seem complex, it's actually more efficient than computing and applying the full rotation matrix for single vector operations.

### Quaternions in Star Tracker Scene Generation

In Section 1 of our pipeline, quaternions specify the spacecraft's attitude during scene generation. The process works as follows:

#### 1. Attitude Parameterization

The spacecraft's orientation is specified as a quaternion **q** representing the rotation from an inertial reference frame to the spacecraft body frame. This quaternion encodes how the star tracker camera is oriented relative to the celestial sphere.

#### 2. Star Vector Transformation

For each star in the catalog with inertial unit vector $\mathbf{v}_I$, we compute the corresponding vector in the camera frame:

$\mathbf{v}_C = \mathbf{R}(\mathbf{q}) \mathbf{v}_I$

where $\mathbf{R}(\mathbf{q})$ is the rotation matrix derived from the attitude quaternion.

#### 3. Angle Preservation

The orthogonal nature of rotation matrices ensures that angular relationships between stars are preserved:

$\cos(\theta_{ij}) = \mathbf{v}_{I,i} \cdot \mathbf{v}_{I,j} = \mathbf{v}_{C,i} \cdot \mathbf{v}_{C,j}$

This angle preservation is crucial for the pattern recognition algorithms in Section 6.

The culmination of our star tracker pipeline (Section 7) produces an optimal attitude quaternion using the QUEST algorithm. Understanding this process requires several key concepts:

### The Wahba Problem Formulation

QUEST solves Wahba's problem: find the rotation that best aligns observed bearing vectors $\mathbf{b}_i$ with reference catalog vectors $\mathbf{r}_i$. In quaternion form, this becomes finding **q** that maximizes:

$g(\mathbf{q}) = \sum_{i=1}^n w_i (\mathbf{b}_i \cdot \mathbf{R}(\mathbf{q}) \mathbf{r}_i)$

where $w_i$ are weights based on measurement uncertainties.

#### The Davenport K-Matrix

The optimization is reformulated as an eigenvalue problem using the 4×4 Davenport K-matrix:

$\mathbf{K} = \begin{bmatrix}\sigma & \mathbf{z}^T \\\mathbf{z} & \mathbf{S}\end{bmatrix}$

where:

* $\sigma = \text{trace}(\mathbf{B})$ with $\mathbf{B} = \sum w_i \mathbf{b}_i \mathbf{r}_i^T$
* $\mathbf{z}$ is a 3×1 vector from the skew-symmetric part of $\mathbf{B}$
* $\mathbf{S}$ is the 3×3 symmetric part of $\mathbf{B} + \mathbf{B}^T$

#### Quaternion Extraction

The optimal quaternion is the unit eigenvector corresponding to $\mathbf{K}$'s largest eigenvalue:

$\mathbf{K} \mathbf{q}_{opt} = \lambda_{max} \mathbf{q}_{opt}$

This eigenvector directly provides the four quaternion components representing the spacecraft's attitude.

#### Quaternion Normalization

Numerical errors can cause quaternions to drift from unit length during computations. Regular renormalization is essential:

$\mathbf{q}_{normalized} = \frac{\mathbf{q}}{|\mathbf{q}|}$

In star tracker applications, normalization should be performed after each quaternion operation to maintain rotation validity.

#### Double Coverage

Unit quaternions exhibit **double coverage**: both **q** and **-q** represent the same physical rotation. This ambiguity can cause issues in attitude estimation if not handled properly. The QUEST algorithm typically resolves this by choosing the quaternion with positive scalar component.

#### Singularity-Free Interpolation

When interpolating between two attitude quaternions **q₁** and **q₂**, use SLERP (Spherical Linear Interpolation):

$\mathbf{q}(t) = \frac{\sin((1-t)\Omega)}{\sin(\Omega)}\mathbf{q}_1 + \frac{\sin(t\Omega)}{\sin(\Omega)}\mathbf{q}_2$

where $\Omega = \cos^{-1}(|\mathbf{q}_1 \cdot \mathbf{q}_2|)$ is the angle between quaternions, and $t \in [0,1]$ is the interpolation parameter.

#### Error Propagation

Attitude uncertainty in quaternion form can be propagated using the covariance matrix of quaternion components. For small errors, the angular uncertainty is approximately:

$\Delta\theta \approx 2\sqrt{q_1^2 + q_2^2 + q_3^2} = 2|\mathbf{q}_{vector}|$

This provides a scalar measure of attitude uncertainty suitable for mission planning and error budgets. The reason this cannot be used as a rule for our own estimations is due to the pesky "small errors" term. The threshold for a "small" error can vary dependent on situation, but generally for star trackers, this will only hold true when error < 10 arcsecs.


### Conclusion

Quaternions provide an elegant, computationally efficient, and singularity-free representation for spacecraft attitude in star tracker applications. While initially abstract, their mathematical properties offer significant advantages over Euler angles for both scene generation and attitude estimation algorithms. The key insights to remember are:

1. **Quaternions encode axis-angle rotations** in a four-component representation that avoids gimbal lock
2. **Unit normalization is crucial** for representing physical rotations
3. **Quaternion multiplication composes rotations** efficiently without trigonometric functions
4. **The QUEST algorithm naturally outputs quaternions** from the eigenvalue solution
5. **Double coverage requires careful handling** in sign conventions

By understanding these principles, engineers can confidently implement and debug quaternion-based attitude systems, leveraging their mathematical elegance for robust spacecraft navigation. The investment in learning quaternion mathematics pays dividends in cleaner code, fewer edge cases, and more reliable attitude determination.



*"Ludwig Boltzmann, who spent much of his life studying statistical mechanics, died in 1906, by his own hand. Paul Ehrenfest, carrying on the work, died similarly in 1933. Now it is our turn to study statistical mechanics. Perhaps it will be wise to approach the subject cautiously".*

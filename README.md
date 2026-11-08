
<img width="1536" height="1024" alt="SARUMAN" src="https://github.com/user-attachments/assets/69e4b0a9-a0ba-4653-8870-2837f1c105ba" />

# SARUMAN

**Spatial Activity Reconstruction Using Machine-learning Algorithms for Nuclear Waste**

SARUMAN is a Python research code for three-dimensional radionuclide activity reconstruction in radioactive waste drums using Angular Segmented Gamma Scanning (ASGS), X-ray computed tomography (CT), and Bayesian inference.

The workflow combines CT-derived attenuation estimates, a machine-learning detector-efficiency surrogate, gamma spectrometry measurements, and stochastic variational inference to reconstruct radionuclide activity distributions and quantify their uncertainty.

## Research Software and Scope of Applicability

SARUMAN was developed as part of doctoral research into Bayesian machine-learning methods for radioactive waste characterization.

**The current implementation is tailored to a specific experimental characterization system.** In particular, the CT-based attenuation model, the trained detector-efficiency surrogate, and several geometrical and acquisition parameters correspond to the equipment and measurement configuration used during development.

Consequently, SARUMAN should not be considered a general-purpose, ready-to-use characterization tool for arbitrary measurement systems. Applying it to different equipment or acquisition configurations may require new calibration data, surrogate retraining, geometrical adaptations, and experimental validation.

Nevertheless, the underlying methodology and many software components are intended to be transferable. Researchers and developers are encouraged to reuse, extend, and adapt the code for other measurement systems, inverse problems, and broader scientific applications, subject to the applicable license and institutional permissions.

**SARUMAN is provided as a research framework and reference implementation, rather than a universally validated characterization solution.**

## Features

- CT processing and three-dimensional voxel mesh generation.
- Ray tracing with reusable geometry caching.
- CT-based gamma-ray attenuation estimation using CT2LAC.
- Detector full-energy peak efficiency estimation using a trained neural-network surrogate.
- Joint Bayesian reconstruction of multiple radionuclides.
- Stochastic variational inference using NumPyro and JAX.
- Posterior uncertainty quantification and posterior predictive checks.
- HDF5-based workflow for separating data preparation from reconstruction.
- Optional diagnostic exports for CT, attenuation, and detector efficiency.

## Installation

SARUMAN requires Python 3.9 or later and a compatible scientific Python environment.

From the project directory:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install --no-build-isolation -e .
```

The application requires TensorFlow/Keras, PyTorch, JAX, NumPyro, and other scientific Python dependencies declared in `pyproject.toml`.

GPU-enabled JAX installations may require additional installation steps depending on the operating system, CUDA version, and hardware.

**Important:** The bundled surrogate models and calibration data were developed for the original experimental setup. Their validity must not be assumed for other instruments.

## Configuration

SARUMAN provides default settings in `src/default_config.yaml`.

Users can create a `config.yaml` in their working directory to override selected parameters. A template is provided in `config.example.yaml`.

For example:

```yaml
path_calculation:
  max_workers: 4

debug:
  enabled: false
```

Configuration priority:

1. Command-line arguments, where supported.
2. User-provided YAML configuration.
3. Bundled default configuration.

An alternative configuration file can be specified using `--config`.

The ray-tracing cache is stored in the user's home directory by default:

```text
~/.cache/saruman/scenarios/
```

The cache can be regenerated when necessary.

## Usage

SARUMAN exposes four commands.

### Prepare

Process CT and ASGS measurements and construct the forward-model data package:

```bash
saruman prepare \
  --ct scan.mhd \
  --spectra ags/ \
  --output drum.h5
```

### View

Inspect a prepared HDF5 package:

```bash
saruman view --input drum.h5
```

### Reconstruct

Run joint Bayesian inference for selected radionuclides:

```bash
saruman reconstruct \
  --input drum.h5 \
  --isotopes Eu-152 Cs-137 \
  --output results/
```

### Run

Execute preparation followed by reconstruction:

```bash
saruman run \
  --ct scan.mhd \
  --spectra ags/ \
  --isotopes Eu-152 Cs-137 \
  --package drum.h5 \
  --output results/
```

Use `saruman <command> --help` for additional options.

## Outputs

The preparation stage produces an HDF5 file containing the processed measurements, reconstructed mesh, detector-efficiency estimates, and transmission information.

The reconstruction stage produces activity estimates, posterior samples, diagnostic figures, and posterior predictive checks.

Optional debug outputs can be enabled through the YAML configuration.

## Experimental Dataset and Reproducibility

The experimental dataset associated with SARUMAN is publicly available on Zenodo:

**[SARUMAN Experimental Mockup Dataset](https://doi.org/10.5281/zenodo.24062457)**

DOI: `10.5281/zenodo.24062457`

The dataset comprises six homogeneous (HOM-1 to HOM-6) and three heterogeneous
(HET-1 to HET-3) radioactive waste drum mockups. Each configuration includes
X-ray CT reconstructions and Angular Segmented Gamma Scanning (ASGS) measurements.

The experimental data are distributed separately from the source code to
facilitate independent access, reproducibility, and reuse.

**Reference publication:**

https://doi.org/10.1016/j.anucene.2026.112850

Researchers are encouraged to consult the publication for the experimental
setup, reconstruction methodology, and interpretation of the results before
using the dataset.

## Scientific Limitations

The inverse problem may be underdetermined, particularly when limited gamma-ray measurements are available.

Reconstruction accuracy and uncertainty estimates depend on the quality of the CT-based attenuation information, detector-efficiency model, measurement uncertainties, and prior assumptions.

Uncertainty estimates should not be interpreted as guarantees of accuracy under model misspecification or outside the validated experimental domain.

Independent validation is required before applying the methodology to new experimental configurations or operational characterization tasks.



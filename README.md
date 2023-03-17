# Probabilistic Domain Adapation in Biomedical Image Segmentation

## Installation

We recommend to use conda to install all required dependencies. To set up a suitable conda environment and install the additional functionality neeeded follow these steps:
- If you don't have conda installed follow these [installation instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html#anaconda-or-miniconda)
- Create a new conda environment with the necessary requirements, using the environment file we provide:
  ```
  conda env create -f environment_gpu.yaml -n <ENV_NAME>
  ```
- Then activate the environment and install our `prob_utils` library:
  ```
  conda activate <ENV_NAME>
  pip install -e .
  ```

Note that you may need to adapt the CUDA version in the `environment_gpu.yaml` file to match your system. 
We provide `environment_cpu.yaml` as an alternative if you don't have access to a CUDA compatible GPU.

Now you can run all scripts for model training, prediction and evaluation in the `<ENV_NAME>` environment.

## Experiments

We provide the code for all three domain adaptation experiments from the paper, all scripts for the respective experiments are in the respective folders:

### LiveCELL

Available training frameworks :
- UNet Source
- PUNet Source
- PUNet Target (optional - with Consensus Weighting/Masking)
- PUNet Mean-Teacher - Separate Training (optional - with Consensus Weighting/Masking)
- PUNet FixMatch - Separate Training (optional - with Consensus Weighting/Masking)
- PUNet Mean-Teacher - Joint Training (optional - with Consensus Weighting/Masking)
- PUNet FixMatch - Joint Training (optional - with Consensus Weighting/Masking)

### MitoEM

Available training frameworks :
- UNet Source
- PUNet Source
- PUNet Mean-Teacher - Separate Training (optional - with Consensus Weighting/Masking)

### Lung X-Ray

Available training frameworks :
- UNet Source
- PUNet Source
- PUNet Mean-Teacher - Separate Training (optional - with Consensus Weighting/Masking)
- PUNet Mean-Teacher - Joint Training (optional - with Consensus Weighting/Masking)

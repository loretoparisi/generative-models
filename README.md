# Generative Models by Stability AI

![sample1](assets/000.jpg)

## Docker

### Dependencies
Installed dependencies
```bash
PyWavelets-1.5.0 aiohttp-3.9.0 aiosignal-1.3.1 altair-5.1.2 antlr4-python3-runtime-4.9.3 appdirs-1.4.4 async-timeout-4.0.3 black-23.7.0 blinker-1.7.0 braceexpand-0.1.7 cachetools-5.3.2 chardet-5.1.0 clip-1.0 cmake-3.27.7 contourpy-1.2.0 cycler-0.12.1 docker-pycreds-0.4.0 einops-0.7.0 fairscale-0.4.13 fire-0.5.0 fonttools-4.45.1 frozenlist-1.4.0 ftfy-6.1.3 gitdb-4.0.11 gitpython-3.1.40 huggingface-hub-0.19.4 importlib-metadata-6.8.0 invisible-watermark-0.2.0 jsonschema-4.20.0 jsonschema-specifications-2023.11.1 kiwisolver-1.4.5 kornia-0.6.9 lightning-utilities-0.10.0 lit-17.0.5 markdown-it-py-3.0.0 matplotlib-3.8.2 mdurl-0.1.2 multidict-6.0.4 mypy-extensions-1.0.0 natsort-8.4.0 ninja-1.11.1.1 nvidia-cublas-cu11-11.10.3.66 nvidia-cuda-cupti-cu11-11.7.101 nvidia-cuda-nvrtc-cu11-11.7.99 nvidia-cuda-runtime-cu11-11.7.99 nvidia-cudnn-cu11-8.5.0.96 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.2.10.91 nvidia-cusolver-cu11-11.4.0.1 nvidia-cusparse-cu11-11.7.4.91 nvidia-nccl-cu11-2.14.3 nvidia-nvtx-cu11-11.7.91 omegaconf-2.3.0 open-clip-torch-2.23.0 opencv-python-4.6.0.66 pandas-2.1.3 pathspec-0.11.2 platformdirs-4.0.0 protobuf-3.20.3 pudb-2023.1 pyarrow-14.0.1 pydeck-0.8.1b0 pyparsing-3.1.1 python-dateutil-2.8.2 pytorch-lightning-2.0.1 pyyaml-6.0.1 referencing-0.31.0 regex-2023.10.3 rich-13.7.0 rpds-py-0.13.1 safetensors-0.4.0 scipy-1.11.4 sentencepiece-0.1.99 sentry-sdk-1.37.1 setproctitle-1.3.3 smmap-5.0.1 streamlit-1.28.2 tenacity-8.2.3 tensorboardx-2.6 termcolor-2.3.0 timm-0.9.12 tokenizers-0.12.1 toml-0.10.2 torch-2.0.1 torchaudio-2.0.2 torchdata-0.6.1 torchmetrics-1.2.0 torchvision-0.15.2 tornado-6.3.3 transformers-4.19.1 triton-2.0.0 tzdata-2023.3 tzlocal-5.2 urwid-2.2.3 urwid_readline-0.13 validators-0.22.0 wandb-0.16.0 watchdog-3.0.0 wcwidth-0.2.12 webdataset-0.2.79 xformers-0.0.22 yarl-1.9.3 zipp-3.17.0
```

## News

**November 21, 2023**

- We are releasing Stable Video Diffusion, an image-to-video model, for research purposes:
    - [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid): This model was trained to generate 14
      frames at resolution 576x1024 given a context frame of the same size. 
      We use the standard image encoder from SD 2.1, but replace the decoder with a temporally-aware `deflickering decoder`.
    - [SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt): Same architecture as `SVD` but finetuned
      for 25 frame generation.
    - We provide a streamlit demo `scripts/demo/video_sampling.py` and a standalone python script `scripts/sampling/simple_video_sample.py` for inference of both models.
    - Alongside the model, we release a [technical report](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets).

  ![tile](assets/tile.gif)

**July 26, 2023**

- We are releasing two new open models with a
  permissive [`CreativeML Open RAIL++-M` license](model_licenses/LICENSE-SDXL1.0) (see [Inference](#inference) for file
  hashes):
    - [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0): An improved version
      over `SDXL-base-0.9`.
    - [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0): An improved version
      over `SDXL-refiner-0.9`.

![sample2](assets/001_with_eval.png)

**July 4, 2023**

- A technical report on SDXL is now available [here](https://arxiv.org/abs/2307.01952).

**June 22, 2023**

- We are releasing two new diffusion models for research purposes:
    - `SDXL-base-0.9`: The base model was trained on a variety of aspect ratios on images with resolution 1024^2. The
      base model uses [OpenCLIP-ViT/G](https://github.com/mlfoundations/open_clip)
      and [CLIP-ViT/L](https://github.com/openai/CLIP/tree/main) for text encoding whereas the refiner model only uses
      the OpenCLIP model.
    - `SDXL-refiner-0.9`: The refiner has been trained to denoise small noise levels of high quality data and as such is
      not expected to work as a text-to-image model; instead, it should only be used as an image-to-image model.

If you would like to access these models for your research, please apply using one of the following links:
[SDXL-0.9-Base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9),
and [SDXL-0.9-Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9).
This means that you can apply for any of the two links - and if you are granted - you can access both.
Please log in to your Hugging Face Account with your organization email to request access.
**We plan to do a full release soon (July).**

## The codebase

### General Philosophy

Modularity is king. This repo implements a config-driven approach where we build and combine submodules by
calling `instantiate_from_config()` on objects defined in yaml configs. See `configs/` for many examples.

### Changelog from the old `ldm` codebase

For training, we use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), but it should be easy to use other
training wrappers around the base modules. The core diffusion model class (formerly `LatentDiffusion`,
now `DiffusionEngine`) has been cleaned up:

- No more extensive subclassing! We now handle all types of conditioning inputs (vectors, sequences and spatial
  conditionings, and all combinations thereof) in a single class: `GeneralConditioner`,
  see `sgm/modules/encoders/modules.py`.
- We separate guiders (such as classifier-free guidance, see `sgm/modules/diffusionmodules/guiders.py`) from the
  samplers (`sgm/modules/diffusionmodules/sampling.py`), and the samplers are independent of the model.
- We adopt the ["denoiser framework"](https://arxiv.org/abs/2206.00364) for both training and inference (most notable
  change is probably now the option to train continuous time models):
    * Discrete times models (denoisers) are simply a special case of continuous time models (denoisers);
      see `sgm/modules/diffusionmodules/denoiser.py`.
    * The following features are now independent: weighting of the diffusion loss
      function (`sgm/modules/diffusionmodules/denoiser_weighting.py`), preconditioning of the
      network (`sgm/modules/diffusionmodules/denoiser_scaling.py`), and sampling of noise levels during
      training (`sgm/modules/diffusionmodules/sigma_sampling.py`).
- Autoencoding models have also been cleaned up.

## Installation:

<a name="installation"></a>

#### 1. Clone the repo

```shell
git clone git@github.com:Stability-AI/generative-models.git
cd generative-models
```

#### 2. Setting up the virtualenv

This is assuming you have navigated to the `generative-models` root after cloning it.

**NOTE:** This is tested under `python3.10`. For other python versions, you might encounter version conflicts.

**PyTorch 2.0**

```shell
# install required packages from pypi
python3 -m venv .pt2
source .pt2/bin/activate
pip3 install -r requirements/pt2.txt
```

#### 3. Install `sgm`

```shell
pip3 install .
```

#### 4. Install `sdata` for training

```shell
pip3 install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
```

## Packaging

This repository uses PEP 517 compliant packaging using [Hatch](https://hatch.pypa.io/latest/).

To build a distributable wheel, install `hatch` and run `hatch build`
(specifying `-t wheel` will skip building a sdist, which is not necessary).

```
pip install hatch
hatch build -t wheel
```

You will find the built package in `dist/`. You can install the wheel with `pip install dist/*.whl`.

Note that the package does **not** currently specify dependencies; you will need to install the required packages,
depending on your use case and PyTorch version, manually.

## Inference

We provide a [streamlit](https://streamlit.io/) demo for text-to-image and image-to-image sampling
in `scripts/demo/sampling.py`.
We provide file hashes for the complete file as well as for only the saved tensors in the file (
see [Model Spec](https://github.com/Stability-AI/ModelSpec) for a script to evaluate that).
The following models are currently supported:

- [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  ```
  File Hash (sha256): 31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b
  Tensordata Hash (sha256): 0xd7a9105a900fd52748f20725fe52fe52b507fd36bee4fc107b1550a26e6ee1d7
  ```
- [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
  ```
  File Hash (sha256): 7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f
  Tensordata Hash (sha256): 0x1a77d21bebc4b4de78c474a90cb74dc0d2217caf4061971dbfa75ad406b75d81
  ```
- [SDXL-base-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9)
- [SDXL-refiner-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9)
- [SD-2.1-512](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.safetensors)
- [SD-2.1-768](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors)

**Weights for SDXL**:

**SDXL-1.0:**
The weights of SDXL-1.0 are available (subject to
a [`CreativeML Open RAIL++-M` license](model_licenses/LICENSE-SDXL1.0)) here:

- base model: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/
- refiner model: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/

**SDXL-0.9:**
The weights of SDXL-0.9 are available and subject to a [research license](model_licenses/LICENSE-SDXL0.9).
If you would like to access these models for your research, please apply using one of the following links:
[SDXL-base-0.9 model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9),
and [SDXL-refiner-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9).
This means that you can apply for any of the two links - and if you are granted - you can access both.
Please log in to your Hugging Face Account with your organization email to request access.

After obtaining the weights, place them into `checkpoints/`.
Next, start the demo using

```
streamlit run scripts/demo/sampling.py --server.port <your_port>
```

### Invisible Watermark Detection

Images generated with our code use the
[invisible-watermark](https://github.com/ShieldMnt/invisible-watermark/)
library to embed an invisible watermark into the model output. We also provide
a script to easily detect that watermark. Please note that this watermark is
not the same as in previous Stable Diffusion 1.x/2.x versions.

To run the script you need to either have a working installation as above or
try an _experimental_ import using only a minimal amount of packages:

```bash
python -m venv .detect
source .detect/bin/activate

pip install "numpy>=1.17" "PyWavelets>=1.1.1" "opencv-python>=4.1.0.25"
pip install --no-deps invisible-watermark
```

To run the script you need to have a working installation as above. The script
is then useable in the following ways (don't forget to activate your
virtual environment beforehand, e.g. `source .pt1/bin/activate`):

```bash
# test a single file
python scripts/demo/detect.py <your filename here>
# test multiple files at once
python scripts/demo/detect.py <filename 1> <filename 2> ... <filename n>
# test all files in a specific folder
python scripts/demo/detect.py <your folder name here>/*
```

## Training:

We are providing example training configs in `configs/example_training`. To launch a training, run

```
python main.py --base configs/<config1.yaml> configs/<config2.yaml>
```

where configs are merged from left to right (later configs overwrite the same values).
This can be used to combine model, training and data configs. However, all of them can also be
defined in a single config. For example, to run a class-conditional pixel-based diffusion model training on MNIST,
run

```bash
python main.py --base configs/example_training/toy/mnist_cond.yaml
```

**NOTE 1:** Using the non-toy-dataset
configs `configs/example_training/imagenet-f8_cond.yaml`, `configs/example_training/txt2img-clipl.yaml`
and `configs/example_training/txt2img-clipl-legacy-ucg-training.yaml` for training will require edits depending on the
used dataset (which is expected to stored in tar-file in
the [webdataset-format](https://github.com/webdataset/webdataset)). To find the parts which have to be adapted, search
for comments containing `USER:` in the respective config.

**NOTE 2:** This repository supports both `pytorch1.13` and `pytorch2`for training generative models. However for
autoencoder training as e.g. in `configs/example_training/autoencoder/kl-f4/imagenet-attnfree-logvar.yaml`,
only `pytorch1.13` is supported.

**NOTE 3:** Training latent generative models (as e.g. in `configs/example_training/imagenet-f8_cond.yaml`) requires
retrieving the checkpoint from [Hugging Face](https://huggingface.co/stabilityai/sdxl-vae/tree/main) and replacing
the `CKPT_PATH` placeholder in [this line](configs/example_training/imagenet-f8_cond.yaml#81). The same is to be done
for the provided text-to-image configs.

### Building New Diffusion Models

#### Conditioner

The `GeneralConditioner` is configured through the `conditioner_config`. Its only attribute is `emb_models`, a list of
different embedders (all inherited from `AbstractEmbModel`) that are used to condition the generative model.
All embedders should define whether or not they are trainable (`is_trainable`, default `False`), a classifier-free
guidance dropout rate is used (`ucg_rate`, default `0`), and an input key (`input_key`), for example, `txt` for
text-conditioning or `cls` for class-conditioning.
When computing conditionings, the embedder will get `batch[input_key]` as input.
We currently support two to four dimensional conditionings and conditionings of different embedders are concatenated
appropriately.
Note that the order of the embedders in the `conditioner_config` is important.

#### Network

The neural network is set through the `network_config`. This used to be called `unet_config`, which is not general
enough as we plan to experiment with transformer-based diffusion backbones.

#### Loss

The loss is configured through `loss_config`. For standard diffusion model training, you will have to
set `sigma_sampler_config`.

#### Sampler config

As discussed above, the sampler is independent of the model. In the `sampler_config`, we set the type of numerical
solver, number of steps, type of discretization, as well as, for example, guidance wrappers for classifier-free
guidance.

### Dataset Handling

For large scale training we recommend using the data pipelines from
our [data pipelines](https://github.com/Stability-AI/datapipelines) project. The project is contained in the requirement
and automatically included when following the steps from the [Installation section](#installation).
Small map-style datasets should be defined here in the repository (e.g., MNIST, CIFAR-10, ...), and return a dict of
data keys/values,
e.g.,

```python
example = {"jpg": x,  # this is a tensor -1...1 chw
           "txt": "a beautiful image"}
```

where we expect images in -1...1, channel-first format.

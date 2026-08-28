# DeepViTAS

Official implementation of **DeepViTAS**, a deep learning architecture for deforestation detection using satellite imagery.

## Overview

DeepViTAS is a deep learning model developed for deforestation detection in satellite imagery. The proposed architecture combines a modified YOLOv8 backbone, a Vision Transformer (ViT), and a Spatial Attention mechanism for semantic segmentation.

## Main Components

* Modified YOLOv8 backbone
* Vision Transformer (ViT)
* Spatial Attention module
* Dice-Focal hybrid loss

## Requirements

The implementation was developed using Python 3.x.

The required Python libraries are listed in `requirements.txt`.

## Installation

Clone this repository:

```bash
git clone https://github.com/GiBenvenuto/DeepViTAS.git
cd DeepViTAS
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Configure the dataset path and training parameters in the corresponding Python files.

To train the model and see the results:

```bash
python train.py
```

## Model Configuration

The Vision Transformer embedding dimension can be configured using different values, including:

* `embed_dim = 128`
* `embed_dim = 256`
* `embed_dim = 512`

Other model and training hyperparameters can also be configured according to the experimental setup, including the input image size, patch size, batch size, learning rate, number of epochs, and early stopping patience.

The hyperparameters used for each experiment are defined in the corresponding Python files and can be adjusted according to the user's requirements.

## Dataset


The datasets used in the experiments are not included in this repository. The datasets must be downloaded separately and placed inside the Datasets directory.

The datasets used in the experiments are:

* Datasets 1 and 2: Available at https://zenodo.org/records/4498086.
* Dataset 3: Available at https://github.com/GiBenvenuto/Precision-Meets-Speed-Attention-Encoder-Decoder-Network-for-Deforestation-Segmentation.

Please refer to the corresponding publications and data sources for information about dataset access.

## Hardware Requirements

Training requirements depend on the input image size, batch size, and model configuration.

For the experiments using 512 × 512 images, a computer with at least 16 GB of system RAM is recommended.

GPU acceleration is recommended for training.

## Reproducibility

The experimental settings reported in the associated publication should be used to reproduce the reported results.

## Citation

If you use this code in your research, please cite the associated publication:



## License

This project is distributed under the license specified in the `LICENSE` file.

## Contact

For questions or issues, please open an issue in this repository.
Or contact giovana.a.benvenuto@unesp.br

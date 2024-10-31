# Mini-Gen: Minimum Reproducible Generative Models

## Environment setup

Run the following command for creating the environment:

```bash
conda create -n mini-gen python=3.11 pip
conda activate mini-gen
pip install -r requirements.txt
pre-commit install
```

## Training

Please check and edit the corresponding configuration in `configs/` folder.

For visualization with Tensorboard:

```bash
conda activate mini-gen
tensorboard --logdir logs/dcgan
```

then go to `localhost:6006`

# ML architectures: an example

Three ways to learn a map from a few parameters to a curve of many values, with a mock data set and an example notebook. The curve is a
mock version of the (2,2) phase of an eccentric binary, but the code does not depend on that: any smooth function of a few parameters,
sampled on a fixed grid, works the same way. No trained model and no real data are included; the mock data set is made on the fly.

| Design | What it does |
|---|---|
| 1. SVD + network | An SVD compresses the curves to a few coefficients; a flat network (6 layers of one width, LayerNorm + GELU in every layer) maps the parameters to them. |
| 2. SVD + autoencoder | Stage 1: an autoencoder learns to compress the coefficients to a small latent vector and back. Stage 2: a parameter network learns that latent vector from the parameters, and the pretrained decoder turns it into coefficients. |
| 3. Average | The mean of two models: (A) the same two-stage autoencoder on the data itself, and (B) the network of design 1. |

The layers, their initialisation, the optimiser (Adam, learning rate 1e-3 cut by 10 on a plateau of the validation loss, down to 1e-8),
the L1 loss and the two-stage autoencoder training are those of our full model. The sizes and epoch counts are smaller, so the notebook
runs on a laptop CPU in about 2 minutes; the full sizes are given in the comments.

## Files

- `example.ipynb`: the example. It makes the data, fits the SVD basis, trains all three designs and compares them on test examples. It is saved with its outputs.
- `mock_data.py`: the mock data: one curve of 1024 values per parameter set (q, e0, zeta0), a smooth rise with small oscillations that fade towards t = 0.
- `architectures.py`: the networks (`FlatDirectNetwork` for designs 1 and 3B, `Encoder`/`Decoder`/`FunnelAutoencoder` and `ParameterToDecoder` for the autoencoders).
- `training.py`: the SVD basis (`fit_svd_basis`), the scaling of the coefficients, and the training loops (`train_direct`, `train_two_stage_autoencoder`).
- `figures/`: one architecture diagram per design, shown in the notebook, with the sizes used there.

## Running it

Python 3.10 or newer with `numpy`, `torch`, `matplotlib` and Jupyter:

    pip install numpy torch matplotlib jupyter
    jupyter notebook example.ipynb

Keep the files and the `figures/` folder together; the notebook imports the three `.py` files from its own folder. A GPU is used when
torch finds one.

## Using your own data

Replace the data cell with your own `theta` (parameters) and `phase` array (one row per example, one column per sample) and an input
transform like `network_inputs` (inputs of order one; periodic parameters as sine and cosine). Everything after that cell works
unchanged. For real data, raise `EPOCHS` (we use up to 5000), `PATIENCE` (125) and the network sizes.

## Which design is best?

That depends on the data; the notebook is a demonstration, not a benchmark. On our real data (1024 SVD coefficients, 10,613 samples per
curve, about 900,000 examples) the autoencoder on the coefficients (design 2) came close to a direct network on average but was worse at
the edges of the parameter range, and the average of two models (design 3) was about twice as far off.

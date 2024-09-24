# MutationNBVAE
A mini-project adapting sparse overdispersed VAEs to mutational data.

To run this yourself, install requirements with 

``` pip install -r requirements.txt```

Download the demo breast cancer dataset from [cBioPortal](https://www.cbioportal.org/study/summary?id=breast_msk_2018) and make sure that the `RAW_DATA` path in the training notebook points to its location.

Code for the variational autoencoder in `nbvae.py`, based on [this arxiv paper](https://arxiv.org/pdf/1905.00616) and loosely adapted from a tensorflow implementation [here](https://github.com/ethanhezhao/NBVAE/tree/master). `mutation_dataset.py` contains a PyTorch dataset class for working with VCF data and mutation counts.

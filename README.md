# 🦐 Gamba: Scalable Attention Based Graph-to-Sequence Modeling

Group project for the course _Deep Learning (263-3210-00L)_ at ETH Zurich.
___

The pdf for the whole report/paper will be there

### Installation:
This is a short introduciton how to run things:

* Create a conda env with `CONDA_OVERRIDE_CUDA=12.4 conda create --name [env name] python=3.12.7`
* Install torch and its dependencies `conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia`
* Install PyG `pip install torch_geometric`
* Install PyG dependencies `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html`
* Install other dependencies `pip install wandb pandas tabulate`


* To run gamba install transformers `pip install transformers`

* Then also install `pip install ray[tune]` and `pip install pynndescent`

To run wandb logging, add your API key to `wandb.key`. (Wandb logging is currently unimplemented)

### Running Experiments:
To run an experiment, use the following command:

```bash
python start.py --config data/configs/sample_config.json
```

#### Using Google Colab:

The experiments can also be run on Google Colab GPUs using the `google_colab.ipynb` notebook. 

The notebook requires that this repository is installed into `/content/drive/MyDrive/DL_Project/DL_Project/`.

### Societal impact:
Too many gambas might collapse the ecosystem.


This is a PyTorch implementation

Dependencies
----------------------
- pytorch 1.9.0
- dgl 0.8.1
- sympy
- argparse
- sklearn

How to run
--------------------------------
The T-Finance and T-Social datasets developed in the paper are on [google drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing). Download and unzip it into `dataset`.

The Yelp and Amazon datasets will be automatically downloaded from the Internet. 


Run main_without_subsample.py to train the whole graph.  e.g.,
	python main_without_subsample.py --dataset amazon --train_ratio 0.01 --hid_dim 64 --order 2 --homo 1 --epoch 100 --run 3 --n_layers 1 --num_heads 1 --seed 5 --layer_norm True --residual True --ffn 2


Run main_subsample.py with subsampling mode. e.g.,
	python main_subsample.py --dataset tfinance --train_ratio 0.4 --hid_dim 128 --order 2 --homo 1 --epoch 100 --run 3 --n_layers 1 --num_heads 3 --seed 10 --layer_norm True --residual True --ffn 2
	
# Code & data accompanying the paper "Energy Transformer".

if you find this code useful, please cite:

@article{hoover2023energy,
  title={Energy Transformer},
  author={Hoover, Benjamin and Liang, Yuchen and Pham, Bao and Panda, Rameswar and Strobelt, Hendrik and Chau, Duen Horng and Zaki, Mohammed J and Krotov, Dmitry},
  journal={arXiv preprint arXiv:2302.07253},
  year={2023}
}

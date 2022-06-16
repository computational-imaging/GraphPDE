## Learning to Solve PDE-constrained Inverse Problems with Graph Networks | ICML 2022
### [Project Page](https://cyanzhao42.github.io/LearnInverseProblem) | [Video](https://www.youtube.com/watch?v=ov0jxa4xHGU) | [Paper](https://arxiv.org/abs/2206.00711)
Official PyTorch implementation.<br>[Learning to Solve PDE-constrained Inverse Problems with Graph Networks](http://www.computationalimaging.org/publications/graphpde/)<br>
[Qingqing Zhao]()\*,
[David B. Lindell](https://davidlindell.com),
[Gordon Wetzstein](https://computationalimaging.org)<br>
Stanford University <br>
<img src='pipline.jpg'/>
## Set up environment
To setup a conda environment use these commands
```
conda env create -f environment.yml
conda activate gnn
```
We also need to install pytorch and pytorch-geometric with following commands:
```
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric
```
## Solve Inverse Problem
Dataset and pretrained model and validation samples can be download [here](https://drive.google.com/file/d/1FnOYE2TThb6QCgDIkaBw0CGQN7D0scbu/view?usp=sharing). Unzip the data.zip folder in the root directory.

Now you can solve invere problem with 2D wave equation with the following commands.
```
# with prior
python InverseProblem/experiment_scripts/run_gnn.py  --config InverseProblem/config/density_gnn_p.ini
python InverseProblem/experiment_scripts/run_gnn.py  --config InverseProblem/config/init_state_gnn_p.ini 
# without prior
python InverseProblem/experiment_scripts/run_gnn.py  --config InverseProblem/config/density_gnn_np.ini 
python InverseProblem/experiment_scripts/run_gnn.py  --config InverseProblem/config/init_state_gnn_np.ini 
```
You may also run the notebooks for a quick demo and visualization.
| File | Description |
| --- | ----------- |
| [notebook/inverse_wave_equation_density.ipynb](https://github.com/computational-imaging/GraphPDE/blob/main/notebook/inverse_wave_equation_density.ipynb) | Full Waveform Inversion |
| [notebook/inverse_wave_equation_init.ipynb](https://github.com/computational-imaging/GraphPDE/blob/main/notebook/inverse_wave_equation_init.ipynb) | Initial State Recovery |
<br>
## Training
We also provide sample training script for both GNN and prior network. Training dataset for both can be downloaded from [here](https://drive.google.com/file/d/1FnOYE2TThb6QCgDIkaBw0CGQN7D0scbu/view?usp=sharing) and unzip the data.zip folder in the root directory.
```
# train GNN forward model
python GNN/train_2d_wave_equation.py --file ./data/training  --diffML --normalize --log --lr_schedule
```
```
# train generative prior
python Prior/autodecoder.py  --num_pe_fns 3 --use_pe --dataset_size 10000 --batch_size 32 --gpu 1  --regularize --irregular_mesh --jitter --prior init_state
python Prior/autodecoder.py  --num_pe_fns 3 --use_pe --dataset_size 10000 --batch_size 32 --gpu 1  --regularize --irregular_mesh --jitter --prior density
```


## Citation

```
@inproceedings{qzhao2022graphpde,
    title={Learning to Solve PDE-constrained Inverse Problems with Graph Networks},
    author={Qingqing Zhao and David B. Lindell and Gordon Wetzstein}
    journal={ICML},
    year={2022}
}
```

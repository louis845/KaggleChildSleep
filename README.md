# KaggleChildSleep

Repo for 8th place solution. More detailed writeup: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/460617

## Requirements
 * Python 3.10.*
 * h5py 3.9.0
 * joblib 1.3.2
 * matplotlib 3.8.0
 * numpy 1.26.0
 * pandas 2.1.1
 * pyarrow 13.0.0
 * PySide2 5.15.8
 * scikit-learn 1.3.1
 * scipy 1.11.3
 * seaborn 0.13.0
 * stumpy 1.12.0
 * torch 2.0.1+cu118
 * tqdm 4.66.1

## Relevant ML techniques used
 * [Transformers](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
   * [Alibi position embeddings](https://arxiv.org/pdf/2108.12409.pdf)
   * [S2 MLP Mixer (old version with circulant matrix)](https://arxiv.org/pdf/2106.07477.pdf)
 * [1D U-Net](https://arxiv.org/pdf/1505.04597.pdf)
 * [ResNet backbone](https://arxiv.org/pdf/1512.03385.pdf)
 * [1D RPN (like Mask RCNN but not exactly)](https://papers.nips.cc/paper_files/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)

## Statistical techniques used
 * Conditional probability inference in last layer of Unet-Transformer output

## Data preprocessing/postprocessing used

### Train
 * Random interval sampler with GroupKFold
 * Randomly shift and scale signals
 * Random 1D elastic deformation

### Inference
 * Fast summing of multiple Gaussian kernels using GPU (pytorch)

## Visualization tools used
 * Custom visualization GUIs with PySide2 + matplotlib
 * model performance visualization
 * time series visualization
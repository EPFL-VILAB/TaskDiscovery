# Task Discovery: Finding the Tasks that Neural Networks Generalize on  <!-- omit in toc -->
[Andrei Atanov](https://andrewatanov.github.io/), [Andrei Filatov](), [Teresa Yao](https://aserety.github.io), [Ajay Sohmshetty](), [Amir Zamir](https://vilab.epfl.ch/zamir/)

 [`Website`](https://taskdiscovery.epfl.ch) | [`arXiv`]() | [`BibTeX`](#citation)


<div align="center" style="padding: 0 100pt">
<img src="assets/figures/pull-figure.png">
</div>

## Abstract  <!-- omit in toc -->

When developing deep learning models, we usually decide what task we want to solve then search in the space of models in order to design one that generalizes well on this task. An intriguing question would be: what if, instead of fixing the task and searching in the model space, we fix the model and search in the task space? Can we find tasks that the model generalizes on? How do they look, or do they indicate anything?

This is the question we address in this paper. We propose a task discovery framework that automatically finds examples of such tasks via optimizing a generalization-based quantity called agreement score. With this framework, we demonstrate that the same set of images can give rise to many tasks on which neural networks generalize well. The understandings from task discovery can also provide a tool to shed more light on deep learning and its failure modes: as an example, we show that the discovered tasks can be used to generate "adversarial train-test splits" which make a model fail at test time, without changing the pixels or labels, but by only selecting how the datapoints should be split between training and testing.


## Table of Contents <!-- omit in toc -->
- [Environment setup](#environment-setup)
- [Assets](#assets)
- [Vizualizing Discovered Tasks](#vizualizing-discovered-tasks)
- [Running discovery experiments](#running-discovery-experiments)
- [Adversarial splits](#adversarial-splits)
  - [CIFAR-10](#cifar-10)
  - [ImageNet](#imagenet)
  - [CelebA](#celeba)



## Environment setup

<!-- ### Conda  -->

To install conda environment please run the following code:

```bash
git clone https://github.com/AndrewAtanov/task-discovery.git
cd task-discovery
```

Then install repository

```bash
conda create -n task_discovery -y python=3.8
source activate task_discovery
pip install -r requirements.txt
```

<!-- ### Docker -->

## Assets 

### Discovered Tasks  <!-- omit in toc -->


* `assets/tasks/discovered.csv` contains 96 discovered tasks: 32 for ResNet-18, 32 for MLP, and 32 for ViT.
* `assets/tasks/ssl-tasks.csv` contains tasks generated using regulated task discovery (with SimCLR) for ResNet-18.

### Checkpoints  <!-- omit in toc -->

### Discovered Tasks  <!-- omit in toc -->


* `assets/tasks/discovered.csv` contains 96 discovered tasks: 32 for ResNet-18, 32 for MLP, and 32 for ViT.
* `assets/tasks/ssl-tasks.csv` contains tasks generated using regulated task discovery (with SimCLR) for ResNet-18.


### Checkpoints  <!-- omit in toc -->

We provide the following task discovery checkpoints:

<!-- TODO: add links -->
- [ResNet-18 with `d=32`]()
- [NLP with `d=32`]()
- [ViT with `d=32`]()
- [ResNet-18 with `d=8`]()


### Adversarial splits <!-- omit in toc -->

You can find adversarial splits in the following directories:
- CIFAR-10: **TBD**
- ImageNet: `assets/adversarial_splits/imagenet`
- CelebA: `assets/adversarial_splits/celeba`

See the [Adversarial splits](#adversarial-splits) section for more information on how to run exepriments.

## Vizualizing Discovered Tasks
We provide an IPython notebook  `visualization.ipynb` to visualize and explore the discovered tasks using the provided checkpoints.


## Running discovery experiments

To run the agreement score computation on provided tasks use the following command:
```
python check_as.py --task_type <real/random/resnet18/mlp/vit> --task_idx <IDX (0-31)> 
```

E.g., a command for calculating the agreement score for the first task for ResNet-18.

```python
python check_as.py --task_type resnet18 --task_idx 0
```

To run task discovery, use the following command:

```
python task-discovery/train-as-uniformity.py --group mlp-discovery --config task-discovery/configs/as-uniformity-encoder/as-uniformity-default.yaml --name {h_dim}-h_dim-N={n_train_images}-uc={uniformity_coef}-noise={noise} --h_dim 32 --n_linear_tasks 32 --uniformity_coef 25 --task_temp 5 --check_val_every_n_epoch 1 --shuffle --dataset cifar10 
```

For running adversarial splits experiments refer to the next section.

## Adversarial splits
Here, we provide instructions on how to create and run adversarial splits experiments. See our website and the paper for the discription of adversarial splits.

<div align="center" style="padding: 0 100pt">
<img src="assets/figures/adversarial-split.png">
</div>


## Experiment


### CIFAR-10

__Precomputed adversarial split__ You can find the adversarial split at `assets/adversarial_splits/cifar10/cifar_splits.pt`. Splits have the following form:

```
{0: [(tensor([    1,     2,     3,  ..., 49986, 49987, 49997]),
   tensor([    0,     4,     5,  ..., 49996, 49998, 49999])),
  ...
 11: [(tensor([    0,     1,     2,  ..., 49990, 49996, 49997]),
   tensor([    3,     4,     5,  ..., 49995, 49998, 49999])),
  ...
   ```
  
The key specifies **real task** for which adversarial split created (doesn't matter for 10-way classification). For each task we have 8 splits. Splits is a tuple of two tensors which specifies indices of train and test set. 

These indices is used in the following way:

 ```
dataset = torchvision.datasets.CIFAR10(...)

split_ids  =  torch.load(PATH)[TASK_IDX][SPLIT_IDX]

dataset_train = Subset(dataset, split_ids[0])
dataset_val = Subset(dataset, split_ids[1])
 ```

__Training on adversarial split__
To run training  with adversarial split use the following command

```python
python cifar10/train.py
```

To run training with standard random split use the following command:

```python
python adversarial_splits/cifar10/train.py --adversarial_splits_path ''
```

__10-way adversarial split__ Use the following code to run adversarial split for 10-way classification
```
python adversarial_splits/cifar10/train.py --task_type real10
```

### ImageNet

__Precomputed adversarial split__
You can find the adversarial split at `assets/adversarial_splits/imagenet/adv-split.torch` and load it via [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html#torch-load). It consists of two arrays of train and test indicies as appeard in the imagenet dataset.
<!-- TODO: (add link to the code). -->

__Creating an adversarial split using a random network__
Use the following command to create an adversarial split:

```
python imagenet/random_net_adv_split.py <path to the imagenet training set>
```
You can find the split at `adv-split.torch`.

__Training a network on an adversarial split__
Use the following command to train on an *adversarial split* (this script is based on the vanilla imagenet for pytorch [example](https://github.com/pytorch/examples/tree/main/imagenet)):
```
python imagenet/train.py  --arch resnet50 --workers 32 --batch-size 256 --save_dir ./logs-adversarial-split/ --split <path to a split> <path to the imagenet root directory>
```
For example, from the root directory:
```
python imagenet/train.py  --arch resnet50 --workers 32 --batch-size 256 --save_dir ./logs-adversarial-split/ --split ../assets/adversarial_splits/imagenet/adv-split.torch <path to the imagenet root directory>
```

Use the following command to train on a *random split*:
```
python imagenet/train.py  --arch resnet50 --workers 32 --batch-size 256 --save_dir ./logs-random-split/ --val_size 0.5 <path to the imagenet root directory>
```

### CelebA

__Precomputed adversarial split__
You can find the adversarial split used in the paper in `assets/adversarial_splits/celeba/list_eval_partition_adv_split.csv`.

__Training a network on the adversarial split__
We use [run_expt.py](https://github.com/kohpangwei/group_DRO/blob/master/run_expt.py) from the [DRO](https://github.com/kohpangwei/group_DRO) official repository. Substitute the original `list_eval_partition.csv` file in the dataset folder (see the expected structure [here](https://github.com/kohpangwei/group_DRO#celeba)) with the adversarial partition file `list_eval_partition_adv_split.csv`. We use the following command to train an ERM learner:
```
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 128 --weight_decay 0.0001 --model resnet50 --n_epochs 51 --log_dir ./cleba_adv_split/ --train_from_scratch --root_dir <path to CelebA dataset>
```

We also proide a random split for comparison at `assets/adversarial_splits/celeba/list_eval_partition_random_split.csv`. Similarly, you can substitute the original partition file with this one and run the same command.



## License  <!-- omit in toc -->

This project is under the MIT license. See [LICENSE](LICENSE.md) for details.

## Citation <!-- omit in toc -->

```BibTeX
@article{atanov2022task,
  author    = {Atanov, Andrei and Filatov, Andrei and Yeo, Teresa and Sohmshetty, Ajay and Zamir, Amir},
  title     = {Task Discovery: Finding the Tasks that Neural Networks Generalize on},
  journal   = {arXiv preprint arXiv: TBD},
  year      = {2022},
}
```



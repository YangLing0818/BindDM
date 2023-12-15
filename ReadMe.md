## Binding-Adaptive Diffusion Models for Structure-Based Drug Design

Official Implementation for **Binding-Adaptive Diffusion Models for Structure-Based Drug Design**

### Environment

```shell
conda env create -f binddm.yaml
conda activate binddm
```

### Data and Preparation
The data preparation follows [TargetDiff](https://arxiv.org/abs/2303.03543). For more details, please refer to [the repository of TargetDiff](https://github.com/guanjq/targetdiff?tab=readme-ov-file#data).

### Training

```python
python train.py
```

### Sampling

```python
python sample.py
```

### Evaluation

```python
python evaluate.py
```

### Acknowledgement
Our code is adapted from the repository of [TargetDiff](https://github.com/guanjq/targetdiff). We thank the authors for sharing their code.
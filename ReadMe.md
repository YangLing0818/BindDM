## Binding-Adaptive Diffusion Models for Structure-Based Drug Design

Official implementation for "[Binding-Adaptive Diffusion Models for Structure-Based Drug Design]()" **(AAAI 2024)**.

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

### Citation
```
@article{huang2022binddm,
  title={Binding-Adaptive Diffusion Models for Structure-Based Drug Design},
  author={Huang, Zhilin and Yang, Ling and Zhang, Zaixi and Zhou, Xiangxin and Bao, Yu and Zheng, Xiawu and Yang, Yuwei and Wang, Yu and Yang, Wenming},
  journal={arXiv preprint arXiv:2211.11138},
  year={2023}
}
```
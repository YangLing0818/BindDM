## Binding-Adaptive Diffusion Models for Structure-Based Drug Design

Official implementation for "[Binding-Adaptive Diffusion Models for Structure-Based Drug Design](https://ojs.aaai.org/index.php/AAAI/article/view/29162)" **(AAAI 2024)**.

*Affiliation: Tsinghua University, Peking University, University of Science and Technology of China, University of Chinese Academy of Sciences, ByteDance, Peng Cheng Laboratory.*


![Alt text](image.png)
**Overview**: BindDM extracts subcomplex from protein-ligand complex, and utilizes it to enhance the binding-adaptive 3D molecule generation in complex.

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
@inproceedings{huang2024binddm,
  title={Binding-Adaptive Diffusion Models for Structure-Based Drug Design},
  author={Huang, Zhilin and Yang, Ling and Zhang, Zaixi and Zhou, Xiangxin and Bao, Yu and Zheng, Xiawu and Yang, Yuwei and Wang, Yu and Yang, Wenming},
  booktitle={The AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

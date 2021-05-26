# Unifying Cross-Lingual Semantic Role Labeling with Heterogeneous Linguistic Resources
Cross-lingual approach to align heterogeneous resources 


## Cite our works
```
@article{cross-lingual-unify-srl,
  title={Unifying Cross-Lingual Semantic Role Labeling with Heterogeneous Linguistic Resources},
  author={Simone Conia, Andrea Bacciu, Roberto Navigli},
  year={2021}
}
```

## How install
```bash
git clone https://github.com/SapienzaNLP/multi-srl.git unify-srl
cd unify-srl/
bash setup.sh
```



This project currently depends on:
- PyTorch 1.5
- PyTorch Lightning 0.8.5<br>
- Torch-scatter<br>
We are in the process of updating the code to PyTorch 1.7 and PyTorch Lightning 1.0.<br>
N.B: Torch-scatter in CUDA version require time to be compiled.


## Results
| Model          | CA | CZ | DE | EN | ES | ZH |
| -------------- |:------:|:------:| :------:| :------:| :------:| :------:|
| CoNLL-2009 ST best  | 80.3  | 85.4 | 79.7 | 85.6 | 80.5 | 78.6 | 
| Marcheggiani et al. (2017a) | 86.0 | - | 87.7 | 80.3 | 81.2 |
| Chen et al. (2019) | 81.7 | 88.1 | 76.4 | 91.1 | 81.3 | 81.7 | 
| Cai and Lapata (2019b) | - | - |  82.7 | 90.0 | 81.8 | 83.6 | 
| Cai and Lapata (2019a) | - | - | 83.8 | 91.2 | 82.9 | 85.0 |
| Lyu et al. (2019) | 80.9 | 87.5 | 75.8 | 90.1 | 80.5 | 83.3 | **92.66** | 
| He et al. (2019) |    | **92.66** | 
<br>
| This work m-BERT  frozen/monolingual | **92.20**   | **92.66** | 


## Authors
* **Simone Conia**  - [github](https://github.com/andreabac3)
* **Andrea Bacciu**  - [github](https://github.com/andreabac3)
* **Roberto Navigli**  - [github](https://github.com/andreabac3)

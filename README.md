<div align="center">    
 
# Unifying Cross-Lingual Semantic Role Labeling with Heterogeneous Linguistic Resources

[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://www.aclweb.org/anthology/2021.naacl-main.31/)
[![Conference](http://img.shields.io/badge/conference-NAACL--2021-4b44ce.svg)](https://2021.naacl.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

## Description
This is the repository for the paper [*Unifying Cross-Lingual Semantic Role Labeling with Heterogeneous Linguistic Resources*](https://www.aclweb.org/anthology/2021.naacl-main.31/),
to be presented at NAACL 2021 by [Simone Conia](https://c-simone.github.io), [Andrea Bacciu](https://github.com/andreabac3) and [Roberto Navigli](http://wwwusers.di.uniroma1.it/~navigli/).


## Abstract
> While cross-lingual techniques are finding increasing success in a wide range of Natural Language Processing tasks, their application
  to Semantic Role Labeling (SRL) has been strongly limited by the fact that each language adopts its own linguistic formalism, from
  PropBank for English to AnCora for Spanish and PDT-Vallex for Czech, inter alia. In this work, we address this issue and present a
  unified model to perform cross-lingual SRL over heterogeneous linguistic resources. Our model implicitly learns a high-quality mapping
  for different formalisms across diverse languages without resorting to word alignment and/or translation techniques. We find that,
  not only is our cross-lingual system competitive with the current state of the art but that it is also robust to low-data scenarios. 
  Most interestingly, our unified model is able to annotate a sentence in a single forward pass with all the inventories it was trained with, 
  providing a tool for the analysis and comparison of linguistic theories across different languages. 

## Download
You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:
```sh
git clone https://github.com/SapienzaNLP/unify-srl.git
```
or [download a zip archive](https://github.com/SapienzaNLP/unify-srl/archive/master.zip).

### Model Checkpoint
* [Link to Drive](https://drive.google.com/file/d/1hF6z22yOoyW0qK7gIL6KNH2T4JSl0mP-/view?usp=sharing)


## Cite this work
```bibtex
@inproceedings{conia-etal-2021-unify-srl,
    title = "Unifying Cross-Lingual Semantic Role Labeling with Heterogeneous Linguistic Resources",
    author = "Conia, Simone  and
      Bacciu, Andrea  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.31",
    pages = "338--351",
}
```

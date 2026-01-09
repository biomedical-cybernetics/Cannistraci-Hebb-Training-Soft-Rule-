
# Brain network science modelling of sparse neural networks enables Transformers and LLMs to perform as fully connected (NeurIPS 2025)

Yingtao Zhang<sup>1,2,4</sup>, Diego Cerretti<sup>1,2,4</sup>, Jialin Zhao<sup>1,2,4</sup>, Ziheng Liao<sup>1,2,4</sup>, Wenjing Wu<sup>1,2,4</sup>, Umberto Michieli<sup>5,6</sup> & Carlo Vittorio Cannistraci<sup>1,2,3,4</sup>

<sup>1</sup> Center for Complex Network Intelligence (CCNI), Research Center in Tsinghua Laboratory of Brain and Intelligence (THBI), Department of Psychological and Cognitive Sciences\
<sup>2</sup> Department of Computer Science  
<sup>3</sup> Department of Biomedical Engineering  
<sup>4</sup> Tsinghua University, Beijing, China  
<sup>5</sup> University of Padova, Italy\
<sup>6</sup> Canva Research

Official research code for **Cannistraci–Hebb Training soft rule (CHTs)** and **CHTss** (*CHTs + sigmoid gradual density decay*), together with the **Bipartite Receptive Field (BRF)** sparse-topology initializer.

CHTs/CHTss are **brain-inspired dynamic sparse training (DST)** methods that (i) introduce a **GPU-friendly matrix-multiplication approximation** of the Cannistraci–Hebb link predictor to enable large-scale usage, (ii) use a **soft sampling rule** for both link removal and regrowth to balance exploration/exploitation, (iii) propose **BRF** for brain-inspired sparse initialization, and (iv) extend to **sigmoid-based gradual density decay** (CHTss) for dynamic sparse training.



## Repository overview

**Key components (typical entry points):**
- `dst_scheduler.py`: dynamic spares training scheduler implementing removal / regrowth / density-decay (CHTs, CHTss).
- `sparse_topology_initialization.py`: BRF / BSW initializers (and helpers).
- `torchrun_main.py`: distributed pretraining driver for LLaMA-family experiments (C4/OpenWebText).
- `llm/`: LLM-specific dependencies + launch scripts.

## Setup

### 1) Create a conda environment

```bash
conda create -n chts python=3.10
conda activate chts
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) (Optional, for path-based link predictor) Compile the C/C++ extension

Some variants (e.g., path-based CH scoring) may rely on a compiled extension.

```bash
python setup.py build_ext --inplace
```


## Quickstart: LLaMA pretraining (torchrun)

### Dense baseline (example)

```bash
torchrun --standalone --nproc_per_node 8 torchrun_main.py \
  --run_name llama60m \
  --model_config configs/llama_60m.json \
  --dataset_name c4 \
  --batch_size 64 --total_batch_size 512 \
  --num_training_steps 10000 --warmup_steps 1000 \
  --optimizer adam --lr 3e-3 --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --save_dir checkpoints/ --only_save_last
```

### CHTs (dynamic sparse training)

```bash
torchrun --standalone --nproc_per_node 8 torchrun_main.py \
  --run_name llama60m \
  --model_config configs/llama_60m.json \
  --dataset_name c4 \
  --batch_size 64 --total_batch_size 512 \
  --num_training_steps 10000 --warmup_steps 1000 \
  --optimizer adam --lr 3e-3 --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --dst_scheduler \
  --update_interval 100 \
  --sparsity 0.90 \
  --zeta 0.10 \
  --remove_method weight_magnitude_soft \
  --regrow_method CH2_L3n_soft \
  --adaptive_zeta \
  --BRF --brf_r 0.3 --degree_dist uniform \
  --start_T 1.0 --end_T 9.0 \
  --log_to_file \
  --save_dir checkpoints/ --only_save_last
```

### CHTss (CHTs + sigmoid gradual density decay)

CHTss combines CHTs with **sigmoid-based gradual density decay**.  

```bash
torchrun --standalone --nproc_per_node 8 torchrun_main.py \
  --run_name llama60m \
  --model_config configs/llama_60m.json \
  --dataset_name c4 \
  --batch_size 64 --total_batch_size 512 \
  --num_training_steps 10000 --warmup_steps 1000 \
  --optimizer adam --lr 3e-3 --weight_decay 0 \
  --dtype bfloat16 \
  --eval_every 1000 \
  --dst_scheduler \
  --update_interval 100 \
  --granet --granet_init_sparsity 0.5 \
  --sparsity_distribution uniform \
  --pruning_method ri \
  --pruning_scheduler s_shape \
  --pruning_T_end 8000 \
  --sparsity 0.90 \
  --zeta 0.10 \
  --remove_method weight_magnitude_soft \
  --regrow_method CH2_L3n_soft \
  --adaptive_zeta \
  --BRF --brf_r 0.3 --degree_dist uniform \
  --start_T 1.0 --end_T 9.0 \
  --log_to_file \
  --save_dir checkpoints/ --only_save_last
```


## BRF initialization (what `--brf_r` means)

BRF is a brain-inspired sparse initializer that biases sparse connectivity toward “spatially closer” features; it is parameterized by **r ∈ [0, 1]** (interpretable as locality vs randomness), and supports fixed-degree or uniform degree variants (see `--degree_dist`). 


## Citation

If you use this repository in your work, please cite:

```bibtex
@misc{zhang2026brainnetworksciencemodelling,
      title={Brain network science modelling of sparse neural networks enables Transformers and LLMs to perform as fully connected}, 
      author={Yingtao Zhang and Diego Cerretti and Jialin Zhao and Wenjing Wu and Ziheng Liao and Umberto Michieli and Carlo Vittorio Cannistraci},
      year={2026},
      eprint={2501.19107},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.19107}, 
}
```

```bibtex
@inproceedings{
zhang2025brain,
title={Brain network science modelling of sparse neural networks enables Transformers and {LLM}s to perform as fully connected},
author={Yingtao Zhang and Diego Cerretti and Jialin Zhao and Wenjing Wu and Ziheng Liao and Umberto Michieli and Carlo Vittorio Cannistraci},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=OM0Qkq9xtY}
}
```

## Acknowledgements

This work was supported by the Zhou Yahui Chair Professorship award of Tsinghua University and the National High-Level Talent Program of the Ministry of Science and Technology of China. 


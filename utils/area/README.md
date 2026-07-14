# AREA resources

The attribute-description corpus in this directory comes from
[LAMDA-CL/ICML2026-AREA](https://github.com/LAMDA-CL/ICML2026-AREA), imported
from commit `e95d036aafbb1b9cb8c24b4c86af344e07c64e07`.

`models/area.py` selects the matching dataset directory automatically. A custom
corpus can be supplied with `description_root`, or with `text_des_path`,
`occ_des_path`, and `aug_des_path` individually.

If you use AREA or its annotations, cite the upstream work:

```bibtex
@inproceedings{xie2026area,
  title={AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning},
  author={Xie, Zhen-Hao and Shi, Yu-Cheng and Zhou, Da-Wei},
  booktitle={ICML},
  year={2026}
}
```

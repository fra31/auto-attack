# AutoAttack<sub>Fix</sub>
Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective,

Yiming Liu, Kezhao Liu, Yao Xiao, Ziyi Dong, Xiaogang Xu, Pengxu Wei, Liang Lin

ICLR 2025

modified from [https://github.com/fra31/auto-attack](https://github.com/fra31/auto-attack)

---

AutoAttack underperforms on models with stochastic outputs because its default sample-selection strategy fails to account for randomness. AutoAttack is an ensemble of different attack methods, it determines whether a sample is adversarial based on a single evaluation, ignoring output variability. To address this, we change the selection to a evaluation of 20 iterations, and selecting the adversarial example that results in the lowest accuracy. This modification improves the success rate by up to 10 percentage points when evaluating Diffusion-based purification models.

---

**Citations:**

AutoAttack<sub>Fix</sub>
```
@inproceedings{liu2025towards,
  title={Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective},
  author={Yiming Liu, Kezhao Liu, Yao Xiao, ZiYi Dong, Xiaogang Xu, Pengxu Wei, Liang Lin},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=shqjOIK3SA}
}
```

**Origional AutoAttack:**
```
@inproceedings{croce2020reliable,
    title = {Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks},
    author = {Francesco Croce and Matthias Hein},
    booktitle = {ICML},
    year = {2020}
}
```

---

**Check Our Adversarial Denoising Diffusion Training (ADDT) in [https://github.com/LYMDLUT/ADDT](https://github.com/LYMDLUT/ADDT)**

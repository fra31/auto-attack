# AutoAttack

"Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks"\
*Francesco Croce*, *Matthias Hein*\
[https://arxiv.org/abs/2003.01690](https://arxiv.org/abs/2003.01690)

We propose to use an ensemble of four diverse attacks to reliably evaluate robustness:
+ **APGD-CE**, our new step size-free version of PGD on the cross-entropy,
+ **APGD-DLR**, our new step size-free version of PGD on the new DLR loss,
+ **FAB**, which minimizes the norm of the adversarial perturbations [(Croce & Hein, 2019)](https://arxiv.org/abs/1907.02044),
+ **Square Attack**, a query-efficient black-box attack [(Andriushchenko et al, 2019)](https://arxiv.org/abs/1912.00049).

**Note**: we fix all the hyperparameters of the attacks, so no tuning is required to test every new classifier.

**Budget**: we use 100 iterations and 5 random restarts for the two versions of APGD and FAB and set a query limit of 5000 for Square Attack. For a faster evaluation, see below.

# Adversarial Defenses Evaluation
We here list adversarial defenses, in the Linf-threat model, recently proposed and evaluated with
+ AutoAttack (**AA**), as described above,
+ AutoAttack+ (**AA+**), i.e. AA with the addition of the targeted versions of APGD-DLR and FAB (see below for details).

We report the source of the model, i.e. if it is publicly *available*, if we received it from the *authors* or if we *retrained* it, the clean accuracy and the reported robust accuracy (note that might be calculated on a subset of the test set or on different models trained with the same defense). The robust accuracy for AA and AA+ is on the full test set.

We plan to add new models as they appear and are made available. Feel free to suggest new defenses to test!

## CIFAR-10 - Linf
The robust accuracy is evaluated at `eps = 8/255`, except for those marked with * for which `eps = 0.031`, where `eps` is the maximal Linf-norm allowed for the adversarial perturbations. The `eps` used is the same set in the original papers.\
**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

|#    |paper           |model     |clean         |report. |AA  |AA+|
|:---:|---|:---:|---:|---:|---:|---:|
|**1**| [(Carmon et al., 2019)](https://arxiv.org/abs/1905.13736)‡| *available*| 89.69| 62.5| 59.65| 59.50|
|**2**| [(Alayrac et al., 2019)](https://arxiv.org/abs/1905.13725)‡| *available*| 86.46| 56.30| 56.92| 56.01|
|**3**| [(Hendrycks et al., 2019)](https://arxiv.org/abs/1901.09960)‡| *available*| 87.11| 57.4| 54.99| 54.86|
|**4**| [(Zhang et al., 2019b)](https://arxiv.org/abs/1901.08573)\*| *available*| 84.92| 56.43| 53.18| 53.04| [53.01](https://github.com/yaodongyu/TRADES)
|**5**| [(Qin et al., 2019)](https://arxiv.org/abs/1907.02610v2)| *available*| 86.28| 52.81| 53.07| 52.82| 52.76<sup>(a)</sup>
|**6**| [(Pang et al., 2020b)](https://arxiv.org/abs/2002.08619)\*| *available*| 84.42| 60.48| 52.38| |
|**7**| [(Kumari et al., 2019)](https://arxiv.org/abs/1905.05186)| *available*| 87.80| 53.04| 49.40| |
|**8**| [(Mao et al., 2019)](http://papers.nips.cc/paper/8339-metric-learning-for-adversarial-robustness)| *authors*| 86.21| 50.03| 47.66| |
|**9**| [(Ding et al., 2020)](https://openreview.net/forum?id=HkeryxBtPB)| *available*| 84.36| 47.18| 45.57| |
|**10**| [(Zhang et al., 2019a)](https://arxiv.org/abs/1905.00877)| *retrained*| 87.20| 47.98| 45.06| |
|**11**| [(Madry et al., 2018)](https://arxiv.org/abs/1706.06083)| *available*| 87.14| 47.04| 44.29| 44.01| [43.99](https://github.com/MadryLab/cifar10_challenge)
|**12**| [(Pang et al., 2020)](https://arxiv.org/abs/1905.10626)| *available*| 80.89| 55.0| 43.78| |
|**13**| [(Wong et al., 2020)](https://arxiv.org/abs/2001.03994)| *available*| 83.34| 46.06| 43.38| |
|**14**| [(Shafahi et al., 2019)](https://arxiv.org/abs/1904.12843)| *available*| 86.11| 46.19| 41.58| |
|**15**| [(Atzmon et al., 2019)](https://arxiv.org/abs/1905.11911)\*| *available*| 81.30| 43.17| 40.61| |
|**16**| [(Zhang & Wang, 2019)](http://papers.nips.cc/paper/8459-defense-against-adversarial-attacks-using-feature-scattering-based-adversarial-training)| *available*| 89.98| 60.6| 38.78| |
|**17**| [(Moosavi-Dezfooli et al., 2019)](http://openaccess.thecvf.com/content_CVPR_2019/html/Moosavi-Dezfooli_Robustness_via_Curvature_Regularization_and_Vice_Versa_CVPR_2019_paper)| *authors*| 83.11| 41.4| 38.67| |
|**18**| [(Zhang & Xu, 2020)](https://openreview.net/forum?id=Syejj0NYvr&noteId=Syejj0NYvr)| *available*| 90.25| 68.7| 38.57| |
|**19**| [(Kim & Wang, 2020)](https://openreview.net/forum?id=rJlf_RVKwr)| *available*| 91.51| 57.23| 36.10| |
|**20**| [(Jang et al., 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.html)| *available*| 78.91| 37.40| 35.09| |
|**21**| [(Wang & Zhang, 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Bilateral_Adversarial_Training_Towards_Fast_Training_of_More_Robust_Models_ICCV_2019_paper.html)| *available*| 92.80| 58.6| 30.96| |
|**22**| [(Xiao et al., 2020)](https://arxiv.org/abs/1905.10510)\*| *available*| 79.28| 52.4| 17.99| |
|**23**| [(Jin & Rinard, 2020)](https://arxiv.org/abs/2003.04286)| *available*| 90.84| 71.22| 4.61| |
|**24**| [(Mustafa et al., 2019)](https://arxiv.org/abs/1904.00887)| *available*| 89.16| 32.32| 0.55| |
|**25**| [(Chan et al., 2020)](https://arxiv.org/abs/1912.10185)| *retrained*| 93.79| 15.5| 0.18| |

## MNIST - Linf
The robust accuracy is computed at `eps = 0.3` in the Linf-norm.

|#    |paper           |model     |clean         |report. |AA  |AA+|
|:---:|---|:---:|---:|---:|---:|---:|
|**1**| [(Zhang et al., 2020)](https://arxiv.org/abs/1906.06316)| *available*| 98.38| 96.38| 93.95| 93.95|
|**2**| [(Gowal et al., 2019)](https://arxiv.org/abs/1810.12715)| *available*| 98.34| 93.88| 92.75| 92.75|
|**3**| [(Zhang et al., 2019b)](https://arxiv.org/abs/1901.08573)| *available*| 99.48| 95.60| 92.76| 92.74| [92.58](https://github.com/yaodongyu/TRADES)
|**4**| [(Atzmon et al., 2019)](https://arxiv.org/abs/1905.11911)| *available*| 99.35| 97.35| 90.85| |
|**5**| [(Madry et al., 2018)](https://arxiv.org/abs/1706.06083)| *available*| 98.53| 89.62| 88.43| 88.43| [88.06](https://github.com/MadryLab/mnist_challenge)
|**6**| [(Jang et al., 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.html)| *available*| 98.47| 94.61| 87.99| |
|**7**| [(Wong et al., 2020)](https://arxiv.org/abs/2001.03994)| *available*| 98.50| 88.77| 82.88| |
|**8**| [(Taghanaki et al., 2019)](http://openaccess.thecvf.com/content_CVPR_2019/html/Taghanaki_A_Kernelized_Manifold_Mapping_to_Diminish_the_Effect_of_Adversarial_CVPR_2019_paper.html)| *retrained*| 98.86| 64.25| 0.00| |

# How to use AutoAttack

### PyTorch models
Import and initialize AutoAttack with

```python
from autoattack import AutoAttack
adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, plus=False)
```

where:
+ `forward_pass` returns the logits and takes input with components in [0, 1] (NCHW format expected),
+ `norm = ['Linf' | 'L2']` is the norm of the threat model,
+ `eps` is the bound on the norm of the adversarial perturbations,
+ `plus = True` adds to the list of the attacks the targeted versions of APGD and FAB (AA+).

To apply the standard evaluation, where the attacks are run sequentially on batches of size `bs` of `images`, use

```python
x_adv = adversary.run_standard_evaluation(images, labels, bs=batch_size)
```

To run the attacks individually, use

```python
dict_adv = adversary.run_standard_evaluation_individual(images, labels, bs=batch_size)
```

which returns a dictionary with the adversarial examples found by each attack.

To specify a subset of attacks add e.g. `adversary.attacks_to_run = ['apgd-ce']`.

### TensorFlow models
To evaluate models implemented in TensorFlow, use

```python
import utils_tf
model_adapted = utils_tf.ModelAdapter(logits, x_input, y_input, sess)

from autoattack import AutoAttack
adversary = AutoAttack(model_adapted, norm='Linf', eps=epsilon, plus=False, is_tf_model=True)
```

where:
+ `logits` is the tensor with the logits given by the model,
+ `x_input` is a placeholder for the input for the classifier (NHWC format expected),
+ `y_input` is a placeholder for the correct labels,
+ `sess` is a TF session.

The evaluation can be run in the same way as done with PT models.

### Example
An example of how to use AutoAttack can be found in `examples/eval.py`. To run the standard evaluation on a pretrained
model on CIFAR-10 use
```
python eval.py [--cheap --individual --plus]
```
where the optional flags activate respectively the *cheap* version (see below), the *individual* version (all the attacks are run on the full test set) and the *plus* version (AA+, including also the targeted attacks).

## Other options
### Cheaper version
Adding the line
```python
adversary.cheap()
```
before running the evaluation, a 5 times cheaper version is used, i.e. no random restarts are given to APGD and FAB and the query limit for Square is 1000. The results are usually slightly worse to those given by the full version.

### Random seed
It is possible to fix the random seed used for the attacks with, e.g., `adversary.seed = 0`. In this case the same seed is used for all the attacks used, otherwise a different random seed is picked for each attack.

## Citation
```
@unpublished{croce2020reliable,
    title = {Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks},
    authors = {Francesco Croce and Matthias Hein},
    note = {preprint arXiv:2003.01690},
    year = {2020}
}
```

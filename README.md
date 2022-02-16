# AutoAttack

"Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks"\
*Francesco Croce*, *Matthias Hein*\
ICML 2020\
[https://arxiv.org/abs/2003.01690](https://arxiv.org/abs/2003.01690)


We propose to use an ensemble of four diverse attacks to reliably evaluate robustness:
+ **APGD-CE**, our new step size-free version of PGD on the cross-entropy,
+ **APGD-DLR**, our new step size-free version of PGD on the new DLR loss,
+ **FAB**, which minimizes the norm of the adversarial perturbations [(Croce & Hein, 2019)](https://arxiv.org/abs/1907.02044),
+ **Square Attack**, a query-efficient black-box attack [(Andriushchenko et al, 2019)](https://arxiv.org/abs/1912.00049).

**Note**: we fix all the hyperparameters of the attacks, so no tuning is required to test every new classifier.

## News
+ [Sep 2021]
	+ We add [automatic checks](https://github.com/fra31/auto-attack/blob/master/flags_doc.md) for potential cases where the standard version of AA might be non suitable or sufficient for robustness evaluation.
	+ The evaluations of models on CIFAR-10 and CIFAR-100 are no longer maintained. Up-to-date leaderboards are available in [RobustBench](https://robustbench.github.io/).
+ [Mar 2021] A version of AutoAttack wrt L1, which includes the extensions of APGD and Square Attack [(Croce & Hein, 2021)](https://arxiv.org/abs/2103.01208), is available!
+ [Oct 2020] AutoAttack is used as standard evaluation in the new benchmark [RobustBench](https://robustbench.github.io/), which includes a [Model Zoo](https://github.com/RobustBench/robustbench) of the most robust classifiers! Note that this page and RobustBench's leaderboards are maintained simultaneously.
+ [Aug 2020]
	+ **Updated version**: in order to *i)* scale AutoAttack (AA) to datasets with many classes and *ii)* have a faster and more accurate evaluation, we use APGD-DLR and FAB with their *targeted* versions.
	+ We add the evaluation of models on CIFAR-100 wrt Linf and CIFAR-10 wrt L2.
+ [Jul 2020] A short version of the paper is accepted at [ICML'20 UDL workshop](https://sites.google.com/view/udlworkshop2020/) for a spotlight presentation!
+ [Jun 2020] The paper is accepted at ICML 2020!

# Adversarial Defenses Evaluation
We here list adversarial defenses, for many threat models, recently proposed and evaluated with the standard version of
**AutoAttack (AA)**, including
+ *untargeted APGD-CE* (no restarts),
+ *targeted APGD-DLR* (9 target classes),
+ *targeted FAB* (9 target classes),
+ *Square Attack* (5000 queries).

See below for the more expensive AutoAttack+ (AA+) and more options.

We report the source of the model, i.e. if it is publicly *available*, if we received it from the *authors* or if we *retrained* it, the architecture, the clean accuracy and the reported robust accuracy (note that might be calculated on a subset of the test set or on different models trained with the same defense). The robust accuracy for AA is on the full test set.

We plan to add new models as they appear and are made available. Feel free to suggest new defenses to test!

**To have a model added**: please check [here](https://github.com/fra31/auto-attack/issues/new/choose).

**Checkpoints**: many of the evaluated models are available and easily accessible at this [Model Zoo](https://github.com/RobustBench/robustbench).

## CIFAR-10 - Linf
The robust accuracy is evaluated at `eps = 8/255`, except for those marked with * for which `eps = 0.031`, where `eps` is the maximal Linf-norm allowed for the adversarial perturbations. The `eps` used is the same set in the original papers.\
**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

**Update**: this is no longer maintained, but an up-to-date leaderboard is available in [RobustBench](https://robustbench.github.io/).

|#    |paper           |model     |architecture |clean         |report. |AA  |
|:---:|---|:---:|:---:|---:|---:|---:|
|**1**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)‡| *available*| WRN-70-16| 91.10| 65.87| 65.88|
|**2**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)‡| *available*| WRN-28-10| 89.48| 62.76| 62.80|
|**3**| [(Wu et al., 2020a)](https://arxiv.org/abs/2010.01279)‡| *available*| WRN-34-15| 87.67| 60.65| 60.65|
|**4**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)‡| *available*| WRN-28-10| 88.25| 60.04| 60.04|
|**5**| [(Carmon et al., 2019)](https://arxiv.org/abs/1905.13736)‡| *available*| WRN-28-10| 89.69| 62.5| 59.53|
|**6**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)| *available*| WRN-70-16| 85.29| 57.14| 57.20|
|**7**| [(Sehwag et al., 2020)](https://github.com/fra31/auto-attack/issues/7)‡| *available*| WRN-28-10| 88.98| -| 57.14|
|**8**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)| *available*| WRN-34-20| 85.64| 56.82| 56.86|
|**9**| [(Wang et al., 2020)](https://openreview.net/forum?id=rklOg6EFwS)‡| *available*| WRN-28-10| 87.50| 65.04| 56.29|
|**10**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)| *available*| WRN-34-10| 85.36| 56.17| 56.17|
|**11**| [(Alayrac et al., 2019)](https://arxiv.org/abs/1905.13725)‡| *available*| WRN-106-8| 86.46| 56.30| 56.03|
|**12**| [(Hendrycks et al., 2019)](https://arxiv.org/abs/1901.09960)‡| *available*| WRN-28-10| 87.11| 57.4| 54.92|
|**13**| [(Pang et al., 2020c)](https://arxiv.org/abs/2010.00467)| *available*| WRN-34-20| 86.43| 54.39| 54.39|
|**14**| [(Pang et al., 2020b)](https://arxiv.org/abs/2002.08619)| *available*| WRN-34-20| 85.14| -| 53.74|
|**15**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| *available*| WRN-34-20| 88.70| 53.57| 53.57|
|**16**| [(Zhang et al., 2020b)](https://arxiv.org/abs/2002.11242)| *available*| WRN-34-10| 84.52| 54.36| 53.51|
|**17**| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569)| *available*| WRN-34-20| 85.34| 58| 53.42|
|**18**| [(Huang et al., 2020)](https://arxiv.org/abs/2002.10319)\*| *available*| WRN-34-10| 83.48| 58.03| 53.34|
|**19**| [(Zhang et al., 2019b)](https://arxiv.org/abs/1901.08573)\*| *available*| WRN-34-10| 84.92| 56.43| 53.08|
|**20**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| *available*| WRN-34-10| 88.22| 52.86| 52.86|
|**21**| [(Qin et al., 2019)](https://arxiv.org/abs/1907.02610v2)| *available*| WRN-40-8| 86.28| 52.81| 52.84|
|**22**| [(Chen et al., 2020a)](https://arxiv.org/abs/2003.12862)| *available*| RN-50 (x3)| 86.04| 54.64| 51.56|
|**23**| [(Chen et al., 2020b)](https://github.com/fra31/auto-attack/issues/26)| *available*| WRN-34-10| 85.32| 51.13| 51.12|
|**24**| [(Sitawarin et al., 2020)](https://github.com/fra31/auto-attack/issues/23)| *available*| WRN-34-10| 86.84| 50.72| 50.72|
|**25**| [(Engstrom et al., 2019)](https://github.com/MadryLab/robustness)| *available*| RN-50| 87.03| 53.29| 49.25|
|**26**| [(Kumari et al., 2019)](https://arxiv.org/abs/1905.05186)| *available*| WRN-34-10| 87.80| 53.04| 49.12|
|**27**| [(Mao et al., 2019)](http://papers.nips.cc/paper/8339-metric-learning-for-adversarial-robustness)| *available*| WRN-34-10| 86.21| 50.03| 47.41|
|**28**| [(Zhang et al., 2019a)](https://arxiv.org/abs/1905.00877)| *retrained*| WRN-34-10| 87.20| 47.98| 44.83|
|**29**| [(Madry et al., 2018)](https://arxiv.org/abs/1706.06083)| *available*| WRN-34-10| 87.14| 47.04| 44.04|
|**30**| [(Pang et al., 2020a)](https://arxiv.org/abs/1905.10626)| *available*| RN-32| 80.89| 55.0| 43.48|
|**31**| [(Wong et al., 2020)](https://arxiv.org/abs/2001.03994)| *available*| RN-18| 83.34| 46.06| 43.21|
|**32**| [(Shafahi et al., 2019)](https://arxiv.org/abs/1904.12843)| *available*| WRN-34-10| 86.11| 46.19| 41.47|
|**33**| [(Ding et al., 2020)](https://openreview.net/forum?id=HkeryxBtPB)| *available*| WRN-28-4| 84.36| 47.18| 41.44|
|**34**| [(Atzmon et al., 2019)](https://arxiv.org/abs/1905.11911)\*| *available*| RN-18| 81.30| 43.17| 40.22|
|**35**| [(Moosavi-Dezfooli et al., 2019)](http://openaccess.thecvf.com/content_CVPR_2019/html/Moosavi-Dezfooli_Robustness_via_Curvature_Regularization_and_Vice_Versa_CVPR_2019_paper)| *authors*| WRN-28-10| 83.11| 41.4| 38.50|
|**36**| [(Zhang & Wang, 2019)](http://papers.nips.cc/paper/8459-defense-against-adversarial-attacks-using-feature-scattering-based-adversarial-training)| *available*| WRN-28-10| 89.98| 60.6| 36.64|
|**37**| [(Zhang & Xu, 2020)](https://openreview.net/forum?id=Syejj0NYvr&noteId=Syejj0NYvr)| *available*| WRN-28-10| 90.25| 68.7| 36.45|
|**38**| [(Jang et al., 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.html)| *available*| RN-20| 78.91| 37.40| 34.95|
|**39**| [(Kim & Wang, 2020)](https://openreview.net/forum?id=rJlf_RVKwr)| *available*| WRN-34-10| 91.51| 57.23| 34.22|
|**40**| [(Wang & Zhang, 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Bilateral_Adversarial_Training_Towards_Fast_Training_of_More_Robust_Models_ICCV_2019_paper.html)| *available*| WRN-28-10| 92.80| 58.6| 29.35|
|**41**| [(Xiao et al., 2020)](https://arxiv.org/abs/1905.10510)\*| *available*| DenseNet-121| 79.28| 52.4| 18.50|
|**42**| [(Jin & Rinard, 2020)](https://arxiv.org/abs/2003.04286v1) | [*available*](https://github.com/charlesjin/adversarial_regularization/blob/6a3704757dcc7c707ff38f8b9de6f2e9e27e0a89/pretrained/pretrained88.pth) | RN-18| 90.84| 71.22| 1.35|
|**43**| [(Mustafa et al., 2019)](https://arxiv.org/abs/1904.00887)| *available*| RN-110| 89.16| 32.32| 0.28|
|**44**| [(Chan et al., 2020)](https://arxiv.org/abs/1912.10185)| *retrained*| WRN-34-10| 93.79| 15.5| 0.26|

## CIFAR-100 - Linf
The robust accuracy is computed at `eps = 8/255` in the Linf-norm, except for the models marked with * for which `eps = 0.031` is used. \
**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).\
\
**Update**: this is no longer maintained, but an up-to-date leaderboard is available in [RobustBench](https://robustbench.github.io/).

|#    |paper           |model     |architecture |clean         |report. |AA  |
|:---:|---|:---:|:---:|---:|---:|---:|
|**1**| [(Gowal et al. 2020)](https://arxiv.org/abs/2010.03593)‡| *available*| WRN-70-16| 69.15| 37.70| 36.88|
|**2**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| *available*| WRN-34-20| 62.55| 30.20| 30.20|
|**3**| [(Gowal et al. 2020)](https://arxiv.org/abs/2010.03593)| *available*| WRN-70-16| 60.86| 30.67| 30.03|
|**4**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| *available*| WRN-34-10| 60.64| 29.33| 29.33|
|**5**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)| *available*| WRN-34-10| 60.38| 28.86| 28.86|
|**6**| [(Hendrycks et al., 2019)](https://arxiv.org/abs/1901.09960)‡| *available*| WRN-28-10| 59.23| 33.5| 28.42|
|**7**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| *available*| WRN-34-10| 70.25| 27.16| 27.16|
|**8**| [(Chen et al., 2020b)](https://github.com/fra31/auto-attack/issues/26)| *available*| WRN-34-10| 62.15| -| 26.94|
|**9**| [(Sitawarin et al., 2020)](https://github.com/fra31/auto-attack/issues/22)| *available*| WRN-34-10| 62.82| 24.57| 24.57|
|**10**| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569)| *available*| RN-18| 53.83| 28.1| 18.95|

## MNIST - Linf
The robust accuracy is computed at `eps = 0.3` in the Linf-norm.

|#    |paper           |model     |clean         |report. |AA  |
|:---:|---|:---:|---:|---:|---:|
|**1**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)| *available*| 99.26| 96.38| 96.34|
|**2**| [(Zhang et al., 2020a)](https://arxiv.org/abs/1906.06316)| *available*| 98.38| 96.38| 93.96|
|**3**| [(Gowal et al., 2019)](https://arxiv.org/abs/1810.12715)| *available*| 98.34| 93.78| 92.83|
|**4**| [(Zhang et al., 2019b)](https://arxiv.org/abs/1901.08573)| *available*| 99.48| 95.60| 92.81|
|**5**| [(Ding et al., 2020)](https://openreview.net/forum?id=HkeryxBtPB)| *available*| 98.95| 92.59| 91.40|
|**6**| [(Atzmon et al., 2019)](https://arxiv.org/abs/1905.11911)| *available*| 99.35| 97.35| 90.85|
|**7**| [(Madry et al., 2018)](https://arxiv.org/abs/1706.06083)| *available*| 98.53| 89.62| 88.50|
|**8**| [(Jang et al., 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.html)| *available*| 98.47| 94.61| 87.99|
|**9**| [(Wong et al., 2020)](https://arxiv.org/abs/2001.03994)| *available*| 98.50| 88.77| 82.93|
|**10**| [(Taghanaki et al., 2019)](http://openaccess.thecvf.com/content_CVPR_2019/html/Taghanaki_A_Kernelized_Manifold_Mapping_to_Diminish_the_Effect_of_Adversarial_CVPR_2019_paper.html)| *retrained*| 98.86| 64.25| 0.00|

## CIFAR-10 - L2
The robust accuracy is computed at `eps = 0.5` in the L2-norm.\
**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

**Update**: this is no longer maintained, but an up-to-date leaderboard is available in [RobustBench](https://robustbench.github.io/).

|#    |paper           |model     |architecture |clean         |report. |AA  |
|:---:|---|:---:|:---:|---:|---:|---:|
|**1**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)‡| *available*| WRN-70-16| 94.74| -| 80.53|
|**2**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)| *available*| WRN-70-16| 90.90| -| 74.50|
|**3**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)| *available*| WRN-34-10| 88.51| 73.66| 73.66|
|**4**| [(Augustin et al., 2020)](https://arxiv.org/abs/2003.09461)‡| *authors*| RN-50| 91.08| 73.27| 72.91|
|**5**| [(Engstrom et al., 2019)](https://github.com/MadryLab/robustness)| *available*| RN-50| 90.83| 70.11| 69.24|
|**6**| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569)| *available*| RN-18| 88.67| 71.6| 67.68|
|**7**| [(Rony et al., 2019)](https://arxiv.org/abs/1811.09600)| *available*| WRN-28-10| 89.05| 67.6| 66.44|
|**8**| [(Ding et al., 2020)](https://openreview.net/forum?id=HkeryxBtPB)| *available*| WRN-28-4| 88.02| 66.18| 66.09|

# How to use AutoAttack

### Installation

```
pip install git+https://github.com/fra31/auto-attack
```

### PyTorch models
Import and initialize AutoAttack with

```python
from autoattack import AutoAttack
adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard')
```

where:
+ `forward_pass` returns the logits and takes input with components in [0, 1] (NCHW format expected),
+ `norm = ['Linf' | 'L2' | 'L1']` is the norm of the threat model,
+ `eps` is the bound on the norm of the adversarial perturbations,
+ `version = 'standard'` uses the standard version of AA.

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
To evaluate models implemented in TensorFlow 1.X, use

```python
from autoattack import utils_tf
model_adapted = utils_tf.ModelAdapter(logits, x_input, y_input, sess)

from autoattack import AutoAttack
adversary = AutoAttack(model_adapted, norm='Linf', eps=epsilon, version='standard', is_tf_model=True)
```

where:
+ `logits` is the tensor with the logits given by the model,
+ `x_input` is a placeholder for the input for the classifier (NHWC format expected),
+ `y_input` is a placeholder for the correct labels,
+ `sess` is a TF session.

If TensorFlow's version is 2.X, use

```python
from autoattack import utils_tf2
model_adapted = utils_tf2.ModelAdapter(tf_model)

from autoattack import AutoAttack
adversary = AutoAttack(model_adapted, norm='Linf', eps=epsilon, version='standard', is_tf_model=True)
```

where:
+ `tf_model` is tf.keras model without activation function 'softmax'

The evaluation can be run in the same way as done with PT models.

### Examples
Examples of how to use AutoAttack can be found in `examples/`. To run the standard evaluation on a pretrained
PyTorch model on CIFAR-10 use
```
python eval.py [--individual] --version=['standard' | 'plus']
```
where the optional flags activate respectively the *individual* evaluations (all the attacks are run on the full test set) and the *version* of AA to use (see below).

## Other versions
### AutoAttack+
A more expensive evaluation can be used specifying `version='plus'` when initializing AutoAttack. This includes
+ *untargeted APGD-CE* (5 restarts),
+ *untargeted APGD-DLR* (5 restarts),
+ *untargeted FAB* (5 restarts),
+ *Square Attack* (5000 queries),
+ *targeted APGD-DLR* (9 target classes),
+ *targeted FAB* (9 target classes).

### Randomized defenses
In case of classifiers with stochastic components one can combine AA with Expectation over Transformation (EoT) as in [(Athalye et al., 2018)](https://arxiv.org/abs/1802.00420) specifying `version='rand'` when initializing AutoAttack.
This runs
+ *untargeted APGD-CE* (no restarts, 20 iterations for EoT),
+ *untargeted APGD-DLR* (no restarts, 20 iterations for EoT).

### Custom version
It is possible to customize the attacks to run specifying `version='custom'` when initializing the attack and then, for example,
```python
if args.version == 'custom':
	adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2
```

## Other options
### Random seed
It is possible to fix the random seed used for the attacks with, e.g., `adversary.seed = 0`. In this case the same seed is used for all the attacks used, otherwise a different random seed is picked for each attack.

### Log results
To log the intermediate results of the evaluation specify `log_path=/path/to/logfile.txt` when initializing the attack.

## Citation
```
@inproceedings{croce2020reliable,
    title = {Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks},
    author = {Francesco Croce and Matthias Hein},
    booktitle = {ICML},
    year = {2020}
}
```

```
@inproceedings{croce2021mind,
    title={Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers}, 
    author={Francesco Croce and Matthias Hein},
    booktitle={ICML},
    year={2021}
}
```

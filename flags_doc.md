## On the usage of AutoAttack

We here describe cases where the standard version of AA might be non suitable or sufficient for robustness evaluation. While AA is designed to generalize across defenses, there are categories like
randomized, non differentiable or dynamic defenses for which it cannot be applied in the standard version, since those rely on differet principles than commonly used robust models. In such cases,
specific modifications or adaptive attacks [(Tramèr et al., 2020)](https://arxiv.org/abs/2002.08347) might be necessary.

## Checks
We introduce a few automatic checks to warn the user in case the classifier presents behaviors typical of non standard models. Below we describe the type of flags which might be raised and provide
some suggestions about how the robustness evaluation could be improved in the specific cases. Note that some of the checks are in line with the analyses and suggestions by recent works
([Carlini et al., 2019](https://arxiv.org/abs/1902.06705); [Croce et al., 2020](https://arxiv.org/abs/2010.09670); [Pintor et al., 2021](https://arxiv.org/abs/2106.09947)) which provide guidelines for 
evaluating robustness and detecting failures of attacks.

### Randomized defenses
**Raised if** the clean accuracy of the classifier on a batch or the corresponding logits vary across multiple runs.\
**Explanation:** non deterministic classifiers need to be evaluated with specific techniques e.g. EoT [(Athalye et al., 2018)](http://proceedings.mlr.press/v80/athalye18a.html) and mislead
standard attacks. We suggest to use AA with `version='rand'`, which inclueds APGD combined with EoT. Note that there might still be some random components
in the network which however do not change the predictions or the logits beyond the chosen threshold.

### Softmax output is given
**Raised if** the model outputs a probability distribution. \
**Explanation:** AA expects the model to return logits, i.e. pre-softmax output of the network. If this is not the case, although the classification is unaltered,
there might be numerical instabilities which prevent the gradient-based attacks to perform well.

### Zero gradient
**Raised if** the gradient at the (random) starting point of APGD is zero for any image when using the DLR loss. \
**Explanation:** zero gradient prevents progress in gradient-based iterative attacks. A source of it could be connected to the cross-entropy loss and the scale of the logits, but a remedy consists in
using margin based losses ([Carlini & Wagner, 2017](https://ieeexplore.ieee.org/abstract/document/7958570); [Croce & Hein, 2020](https://arxiv.org/abs/2003.01690)). Vanishing gradients can be also due to specific
components of the networks, like input quantization (see e.g. [here](https://github.com/fra31/auto-attack/issues/44)), which do not allow
backpropagation. In this case one might use BPDA [(Athalye et al., 2018)](http://proceedings.mlr.press/v80/athalye18a.html), which approximates such functions with differentiable counterparts, or black-box attacks, especially those, like Square Attack, which do not rely on
gradient estimation.

### Square Attack improves the robustness evaluation
**Raised if** Square Attack reduces the robust accuracy yielded by the white-box attacks. \
**Explanation:** as mentioned by [Carlini et al. (2019)](https://arxiv.org/abs/1902.06705), black-box attacks performing better than white-box ones is one of the hints of overestimation of robustness. In this case one might run
Square Attack with higher budget (more queries, random restarts) or design adaptive attacks, since it is likely that the tested defense has some features preventing standard gradient-based methods
to be effective.

### Optimization at inference time (only PyTorch models)
**Raised if** standard PyTorch functions for computing the gradients are called when running inference with the given classifier. \
**Explanation:** several defenses which include some optimization loop in the inference procedure have appeared. While AA can give a first estimation of the robustness, it is necessary in this case
to design adaptive attacks, since such models usually modify the input before classifying it, which requires specific techniques for evaluation. Note that this check is non trivial to make automatic,
and we invite the user to be aware that AA might be not the best option to evaluate dynamic defenses.
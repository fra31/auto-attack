# AutoAttack

Example:

```
from autoattack import AutoAttack
adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon)

x_adv = adversary.run_standard_evaluation(images, labels, bs=batch_size)
```

This runs the standard evaluation. To specify a subset of attacks add e.g. `adversary.attacks_to_run = ['square']`.

import torch
import warnings
import math
import sys

from autoattack.other_utils import L2_norm


funcs = {'grad': 0,
    'backward': 0,
    #'enable_grad': 0
    '_make_grads': 0,
    }

checks_doc_path = 'flags_doc.md'


def is_randomized(model, x, y, bs=250, n=5, alpha=1e-4, logger=None):
    acc = []
    corrcl = []
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            output = model(x)
            corrcl_curr = (output.max(1)[1] == y).sum().item()
            corrcl.append(corrcl_curr)
            outputs.append(output / (L2_norm(output, keepdim=True) + 1e-10))
    acc = [c != corrcl_curr for c in corrcl]
    max_diff = 0.
    for c in range(n - 1):
        for e in range(c + 1, n):
            diff = L2_norm(outputs[c] - outputs[e])
            max_diff = max(max_diff, diff.max().item())
            #print(diff.max().item(), max_diff)
    if any(acc) or max_diff > alpha:
        msg = 'it seems to be a randomized defense! Please use version="rand".' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
        return True
    return False


def check_range_output(model, x, alpha=1e-5, logger=None):
    with torch.no_grad():
        output = model(x)
    fl = [output.max() < 1. + alpha, output.min() >  -alpha,
        ((output.sum(-1) - 1.).abs() < alpha).all()]
    if all(fl):
        msg = 'it seems that the output is a probability distribution,' +\
            ' please be sure that the logits are used!' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    return output.shape[-1]


def check_zero_gradients(grad, logger=None):
    z = grad.view(grad.shape[0], -1).abs().sum(-1)
    #print(grad[0, :10])
    if (z == 0).any():
        msg = f'there are {(z == 0).sum()} points with zero gradient!' + \
            ' This might lead to unreliable evaluation with gradient-based attacks.' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_square_sr(acc_dict, alpha=.002, logger=None):
    if 'square' in acc_dict.keys() and len(acc_dict) > 2:
        acc = min([v for k, v in acc_dict.items() if k != 'square'])
        if acc_dict['square'] < acc - alpha:
            msg = 'Square Attack has decreased the robust accuracy of' + \
                f' {acc - acc_dict["square"]:.2%}.' + \
                ' This might indicate that the robustness evaluation using' +\
                ' AutoAttack is unreliable. Consider running Square' +\
                ' Attack with more iterations and restarts or an adaptive attack.' + \
                f' See {checks_doc_path} for details.'
            if logger is None:
                warnings.warn(Warning(msg))
            else:
                logger.log(f'Warning: {msg}')


''' from https://stackoverflow.com/questions/26119521/counting-function-calls-python '''
def tracefunc(frame, event, args):
    if event == 'call' and frame.f_code.co_name in funcs.keys():
        funcs[frame.f_code.co_name] += 1

        
def check_dynamic(model, x, is_tf_model=False, logger=None):
    if is_tf_model:
        msg = 'the check for dynamic defenses is not currently supported'
    else:
        msg = None
        sys.settrace(tracefunc)
        model(x)
        sys.settrace(None)
        #for k, v in funcs.items():
        #    print(k, v)
        if any([c > 0 for c in funcs.values()]):
            msg = 'it seems to be a dynamic defense! The evaluation' + \
                ' with AutoAttack might be insufficient.' + \
                f' See {checks_doc_path} for details.'
    if not msg is None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    #sys.settrace(None)


def check_n_classes(n_cls, attacks_to_run, apgd_targets, fab_targets,
    logger=None):
    msg = None
    if 'apgd-dlr' in attacks_to_run or 'apgd-t' in attacks_to_run:
        if n_cls <= 2:
            msg = f'with only {n_cls} classes it is not possible to use the DLR loss!'
        elif n_cls == 3:
            msg = f'with only {n_cls} classes it is not possible to use the targeted DLR loss!'
        elif 'apgd-t' in attacks_to_run and \
            apgd_targets + 1 > n_cls:
            msg = f'it seems that more target classes ({apgd_targets})' + \
                f' than possible ({n_cls - 1}) are used in {"apgd-t".upper()}!'
    if 'fab-t' in attacks_to_run and fab_targets + 1 > n_cls:
        if msg is None:
            msg = f'it seems that more target classes ({apgd_targets})' + \
                f' than possible ({n_cls - 1}) are used in FAB-T!'
        else:
            msg += f' Also, it seems that too many target classes ({apgd_targets})' + \
                f' are used in {"fab-t".upper()} ({n_cls - 1} possible)!'
    if not msg is None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')



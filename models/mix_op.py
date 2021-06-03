import math

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from models.layers import *
import utils
from utils.model_utils import detach_variable


def build_candidate_ops(candidate_ops,
                        in_channels: int,
                        out_channels: int,
                        stride: int,
                        ops_order: str):
    if candidate_ops is None:
        raise ValueError('Must specify a candidate set.')

    name_to_ops = {
        'Identity': lambda in_c, out_c, s: IdentityLayer(in_channels=in_c,
                                                         out_channels=out_c,
                                                         ops_order=ops_order),
        'Zero': lambda in_c, out_c, s: ZeroLayer(stride=s),
    }

    name_to_ops.update({
        # 3X3 MBConvolutions
        '3x3_MBConv1': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=3,
                                                                  stride=s,
                                                                  expand_ratio=1),
        '3x3_MBConv2': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=3,
                                                                  stride=s,
                                                                  expand_ratio=2),
        '3x3_MBConv3': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=3,
                                                                  stride=s,
                                                                  expand_ratio=3),
        '3x3_MBConv4': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=3,
                                                                  stride=s,
                                                                  expand_ratio=4),
        '3x3_MBConv5': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=3,
                                                                  stride=s,
                                                                  expand_ratio=5),
        '3x3_MBConv6': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=3,
                                                                  stride=s,
                                                                  expand_ratio=6),

        # 5X5 MBConvolutions
        '5x5_MBConv1': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=5,
                                                                  stride=s,
                                                                  expand_ratio=1),
        '5x5_MBConv2': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=5,
                                                                  stride=s,
                                                                  expand_ratio=2),
        '5x5_MBConv3': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=5,
                                                                  stride=s,
                                                                  expand_ratio=3),
        '5x5_MBConv4': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=5,
                                                                  stride=s,
                                                                  expand_ratio=4),
        '5x5_MBConv5': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=5,
                                                                  stride=s,
                                                                  expand_ratio=5),
        '5x5_MBConv6': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=5,
                                                                  stride=s,
                                                                  expand_ratio=6),

        # 7X7 MBConvolutions
        '7x7_MBConv1': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=7,
                                                                  stride=s,
                                                                  expand_ratio=1),
        '7x7_MBConv2': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=7,
                                                                  stride=s,
                                                                  expand_ratio=2),
        '7x7_MBConv3': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=7,
                                                                  stride=s,
                                                                  expand_ratio=3),
        '7x7_MBConv4': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=7,
                                                                  stride=s,
                                                                  expand_ratio=4),
        '7x7_MBConv5': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=7,
                                                                  stride=s,
                                                                  expand_ratio=5),
        '7x7_MBConv6': lambda in_c, out_c, s: MBInvertedConvLayer(in_channels=in_c,
                                                                  out_channels=out_c,
                                                                  kernel_size=7,
                                                                  stride=s,
                                                                  expand_ratio=6),
    })

    return [
        name_to_ops[name](in_channels, out_channels, stride) for name in candidate_ops
    ]


class MixedEdge(BaseModule):
    # Full, two, None, full_v2
    MODE = None

    def __init__(self, candidate_ops):
        super(MixedEdge, self).__init__()

        self._candidate_ops = nn.ModuleList(candidate_ops)

        # Architecture Parameters
        self._arch_param_path_alpha = Parameter(torch.Tensor(self._n_choices))

        # Binary Gates
        self._arch_param_path_wb = Parameter(torch.Tensor(self._n_choices))

        self._active_index = [0]
        self._inactive_index = None

        self._log_prob = None
        self._current_prob_over_ops = None

    @property
    def _n_choices(self):
        return len(self._candidate_ops)

    @property
    def _probs_over_ops(self):
        # Softmax to probability
        probs = F.softmax(self._arch_param_path_alpha, dim=0)

        return probs

    @property
    def _chosen_index(self):
        probs = self._probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def _chosen_op(self):
        index, _ = self._chosen_index
        return self._candidate_ops[index]

    @property
    def _random_op(self):
        index = np.random.choice([_i for _i in range(self._n_choices)], 1)[0]
        return self._candidate_ops[index]

    def _entropy(self, eps=1e-8):
        probs = self._probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        return entropy

    @property
    def _active_op(self):
        # Assume only one path is active
        return self._candidate_ops[self._active_index[0]]

    def is_zero_layer(self):
        return self._active_op.is_zero_layer()

    def set_chosen_op_active(self):
        chosen_idx, _ = self._chosen_index

        # TODO(joohyun): Use pop?
        self._active_index = [chosen_idx]
        self._inactive_index = [_i for _i in range(0, chosen_idx)] + \
                               [_i for _i in range(chosen_idx + 1, self._n_choices)]

    """ Start of module requirements """

    def forward(self, x):
        if MixedEdge.MODE == 'full' or MixedEdge.MODE == 'two':
            output = 0
            for _i in self._active_index:
                op_i = self._candidate_ops[_i](x)
                output = output + self._arch_param_path_wb[_i] * op_i
            for _i in self._inactive_index:
                op_i = self._candidate_ops[_i](x)
                output = output + self._arch_param_path_wb[_i] * op_i.detach()

        elif MixedEdge.MODE == 'full_v2':
            def run_function(candidate_ops, active_id):
                def forward(_x):
                    return candidate_ops[active_id](_x)

                return forward

            def backward_function(candidate_ops, active_id, binary_gates):
                def backward(_x, _output, grad_output):
                    binary_grad = torch.zeros_like(binary_gates.data)
                    with torch.no_grad():

                        # k = binary gate index
                        for k in range(len(candidate_ops)):
                            if k != active_id:
                                out_k = candidate_ops[k](_x.data)
                            else:
                                out_k = _output.data
                            grad_k = torch.sum(out_k * grad_output)
                            binary_grad[k] = grad_k
                    return binary_grad

                return backward

            output = ArchGradientFunction.apply(
                x,
                self._arch_param_path_wb,
                run_function(self._candidate_ops, self._active_index[0]),
                backward_function(self._candidate_ops, self._active_index[0], self._arch_param_path_wb)
            )
        else:
            output = self._active_op(x)
        return output

    @property
    def module_str(self):
        chosen_index, probs = self._chosen_index
        return 'Mix({}, {.3f}'.format(self._candidate_ops[chosen_index].module_str, probs)

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def get_flops(self, x):
        # Only active paths taken into consideration when calculating FLOPs
        flops = 0
        for i in self._active_index:
            delta_flop, _ = self._candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)

    """ End of requirements """

    def binarize(self):
        """
            Prepare:
            active_index,
            inactive_index,
            arch_param_path_wb,
            log_prob (optional),
            current_prob_over_ops (optional)
        """
        self._log_prob = None

        # Reset binary gates
        self._arch_param_path_wb.data.zero_()

        # Binarize according to probs
        probs = self._probs_over_ops
        if MixedEdge.MODE == 'two':
            # Sample two ops according to 'probs'
            sample_op = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(
                torch.stack([self._arch_param_path_alpha[idx] for idx in sample_op]),
                dim=0
            )

            # Initialize prob over ops
            self._current_prob_over_ops = torch.zeros_like(probs)

            for i, idx in enumerate(sample_op):
                self._current_prob_over_ops[idx] = probs_slice[i]

            # Chose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0]  # Chosen index would be 0 or 1
            active_op = sample_op[c].item()
            inactive_op = sample_op[1 - c].item()
            self._active_index = [active_op]
            self._inactive_index = [inactive_op]

            # Set binary gate
            self._arch_param_path_wb.data[active_op] = 1.0

        else:
            sample = torch.multinomial(probs.data, 1)[0].item()
            self._active_index = [sample]
            self._inactive_index = [_i for _i in range(0, sample)] + \
                                   [_i for _i in range(sample + 1, self.n_choices)]
            self._log_prob = torch.log(probs[sample])
            self._current_prob_over_ops = probs

            # Set binary gate
            self._arch_param_path_wb.data[sample] = 1.0

        # Avoid over-regularization
        for _i in range(self._n_choices):
            for name, param in self._candidate_ops[_i].named_parameters():
                param.grad = None

    def set_arch_param_grad(self):
        binary_grads = self._arch_param_path_wb.grad.data
        if self._active_op.is_zero_layer():
            self._arch_param_path_alpha.grad = None
            return

        if self._arch_param_path_alpha.grad is None:
            self._arch_param_path_alpha.grad = torch.zeros_like(self._arch_param_path_alpha.data)

        if MixedEdge.MODE == 'two':
            involved_idx = self._active_index + self._inactive_index
            probs_slice = F.softmax(
                torch.stack([self._arch_param_path_alpha[idx] for idx in involved_idx]),
                dim=0
            ).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    self._arch_param_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * (utils.delta_ij(i, j) - probs_slice[i])
            for _i, idx in enumerate(self._active_index):
                self._active_index[_i] = (idx, self._arch_param_path_alpha.data[idx].item())
            for _i, idx in enumerate(self._inactive_index):
                self._inactive_index[_i] = (idx, self._arch_param_path_alpha.data[idx].item())
        else:
            probs = self._probs_over_ops.data
            for i in range(self._n_choices):
                for j in range(self._n_choices):
                    self._arch_param_path_alpha.grad.data[i] += \
                        binary_grads[j] * probs[j] * (utils.delta_ij(i, j) - probs[i])

    def rescale_updated_arch_param(self):
        if not isinstance(self._active_index[0], tuple):
            assert self._active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (self._active_index + self._inactive_index)]
        old_alphas = [alpha for _, alpha in (self._active_index - self._inactive_idex)]
        new_alphas = [self._arch_param_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self._arch_param_path_alpha.data[idx] -= offset


class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = detach_variable(x)
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)

        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None

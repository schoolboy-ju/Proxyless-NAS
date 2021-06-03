def make_divisible(v, divisor, min_val=None):
    """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_val:
        :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "Invalid kernel size: {}".format(kernel_size)
        padding_1 = get_same_padding(kernel_size[0])
        padding_2 = get_same_padding(kernel_size[1])
        return padding_1, padding_2
    assert isinstance(kernel_size, int), "Kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "Kernel size should be odd number"
    return kernel_size // 2


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list


def list_sum(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] + list_sum(x[1:])


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0

import time
import sys
import logging
import math
import numpy as np

import matplotlib.pyplot as plt
import torch


def export_plot(ys, ylabel, title, filename):
    """
    Export a plot in filename

    Args:
        ys: (list) of float / int to plot
        filename: (string) directory
    """
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def get_logger(filename):
    """
    Return a logger instance to a file
    """
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return logger


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, discount=0.9):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.exp_avg = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self.discount = discount

    def update(self, current, values=[], exact=[], strict=[], exp_avg=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [
                    v * (current - self.seen_so_far),
                    current - self.seen_so_far,
                ]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += current - self.seen_so_far
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v
        for k, v in exp_avg:
            if k not in self.exp_avg:
                self.exp_avg[k] = v
            else:
                self.exp_avg[k] *= self.discount
                self.exp_avg[k] += (1 - self.discount) * v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = "%%%dd/%%%dd [" % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += "=" * (prog_width - 1)
                if current < self.target:
                    bar += ">"
                else:
                    bar += "="
            bar += "." * (self.width - prog_width)
            bar += "]"
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ""
            if current < self.target:
                info += " - ETA: %ds" % eta
            else:
                info += " - %ds" % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += " - %s: %.4f" % (
                        k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]),
                    )
                else:
                    info += " - %s: %s" % (k, self.sum_values[k])

            for k, v in self.exp_avg.iteritems():
                info += " - %s: %.4f" % (k, v)

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += (prev_total_width - self.total_width) * " "

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = "%ds" % (now - self.start)
                for k in self.unique_values:
                    info += " - %s: %.4f" % (
                        k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]),
                    )
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


def batch_iterator(*args, batch_size=1000, shuffle=False):
    """
    Given a torch tensor or a sequence of torch tensors (which must all have
    the same first dimension), returns a generator which iterates over the
    tensor(s) in mini-batches of size batch_size.
    Pass shuffle=True to randomize the order.
    """
    if type(args) in {list, tuple}:
        multi_arg = True
        n = len(args[0])
        for i, arg_i in enumerate(args):
            assert torch.is_tensor(arg_i)
            assert len(arg_i) == n
    else:
        multi_arg = False
        n = len(args)

    indices = torch.randperm(n) if shuffle else torch.arange(n)

    n_batches = math.ceil(float(n) / batch_size)
    for batch_index in range(n_batches):
        batch_start = batch_size * batch_index
        batch_end = min(batch_size * (batch_index + 1), n)
        batch_indices = indices[batch_start:batch_end]
        if multi_arg:
            yield tuple(arg[batch_indices] for arg in args)
        else:
            yield args[batch_indices]

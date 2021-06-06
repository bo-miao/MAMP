import os
import logging
import warnings
warnings.filterwarnings("ignore")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    handler = logging.StreamHandler()
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def get_davis_ref_index(i, mem_gap, ref):
    if ref == 5:
        ref_index = list(filter(lambda x: x <= i, [0, 5])) + list(
            filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
        ref_index = sorted(list(set(ref_index)))
    elif ref == 4:
        ref_index = [0] + list(filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
    elif ref == 2:
        ref_index = [0] + [i - 1]
        ref_index = sorted(list(set(ref_index)))
    elif ref == 1:
        ref_index = [i]
    elif ref == 0:
        ref_index = [0]
    else:
        raise NotImplementedError

    return ref_index


def get_youtube_ref_index(i, mem_gap, ref, annotation_index):
    if ref == 4:
        ref_index = list(set(filter(lambda x: x <= i, annotation_index)))
        rest = 4 - len(ref_index)
        ref_index += list(filter(lambda x: x > 0, range(i, i - mem_gap * rest, -mem_gap)))[::-1]
        ref_index = sorted(ref_index)
    elif ref == 2:
        ref_index = list(filter(lambda x: x <= i, annotation_index)) + [i-1]
        ref_index = sorted(list(set(ref_index)))
    else:
        raise NotImplementedError

    return ref_index


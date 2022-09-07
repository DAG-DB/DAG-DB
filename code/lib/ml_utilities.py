import argparse
import datetime
import importlib
from itertools import product
import json
import logging
import os
from os import path
import platform
import shutil
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
from dateutil.tz import tzlocal
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import one_hot
# from torch.utils.tensorboard import SummaryWriter

# Needed for engine='odf' in df.to_excel:
import odf  # Used in to_excel


ADDITIONAL_ARGS = {
    '--max_size': 'MAX_SIZE',
    '--lamb': 'LAMBDA',
    '--temperature': 'NOISE_TEMPERATURE',
    '--h_null': 'h_NULL',
    '--h_lr': 'h_LR',
    '--rho': 'z_RHO',
    '--mu': 'z_MU',
    '--dags': 'DAGS'
}
parser = argparse.ArgumentParser(description='Args for running.')
parser.add_argument('-c', default=None, help='which config file to use')
parser.add_argument('-j', default=None, help='Job ID within an array')
for arg, hp_name in ADDITIONAL_ARGS.items():
    parser.add_argument(
        arg, default=None, help=f'Sets {hp_name}')
args = parser.parse_args()

if args.c is None:
    CONFIG_FILE = 'config'
else:
    CONFIG_FILE = f'named_configs.{args.c}'
c = importlib.import_module(CONFIG_FILE)
CONFIG_FILE += '.py'

job_id = '' if args.j is None else '_' + str(args.j).zfill(3)

"""Note: DO NOT use h. hyperparameters as default arguments in any module --- 
as they are only evaluated when the function is first defined.
c. constants can be used as default arguments.
"""

STD_NOW = datetime.datetime.now(tz=tzlocal()).strftime("%Y-%m-%d_%Z_%H:%M:%S,%f")[:-3]
START_TIME = perf_counter()
# writer = SummaryWriter()


# See https://exceptionshub.com/how-to-get-filename-of-the-__main__-module-in-python.html
# def main_folder():
#     return path.basename(
#         path.dirname(path.abspath(sys.modules['__main__'].__file__)))


# timezone stuff from
# https://stackoverflow.com/questions/35057968/get-system-local-timezone-in-python
def time_stamp(title: str, folder: str = None,
               the_now: str = STD_NOW):
    """ Returns a string giving the filepath with title preceded by the
    the_now's date, time and timezone, and (if folder is not None) in the
    folder.
    However, if the_now == None, then use the selections date, time, timezone.
    """
    the_now = the_now or datetime.datetime.now(tz=tzlocal()).strftime(
        "%Y-%m-%d_%Z_%H:%M:%S,%f"[:-3])
    filestem = the_now + title
    if folder is None:
        return filestem, filestem + '.log'
    else:
        return filestem, os.path.join(folder, filestem) + '.log'

if CONFIG_FILE == 'config.py':
    LOG_FILESTEM, LOG_FILEPATH = time_stamp(job_id, c.LOGS_FOLDER)
else:
    LOG_FILESTEM, LOG_FILEPATH = time_stamp(
        '_' + args.c + job_id, c.LOGS_FOLDER)


logging.basicConfig(
    format='%(asctime)s,%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %Z %H:%M:%S',
    filename=LOG_FILEPATH,
    encoding='utf-8',
    filemode='w',
    level=logging.DEBUG
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)
log = logging.info
warning = logging.warning


if c.DEVICE is None:
    c.DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
else:
    c.DEVICE = torch.device(c.DEVICE)
try:
    OBJECTIVE_GOOD_DIRECTION = c.OBJECTIVE_GOOD_DIRECTION
except AttributeError:
    OBJECTIVE_GOOD_DIRECTION = "Low"

# def log(entry='', backspaces=0, end=None):
#     """
#     :param entry: str or ready to be turned into str by str built-in
#     :param backspaces: int, the number of backspaces to include when printing
#     :return:
#     """
#     if end is None:
#         print('\b' * backspaces + str(entry))
#     else:
#         print('\b' * backspaces + str(entry), end=end)
#     logging.info(str(entry))


def value_code(value, save=True, to_json=False):
    if isinstance(value, torch.device):
        value = 'torch.device("cuda" if torch.cuda.is_available() else ' \
                '"cpu")  # Used: \"' + str(value) + '\"'
        return value
    elif callable(value):
        value = value.__module__ + '.' + value.__name__
        return value
    elif not isinstance(value, str):
        return value
    if save and (not to_json):
        return '\'' + value + '\''
    else:
        return value


class Constants():
    def __init__(self):
        self.constants = {key: value for key, value in
                          vars(c).items() if key[0].isupper()}
        self.hyperparameters = h
        # self.hyperparameters = c.hyperparameters
        """if 'CHRONOLOGY' not in self.constants.keys():
            self.constants['CHRONOLOGY'] = list()
        self.constants['CHRONOLOGY'].append(STD_NOW)
        """

    def print(self):
        for key, value in self.constants.items():
            print(f'{key} = {value_code(value, save=False)}')
        print('\nhyperparameters = {')
        for key, value in self.hyperparameters.items():
            print(f'\t{key}: {value_code(value, save=False)}, ')
        print('}\n')

    def log(self):
        log(f'\nLog for run {LOG_FILESTEM}\n')
        for key, value in self.constants.items():
            log(f'{key} = {value_code(value)}')
        log('\nhyperparameters = {')
        last_key = len(self.hyperparameters) - 1
        for k, (key, value) in enumerate(self.hyperparameters.items()):
            if k < last_key:
                log(f"\t'{key}': {value_code(value)},")
            else:
                log(f"\t'{key}': {value_code(value)}")
        log('}\n')

    def __repr__(self, save=True):
        if save:
            repr_list = [f'{key} = {value_code(value)}' for key, value
                         in
                         self.constants.items()]
        else:
            repr_list = [f'{key} = {value}' for key, value in
                         self.constants.items()]
        repr = '\n'.join(repr_list)
        repr = 'Constants: {\n' + repr + '\n}'
        return repr

    def config_modules(self):
        return [name for name in dir(c) if name[0].islower()]


# https://dev.to/0xbf/use-dot-syntax-to-access-dictionary-key-python-tips-10ec
class DictDot(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)


"""hp is the same as c.hyperparameters, but with any hyperparameters values 
which aren't lists changed into single element lists
"""
h = DictDot(c.hyperparameters)
# h = DictDot({key: None for key in c.hyperparameters})
hp = c.hyperparameters.copy()
bases = list()
for key, value in c.hyperparameters.items():
    if not isinstance(value, list):
        hp[key] = [value]
    else:
        bases.append(len(value))
keys, values = zip(*hp.items())
n_h = np.prod([len(value) for value in values])
last_key = len(keys) - 1


for arg, hp_name in ADDITIONAL_ARGS.items():
    arg_key = arg[2:]
    arg_value = eval(f'args.{arg_key}')
    if arg_value is None:
        continue
    assert eval(f'h.{hp_name} is None')
    exec(f'h.{hp_name} = {arg_value}')

def set_numpy_rng(key, rs):
    try:
        seed = rs
        h[key] = np.random.default_rng(seed)
    except KeyError:
        exit(f'Error setting or seeding numpy rng')


def set_numpy_seed(key, rs):
    try:
        seed = rs
        h[key] = seed
    except KeyError:
        exit(f'Error setting numpy seed')


def set_torch_rng(key, rs):
    try:
        # See https://pytorch.org/docs/stable/generated/torch
        # .Generator.html on manual_seed()
        device = c.DEVICE
        torch_seed_rng = np.random.default_rng(rs)
        zeros_ones = [0] * 16 + [1] * 16
        torch_seed_rng.shuffle(zeros_ones)  # Note numpy shuffle is inplace
        torch_seed = sum([bit * (2 ** (32 - b))
                          for b, bit in enumerate(zeros_ones)])
        h[key] = torch.Generator(device=device).manual_seed(
            torch_seed)
        pass
    except:
        exit(f'Error setting or seeding torch rng')


def set_rngs(seed_set=None):
    seed_set = h.DAGS if seed_set is None else seed_set
    if isinstance(seed_set, str):
        h.n_rs = None
    else:
        set_numpy_seed('n_rs', seed_set[0])


def set_and_log_h(keys, bundle, last_key=None, set_h=True, log_flag=True):
    global h
    last_key = last_key or len(keys) - 1
    if set_h:
        h.clear()
    if log_flag:
        log('hyperparameters = {')
    for k, (key, value) in enumerate(zip(keys, bundle)):
        if set_h:
            h[key] = value
        if not log_flag:
            continue
        if k < last_key:
            log(f"\t'{key}': {value_code(value)},")
        else:
            log(f"\t'{key}': {value_code(value)}")
    if log_flag:
        log('}\n')
    set_rngs()


def time_interval(time_period: float):
    return datetime.timedelta(seconds=round(time_period))


def best_result_format(best_result):
    if isinstance(best_result, float):
        # print('float')
        return f'{best_result:.03f}'
    elif isinstance(best_result, tuple):
        if len(best_result) == 1:
            return f'{best_result[0]:.03f}'
        # print('tuple')
        best_result_string = '('
        for elem in best_result:
            best_result_string += f'{best_result_format(elem)}, '
        return best_result_string[:-2] + ')'
    return f'{best_result}'


def log_intro_hp_run(n_h, hp_run, time_elapsed):
    to_log = f'>>>> {hp_run=} of {n_h}'
    if hp_run > 1:
        now = datetime.datetime.now(tz=tzlocal())
        time_estimate = time_elapsed * n_h / (hp_run - 1)
        end_estimate = now + datetime.timedelta(
            seconds=(time_estimate - time_elapsed))
        to_log += f', time elapsed' \
                  f' {time_interval(time_elapsed)} ' \
                  f'of' \
                  f' estimated {time_interval(time_estimate)}, '
        to_log += '\nimplying ending at '
        if platform.system() == 'Windows':
            to_log += end_estimate.strftime(
                "%H:%M:%S%Z on %A %#d %B %Y")
        else:
            to_log += end_estimate.strftime(
                "%H:%M:%S%Z on %A %-d %B %Y")
    log(to_log)


def log_end_run(n_h, last_key, time_elapsed, results, best_so_far):
    log('\n\n')
    log(f'Time taken over all {n_h} given sets of hyperparameters'
        f'={time_interval(time_elapsed)}, '
        f'averaging {time_interval(time_elapsed / n_h)} per run')
    log('\n\n ---- Table of results ----\n')
    n_figures = len(bases)
    extra_spaces = max(4 - n_figures, 0)
    log(((n_figures + 1 + extra_spaces - 4) * ' ') + 'code  hp_run  '
                                                     'result')
    numbers = ['']
    for base in bases:
        numbers = [number + str(i) for number in numbers for i in
                   range(base)]
    for hp_run, result in enumerate(results, 1):
        log((extra_spaces * ' ') + f' {numbers[hp_run - 1]} '
                                   f'{hp_run:>7} '
                                   f' {best_result_format(result)}')
    log(' ' + ('-' * 26) + '\n')
    best_result_string = best_result_format(best_so_far[0])
    log(f'++++ Best result was {best_result_string} on hp_run'
        f'={best_so_far[1]} with')
    set_and_log_h(best_so_far[2].keys(), best_so_far[2].values(), last_key,
                  set_h=False)
    end_log_message()


def end_log_message():
    time_taken = perf_counter() - START_TIME
    tt_sec = str(round(time_taken % 60)).zfill(2)
    tt_min = str(int((time_taken // 60) % 60)).zfill(2)
    tt_hr = str(int(time_taken // 3600))
    log(f'\nTime taken {tt_hr}:{tt_min}:{tt_sec} (h:mm:ss)')
    log(f'\nEnd of log for run {LOG_FILESTEM}')


# def over_hp(func):
#     def wrapper_over_hp(*args, **kwargs):
#         # http://stephantul.github.io/python/2019/07/20/product-dict/
#         best_so_far_initial = (None, None, None, None)
#         best_so_far = best_so_far_initial
#         over_hp_start = perf_counter()
#         results = list()
#         full_results = list()
#         for hp_run, bundle in enumerate(product(*values), 1):
#             log('\n')
#             time_elapsed = perf_counter() - over_hp_start
#             log_intro_hp_run(n_h, hp_run, time_elapsed)
#             set_and_log_h(keys, bundle, last_key)
#             set_rngs()
#             h['hp_run'] = hp_run  # Needed for TensorBoard and saved models
#
#             best_of_this_hp_run = func(*args, **kwargs)  # for debugging
#             # try:  # for running, e.g. to handle config typos
#             #     best_of_this_hp_run = func(*args, **kwargs)
#             # except:
#             #     match OBJECTIVE_GOOD_DIRECTION:
#             #         case "Low":
#             #             best_of_this_hp_run = [np.inf]
#             #         case "High":
#             #             best_of_this_hp_run = [-np.inf]
#
#             if best_of_this_hp_run is None:
#                 best_of_this_hp_run = 0
#             del h['hp_run']
#             log(f'\nEnd of hp run {hp_run}.  Result of run:')
#             # log(best_of_this_hp_run)
#             if type(best_of_this_hp_run) == list:
#                 result = best_of_this_hp_run[0]
#             else:
#                 result = best_of_this_hp_run
#             log(result)
#             results.append(result)
#             full_results.append(best_of_this_hp_run)
#             if (best_so_far == best_so_far_initial) \
#                     or (
#                     (OBJECTIVE_GOOD_DIRECTION == "Low")
#                     and (result < best_so_far[0])
#             ) or (
#                     (OBJECTIVE_GOOD_DIRECTION == "High")
#                     and (result > best_so_far[0])):
#                 best_so_far = (result, hp_run, h.copy(), best_of_this_hp_run)
#         time_elapsed = perf_counter() - over_hp_start
#         log_end_run(n_h, last_key, time_elapsed, results, best_so_far)
#         for key in keys:
#             h[key] = None
#         return full_results
#
#     return wrapper_over_hp


# def test_hp(func):
#     # if n_h > 1:
#     #     exit('The dictionary config.hyperparameters contains options - need '
#     #          'singleton for test.')
#
#     def wrapper_test_hp(*args, **kwargs):
#         best_so_far_initial = (None, None, None, None)
#         best_so_far = best_so_far_initial
#         test_hp_start = perf_counter()
#         for bundle in product(*values):
#             log('\n')
#             time_elapsed = perf_counter() - test_hp_start
#             set_and_log_h(keys, bundle, last_key, log_flag=False)
#             set_rngs()
#             results = func(*args, **kwargs)
#         time_elapsed = perf_counter() - test_hp_start
#         log(f'Time taken for whole test {time_elapsed}')
#         log('\n')
#         for key in keys:
#             h[key] = None
#         return results
#
#     return wrapper_test_hp


def div(a, b):
    return a / b if b else np.inf


def savefig(title):
    title += '.pdf'
    fname = time_stamp(folder=c.LOGS_FOLDER, title=title)
    plt.savefig(fname)


def save_model(model, model_sub_folder= '', model_name=None,
               parameter_name=None,
               parameter=None):
    """ Can either save a model directly in the c.MODEL_FOLDER or save in
    c.MODEL_FOLDER/<model_title name associated with run>/model_sub_folder
    """
    model_title = '_model'
    model_title = LOG_FILESTEM + '_' + str(h.hp_run).rjust(4, '_') + \
                  model_title
    # if parameter is not None:
    #     model_title += '_'
    #     if parameter_name is not None:
    #         model_title += parameter_name
    #     model_title += str(parameter)
    # model_title = time_stamp(model_title)
    # if model_sub_folder != '':
    #     model_name = 'model' if model_name == '' else model_name
    #     path = os.path.join(c.MODEL_FOLDER, model_title, model_sub_folder)
    #     Path(path).mkdir(parents=True, exist_ok=True)
    #     path = os.path.join(path, model_name + '.pt')
    # else:
    model_title += '' if model_name is None else '_' + model_name
    path = os.path.join(c.MODEL_FOLDER, model_title + '.pt')
    torch.save(model.state_dict(), path)
    return path

def get_latents_save_name(tag):
    latents_title = LOG_FILESTEM + '_' + str(h.hp_run).rjust(4, '_') + \
                                                          f'_{tag}_latents.lzma'
    # latents_title = time_stamp(latents_title)
    return os.path.join(c.LATENTS_FOLDER, latents_title)


def load_model(model, model_name, model_folder='', model_sub_folder=''):
    path = os.path.join(
        c.MODEL_FOLDER, model_folder, model_sub_folder, model_name + '.pt')
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_best_of_run(model):
    model_title = str(h.hp_run) + '_model'
    model_title = time_stamp(model_title)
    path = os.path.join(c.MODEL_FOLDER, model_title + '.pt')
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def timeit(func):
    """From https://stackoverflow.com/questions/35656239/how-do-i-time-script-execution-time-in-pycharm-without-adding-code-every-time#37303872
     (lower down the page)"""
    def measure_time(*args, **kw):
        start_time = perf_counter()
        result = func(*args, **kw)
        log("Processing time of %s(): %.6f seconds."
              % (func.__qualname__, perf_counter() - start_time))
        return result

    return measure_time


def to_device_tensor(x, dtype=torch.float32):
    """
    Convert array to device tensor
    :param x: numpy array
    :return:  pytorch c.DEVICE tensor
    """
    # if c.DEVICE == torch.device("cpu"):
    assert not isinstance(x, torch.Tensor)
    try:
        return torch.tensor(x, dtype=dtype, device=c.DEVICE)
    except TypeError:  # e.g. x is an array on Nones
        return x


def list_to_one_hot_device_tensor(x, num_classes):
    """
    Convert list to device tensor
    :param x: list
    :return:  pytorch c.DEVICE tensor
    """
    return one_hot(
        to_device_tensor(np.array(x), dtype=torch.int64),
        num_classes=num_classes
    ).float()


def to_array(x):
    """
    Convert device tensor to array
    :param x: pytorch c.DEVICE tensor
    :return: numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    elif np.isscalar(x):
        return np.array([x])
    elif isinstance(x, list):
        try:
            return np.stack(x)
        except TypeError:
            return np.array(x)
    return x.cpu().detach().numpy()


def get_optimizer(trial, net, optimizer_string):
    """
    Get optimizer from string

    :param net: torch.nn.Module to optimize on
    :param optimizer_string: str
    :return: torch.optim.Optimizer
    """
    left_bracket_position = optimizer_string.find("(")
    return eval(
        optimizer_string[:left_bracket_position] + "(net.parameters(), "
        + optimizer_string[left_bracket_position + 1:])


def get_lr_scheduler(trial, optimizer, lr_scheduler_string=None):
    """
    Get lr_scheduler from string

    :param optimizer: torch.nn.optim to schedule
    :param lr_scheduler_string: str
    :return: torch.optim.lr_scheduler
    """
    if lr_scheduler_string is None:
        return None
    else:
        left_bracket_position = lr_scheduler_string.find("(")
        return eval(
            lr_scheduler_string[:left_bracket_position] + "(optimizer, "
            + lr_scheduler_string[left_bracket_position + 1:])


def get_sampler(trial, dataset, batch_sampler_string=None, batch_size=None):
    """
    Get lr_scheduler from string

    :param dataset: torch dataset to use
    :param batch_sampler_string: str
    :return: torch.optim.lr_scheduler
    """
    if batch_sampler_string is None:
        return None
    else:
        left_bracket_position = batch_sampler_string.find("(")
        return eval(
            batch_sampler_string[:left_bracket_position] +
            f"(dataset, num_samples={batch_size}," +
            batch_sampler_string[left_bracket_position + 1:]
        )


def save_to_ods(i, data, tag):
    if not c.SAVE_TO_ODS:
        return
    assert len(data.shape) == 2
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    df = pd.DataFrame(data)
    filename = LOG_FILESTEM
    # if i is not None:
    #     filename += f'_{i}_'
    filename += f'_{tag}.ods'
    df.to_excel(
        os.path.join(c.LOGS_FOLDER, filename),
        sheet_name=str(i),
        engine='odf'
    )


def save_metrics(i, metrics):
    # log('Before params')
    params = {
        hp_name: [eval(f'h.{hp_name}')]
        for hp_name in ADDITIONAL_ARGS.values()
    }
    # log(f'After {params=}')
    metrics = {key: [metrics[key]] for key in metrics}
    # log(f'After {metrics=}')
    to_save = pd.DataFrame.from_dict({**params, 'i': [i], **metrics,
                                      'graph_set': h.DAGS})
    # log(f'After {to_save=}')
    config_file = CONFIG_FILE if args.c is None else args.c
    file_path = os.path.join(c.RESULTS_FOLDER, config_file + '.csv')
    # log(f'After {file_path=}')
    to_save.to_csv(
        file_path, mode='a', header=not os.path.exists(file_path), index=False)
    log(f'Last epoch results saved in {args.c}.csv')


consts = Constants()
consts.log()

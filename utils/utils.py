import ast
import collections
import os
import random
import re
import signal
import threading
import time
import traceback
import json

import numpy as np
import psutil

from codes.logger.logger import logger


def deprecated(func):
    """
    This is a decorator for deprecated functions.
    """

    def wrapper(*args, **kwargs):
        print(f"[Warning] Function '{func.__name__}' is deprecated.")
        return func(*args, **kwargs)

    return wrapper


def get_HH_mm_ss(td):
    days, seconds = td.days, td.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return hours, minutes, secs


def get_files(target_dir, file_type: str):
    # input: root path of specific frameworks
    # output: list format: ["file_dir_1", "file_dir_2", ...]
    # function, go through all files in the framework and only find python files
    file_lists = []
    for root, subdirs, files in os.walk(target_dir):
        for file in files:
            if not file.endswith(file_type):
                continue
            file_lists.append(os.path.join(root, file))
    return file_lists


def misline_2_range(missline, max_num):
    miss_range = np.ones(max_num + 1)
    miss_branch = []
    for miss in missline:
        if "->" in miss:  # This is a branch
            miss_branch.append(miss)
            continue
        elif "-" in miss:  # This is a statement
            bottom, top = miss.split("-")
            bottom, top = (int(bottom), int(top))
            miss_range[bottom:top + 1] = 0
        else:
            miss_range[int(miss)] = 0

    miss_branch_dict = collections.defaultdict(list)
    for branch in miss_branch:
        start_no = branch.split("->")[0]
        end_no = branch.split("->")[1]
        miss_branch_dict[start_no].append(end_no)
    return miss_range, miss_branch_dict


def parse_miss_line(miss_line, max_line):
    # miss_range: np.array([0,1,1,...]) with shape: max_line+1, miss_branch: {start_no: [end_no], ...}, all keys and values are str
    miss_range, miss_branches = misline_2_range(miss_line, max_line)
    return miss_range, miss_branches


def concatenate_vector(vector_1, vector_2):
    return np.concatenate((vector_1, vector_2), axis=0)


def clean_code(result, silent: bool = False):
    """
    If we cannot identify the code, we return the whole result.
    :param result: string, the answer returned from GPT
    :param silent: boolean, silent or not
    :return: string or list, the cleaned code or the list of cleaned code
    """
    result = re.sub(r"\n+", "\n", result)  # we replace multiple \n with \n
    try:
        identifier = result.strip().split("\n")[0]
        if identifier not in ["```", "```python", "<code>"]:
            identifier = result.split("import")[0].split("\n")[-2]
    except:
        identifier = ""
    if silent is False:
        logger.info(f"Identifier: {identifier}")
    # we make sure that the wrapped code is started with import
    # the () in pattern represent the group of re.search. For instance re.search('A(B)C', result),
    # will return three groups for A, B, and C, respectively.
    if identifier == "<code>":
        pattern = r"\<code\>\n(import[\s\S]*?)\</code\>"
    elif identifier.startswith("```"):
        pattern = rf"{identifier}\n([\s\S]*?)```"
    else:
        if silent is False:
            logger.info(f"Unknown Identifier: {identifier}, use the default one: ```")
        pattern = r"```\n([\s\S]*?)```"
    try:
        pattern = re.compile(pattern)
    except:
        if silent is False:
            logger.info(f"Pattern: {pattern}")
        return result
    code_list = []
    for code_match in re.finditer(pattern, result):
        code_list.append(code_match.group(1))
    if len(code_list) == 0:
        if silent is False:
            logger.info("No code found in string.")
    try:
        # we remove all comments
        code_list = [ast.unparse(ast.parse(code)) for code in code_list]
    except:
        pass
    if len(code_list) == 0:
        return ""
    elif len(code_list) == 1:
        return code_list[0]
    else:
        return code_list

def count_tokens(string):
    """
    Count the number of tokens in string.
    Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    Simple way: one token = 4 chars
    count_tokens("test") = 1
    count_tokens("This is a sample string") = 5.75
    :param string:
    :return:
    """
    num_tokens = len(string)
    return num_tokens / 4


def _merge_status(status_list) -> str:
    """
    Internal function to merge multiple runtime status.
    If at least one status is "success", return "success".
    Otherwise return "fail".
    :param status_list:
    :return:
    """
    status = "fail"
    for s in status_list:
        if s == "success":
            status = s
    return status


def _merge_arcs(arcs_list) -> dict:
    """
    Internal function to merge arcs based on filename.
    :param arcs_list:
    :return: dict, the merged arcs
    """
    merged_arcs = {}
    for arcs in arcs_list:
        for file_path in arcs:
            if file_path not in merged_arcs:
                merged_arcs[file_path] = arcs[file_path]
            else:
                for ar in arcs[file_path]:
                    if ar not in merged_arcs[file_path]:
                        merged_arcs[file_path].append(ar)
    return merged_arcs


def catch_error_message(message: str) -> str:
    """
    Parse the message, get the error message.
    Example:
    message='
Traceback (most recent call last):
  File "/root/codes/testing/CodeCaller.py", line 182, in exec_code
    exec(new_code)
  File "<string>", line 3, in <module>
  File "/root/codes/catcher/VarCatcher.py", line 103, in wrapper
    res = func(*args, **kwargs)
  File "/opt/conda/envs/gpttest/lib/python3.9/site-packages/tensorflow/python/util/tf_export.py", line 408, in wrapper
    raise TypeError(
TypeError: reverse_v2 only takes keyword args (possible keys: ['tensor', 'axis', 'name']). Please pass these args as kwargs instead.'
    catch_error_message(message) = "TypeError: reverse_v2 only takes keyword args (possible keys: ['tensor', 'axis', 'name']). Please pass these args as kwargs instead."
    :param message: message to be parsed
    :return: the error message.
    """
    return message.rstrip("\n ").split('\n')[-1]


def get_args_list(signature: str) -> list[str]:
    """
    Extracts the list of argument names from an API signature.
    If signature starts with "torch.", we will omit args after '*' (if any).
    get_args_list("tf.raw_ops.ReverseV2(tensor,axis,name=None)") = ['tensor', 'axis']
    get_args_list("torch.bitwise_or(input, other, *, out=None)") = ['input', 'other']
    :param signature: the API signature, example: tf.raw_ops.ReverseV2(tensor,axis,name=None)
    :return:
    """
    try:
        node = ast.parse(signature).body[0].value
        args_list = [ast.unparse(arg) for arg in node.args]
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            assert isinstance(keyword.arg, str)
            args_list.append(keyword.arg)
    except:
        pattern = r"\((.*?)\)"
        match = re.search(pattern, signature)
        if match is None:
            return []
        args_str = match.group(1)
        args_list = [arg.strip().split("=")[0] for arg in args_str.split(",")]
    if "name" in args_list and args_list.index("name") == len(args_list) - 1:
        # We pop the name if it is indexed in the last position
        args_list.pop(args_list.index("name"))
    if "" in args_list:
        args_list.pop(args_list.index(""))
    if "*args" in args_list:
        args_list.pop(args_list.index("*args"))
    if signature.startswith("torch."):
        torch_args_list = []
        for arg in args_list:
            if arg == "*":
                break
            torch_args_list.append(arg)
        return torch_args_list
    return args_list


def get_default_value(signature: str) -> [str, str]:
    """
    For torch.arange and torch.range, their signatures are like this:
    torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    which is not parsable by the ast since the SyntaxError: positional argument follows keyword argument
    :param signature:
    :return:
    """
    try:
        signature = signature.replace("*,", "")
        node = ast.parse(signature).body[0].value
        default_value_dict = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            arg_name = keyword.arg if isinstance(keyword.arg, str) else ast.unparse(keyword.arg)
            if isinstance(keyword.value, ast.Constant):
                value_name = keyword.value.value
            elif isinstance(keyword.value, (ast.Call, ast.Attribute)):
                value_name = {"obj": ast.unparse(keyword.value)}
            elif isinstance(keyword.value, (ast.List, ast.Tuple)):
                value_name = {"obj": ast.unparse(keyword.value)}
            else:
                value_name = ast.unparse(keyword.value)
            default_value_dict[arg_name] = value_name
        return default_value_dict
    except SyntaxError:
        if signature.startswith("torch.arange") or signature.startswith("torch.range"):
            return {"start": 0, "step": 1}
        else:
            print(f"Error when parsing the signature: {signature}, the error is: {traceback.format_exc()}")
            return {}
    except:
        print(f"Error when parsing the signature: {signature}, the error is: {traceback.format_exc()}")
        return {}


def get_api_name(signature: str) -> str:
    api_name = signature.split("(")[0]
    return api_name

def get_pkg_name(signature: str) -> str:
    api_name = get_api_name(signature)
    pkg_name = ".".join(api_name.split(".")[:-1])
    return pkg_name

def save_file(content: str, file_path: str) -> None:
    base_dir = os.path.dirname(file_path)
    os.makedirs(base_dir, exist_ok=True)
    with open(file_path, "w") as file:
        file.write(content)


def monitor_memory_and_time(process, limit_bytes=5 * 1024 * 1024 * 1024, timeout_seconds=60):
    """
    # this function monitors the memory usage of the process,
    # if it exceeds the limit, terminate the process.
    :param process: the process in subprocess.Popen format
    :param limit_bytes: max RAM, default is 1G (1*1024*1024*1024)
    :param timeout_seconds: max time seconds, default is 1 minute (60s)
    :return: the error message (if any), the return code
    """
    main_pid = process.pid
    # print(f"Start running process: {main_pid}")
    start_time = time.time()
    while True:
        if process.poll() is not None:
            break
        try:
            main_process = psutil.Process(main_pid)
            memory_usage = main_process.memory_info().rss
            for c in main_process.children(recursive=True):
                memory_usage += c.memory_info().rss
            # print(f"Memory used: {memory_usage/1024/1024/1024}GB")
        except psutil.NoSuchProcess:
            break
        if memory_usage > limit_bytes:
            print("Memory limit exceeded! Terminating the process.")
            main_process = psutil.Process(main_pid)
            for c in main_process.children(recursive=True):
                os.kill(c.pid, signal.SIGTERM)
            os.kill(main_pid, signal.SIGTERM)
        if time.time() - start_time > timeout_seconds:
            print(f"Timeout reached! Terminating the process.")
            main_process = psutil.Process(main_pid)
            for c in main_process.children(recursive=True):
                os.kill(c.pid, signal.SIGTERM)
            os.kill(main_pid, signal.SIGTERM)
        time.sleep(0.5)
    error = process.stderr.read().decode('utf-8')
    # print(f"Process: {main_pid} is finished")
    return error, process.returncode

def string_startswith(string, key_list: [str]) -> bool:
    for key in key_list:
        if string.startswith(key):
            return True
    return False

def list_files_with_keyword(directory_path, keyword, ext: str = ""):
    file_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if not file.endswith(ext):
                continue
            if keyword in open(os.path.join(root, file), 'r').read():
                file_list.append(os.path.join(root, file))
    return file_list

def load_apis(package_name: str = "tf.linalg.", library_name: str = "tensorflow") -> list:
    """
    This is the benchmark function we used to filter out DL library APIs.
    :param package_name:
    :param library_name:
    :return:
    """
    if library_name == "tensorflow":
        api_path = "./data/tf_api_list.txt"
        skip_pref = json.load(open("./data/tf_skip_pref.json", "r"))
    elif library_name == "pytorch":
        api_path = "./data/torch_api_list.txt"
        skip_pref = json.load(open("./data/torch_skip_pref.json", "r"))
    else:
        raise NotImplementedError(f"API signature collection for library: {library_name} is not implemented.")
    with open(api_path, "r") as file:
        api_sig_list = file.read().strip().split("\n")
    filtered_sig_list = []
    for api_signature in api_sig_list:
        pkg_name = get_pkg_name(api_signature)
        api_name = get_api_name(api_signature)
        args_list = get_args_list(api_signature)
        if api_name.split(".")[-1][0].isupper() and not api_name.startswith("tf.raw_ops"):
            # we skip class except those in tf.raw_ops
            continue
        if api_signature.strip() == "" or (len(args_list) == 0 and not api_name.startswith("tf.raw_ops")):
            continue
        if string_startswith(api_name, skip_pref):
            continue
        if package_name == "" or pkg_name == package_name:
            filtered_sig_list.append(api_signature)
    return filtered_sig_list


def tf_api_to_signature(api_name: str) -> str:
    with open('./data/tf_api_list.txt') as file:
        content = file.read().rstrip().splitlines()
    for sig in content:
        if sig.startswith(f"{api_name}("):
            return sig

def torch_api_to_signature(api_name: str) -> str:
    with open('./data/torch_api_list.txt') as file:
        content = file.read().rstrip().splitlines()
    for sig in content:
        if sig.startswith(f"{api_name}("):
            return sig

def is_py_variable(node) -> bool:
    return isinstance(node, (ast.Attribute, ast.Name))

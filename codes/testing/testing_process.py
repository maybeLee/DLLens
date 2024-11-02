import os
import multiprocessing
import traceback
import json
import numpy as np
from codes.logger.logger import logger
from codes.counterpart.evaluate_counterpart import compare_res
from codes.catcher.constraint_catcher_from_error import ErrorConstraintCatcher
import time
import psutil
import signal
import pickle
from pathlib import Path
import re
from utils.utils import is_numeric
from codes.constraints.constraint_solver import Tensor
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
TERMINATE_SIGNAL = "terminate"


def replace_numbers_with_xxx(input_text):
    modified_text = re.sub(r'\d+', 'XXX', input_text)
    modified_text = re.sub(r'\d+\.\d+', 'XXX', modified_text)
    return modified_text


def child_process(conn):
    import tensorflow
    import tensorflow as tf
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    import numpy as np
    import importlib

    while True:
        # Receive data from the parent process
        parent_message = conn.recv()
        if parent_message == TERMINATE_SIGNAL:
            break
        else:
            code_path = parent_message
        with open(code_path, "r") as file:
            code = file.read().strip()
        namespace = {}
        err_msg = ""
        importlib.reload(tensorflow)
        importlib.reload(tf)
        importlib.reload(np)
        try:
            exec(code, {"pickle": pickle, "tensorflow": tensorflow, "tf": tf, "torch": torch, "F": F, "nn": nn, "np": np}, namespace)
            status = 0
        except Exception as e:
            max_length = 300
            err_msg = f"{e.__class__.__name__}: {str(e)[:max_length]}"
            status = 1
        # Send a response back to the parent
        res_dict = {"status": status, "err_msg": err_msg}
        conn.send(json.dumps(res_dict))


class MainTester(object):
    def __init__(self):
        # Create a communication pipe
        self.parent_conn, child_conn = multiprocessing.Pipe()
        self.child = multiprocessing.Process(target=child_process, args=(child_conn,))
        self.child.start()
        self.valid_count = 0
        self.error_dict = defaultdict(list)

    def launch_child_process(self):
        self.parent_conn, child_conn = multiprocessing.Pipe()
        self.child = multiprocessing.Process(target=child_process, args=(child_conn,))
        self.child.start()
        if self.child.is_alive():
            print(f"The process is successfully started, PID: {self.child.pid}")
        else:
            print("The process is not successfully started")

    def stop_child(self):
        try:
            self.parent_conn.send(TERMINATE_SIGNAL)
            self.child.join()
        except Exception as e:
            print(f"Fail to stop the process, the error message is: {e}")
        if not self.child.is_alive():
            print(f"The process is successfully stopped")
        else:
            print("WARNING!!! The process is not successfully stopped")
    
    def kill_child_process(self, pid):
        process = psutil.Process(pid)
        for c in process.children(recursive=True):
            os.kill(c.pid, signal.SIGKILL)
        os.kill(pid, signal.SIGKILL)
        s_time = time.time()
        while self.child.is_alive():
            if time.time() - s_time > 5:
                logger.info(f"Error! The process {pid} is not killed! after 5 seconds")
                return
            continue
        logger.info(f"Successfully killed the process {pid}, total time cost is: {time.time()-s_time} seconds")
        

    def _test(self, code_path, limit_bytes=5 * 1024 * 1024 * 1024, timeout_seconds=30, take_coffee=True):
        """
        :param code_path:
        :param limit_bytes: max RAM, default is 5G (5*1024*1024*1024)
        :param timeout_seconds: max time seconds, default is 1 minute (60s)
        :return: status: 0 for success, 1 for exception, others for timeout or crash
        """
        if (self.valid_count + 1) % 1000 == 0 and take_coffee is True:
            logger.info(f"Currently the same environment has been used for {self.valid_count} times. Take a coffee, generate a new one")
            self.stop_child()
            self.launch_child_process()
        self.valid_count += 1
        if not self.child.is_alive():
            logger.info(f"Error! the child process is killed! We regenerate one.")
            self.launch_child_process()
        self.parent_conn.send(code_path)
        child_pid = self.child.pid
        start_time = time.time()
        memory_exceed, timeout = False, False
        while (not memory_exceed) and (not timeout):
            if self.parent_conn.poll():
                break
            # if the child's process exceeds memory limits or time limits, we will kill it and stop receiving it.
            try:
                process = psutil.Process(child_pid)
                memory_usage = process.memory_info().rss
                for c in process.children(recursive=True):
                    memory_usage += c.memory_info().rss
                # print(f"Memory used: {memory_usage/1024/1024/1024}GB")
            except psutil.NoSuchProcess:
                continue
            except FileNotFoundError:
                continue
            if memory_usage > limit_bytes:
                logger.info(f"Memory limit exceeded! Terminating the process {child_pid}.")
                self.kill_child_process(child_pid)
                memory_exceed = True
                continue
            if time.time() - start_time > timeout_seconds:
                logger.info(f"Timeout reached! Terminating the process {child_pid}.")
                self.kill_child_process(child_pid)
                timeout = True
            time.sleep(0.00000001)
        # execution is finished, fetching the testing message
        if (not memory_exceed) and (not timeout):
            try:
                child_response = json.loads(self.parent_conn.recv())
            except EOFError:
                child_response = {"status": -100, "err_msg": "core dump happens!"}
        else:
            child_response = {"status": -15, "err_msg": "Memory Exceed or Timeout!"}
        return child_response['status'], child_response['err_msg']

    @staticmethod
    def _is_exception_bug(error_dict, api_name: str = "") -> bool:
        lib_name = "tensorflow" if api_name.startswith("tf.") else "pytorch"
        if error_dict[lib_name].strip() == "":
            return False
        elif ErrorConstraintCatcher.is_invalid_err_msg(error_dict[lib_name], api_name):
            return False
        else:
            return True

    def _is_not_duplicated_bug(self, api_name, error_dict) -> bool:
        for lib_name in error_dict:
            error_msg = error_dict[lib_name]
            if error_msg == "":
                continue
            error_msg = replace_numbers_with_xxx(error_msg)
            if error_msg in self.error_dict[api_name]:
                return False
            else:
                self.error_dict[api_name].append(error_msg)
                return True
        return False

    @staticmethod
    def _is_inconsistent_bug(argument_dict, outputs) -> bool:
        output_numeric = is_numeric(outputs['tensorflow'])
        if not output_numeric:
            return True
        else:
            # we check if the input is uint tensor, if so, we do not report the inconsistent bug
            for arg in argument_dict:
                if isinstance(argument_dict[arg], Tensor):
                    if argument_dict[arg].dtype in ['uint16', 'uint32', 'uint64']:  # pytorch does not support uint16,32,64
                        return False
            return True
    
    @staticmethod
    def _rank_inconsistent_level(argument_dict, outputs) -> str:
        has_tensor = False
        for arg in argument_dict:
            if isinstance(argument_dict[arg], Tensor):
                has_tensor = True
                if argument_dict[arg].set_inf or argument_dict[arg].set_nan:
                    return "inf_nan"
                elif argument_dict[arg].dtype in ['uint8', 'int8', 'bfloat16']:  # pytorch does not support uint16,32,64
                    return "low_precision"
        if has_tensor is True:
            return "normal_value"
        return "others"

    def bug_report(self, bug_report_dir: Path, argument_dict: dict, saved_lib: str, api_name: str, bug_type: str, code_path_dict, error_dict, outputs=None):
        """Report the bug based on bug type

        Args:
            bug_report_dir (str): the directory storing the bug report
            argument_dict (dict): test input used for analysis
            api_name (str): buggy api name
            bug_type (str): type of bug, including crash, exception, inconsistent
        """
        bug_type_dir = bug_report_dir / bug_type
        bug_api_dir = bug_type_dir / api_name  # e.g., ../bugs/crash/torch.sin
        if bug_type == "crash":
            bug_api_dir.mkdir(parents=True, exist_ok=True)
            bug_info = f"Message: \n{error_dict[saved_lib]}\n"
            code_path = code_path_dict[saved_lib]
            dest_path = bug_api_dir / (saved_lib+"_"+Path(code_path).name)  # e.g., ../bugs/crash/torch.sin/lib_name_xxx.py
            code = Path(code_path).read_text()
            Path(dest_path).write_text(f"'''\n{bug_info}\n'''\n"+code)
        elif bug_type == "exception":
            if not self._is_exception_bug(error_dict, api_name):
                return
            if not self._is_not_duplicated_bug(api_name, error_dict):
                return
            library_name = "tensorflow" if api_name.startswith("tf.") else "pytorch"
            error_message = error_dict[library_name]
            error_type = ""
            if "ValueError:" in error_message or "IndexError:" in error_message or "TypeError:" in error_message or "RuntimeError:" in error_message:
                # We may not consider these bugs as True Positives
                error_type = error_message.split(":")[0].strip() + "-"
            bug_api_dir.mkdir(parents=True, exist_ok=True)
            bug_info = f"TF Message: {error_dict['tensorflow']}\nTorch Message: {error_dict['pytorch']}\n"
            code_path_tf = code_path_dict["tensorflow"]
            code_path_torch = code_path_dict["pytorch"]
            dest_path = bug_api_dir / f"{error_type}{Path(code_path_tf).name}"  # e.g., ../bugs/exception/torch.sin/xxx.py
            code_tf = Path(code_path_tf).read_text().replace('\n', '\n    ')
            code_torch = Path(code_path_torch).read_text().replace('\n', '\n    ')
            content = f"""
'''
{bug_info}
'''
try:
    {code_tf}
except Exception as e:
    print("TF Exception: ", e)
try:
    {code_torch}
except Exception as e:
    print("Torch Exception: ", e)
"""
            Path(dest_path).write_text(content)
        elif bug_type == "inconsistent":
            if not self._is_inconsistent_bug(argument_dict, outputs):
                return
            bug_level = self._rank_inconsistent_level(argument_dict, outputs)
            bug_info = f"Message: \nOutput is: {outputs}\n"
            bug_api_dir.mkdir(parents=True, exist_ok=True)
            code_path_tf = code_path_dict["tensorflow"]
            code_path_torch = code_path_dict["pytorch"]
            dest_path = bug_api_dir / f"{bug_level}-{Path(code_path_tf).name}"  # e.g., ../bugs/inconsistent/torch.sin/xxx.py
            code_tf = Path(code_path_tf).read_text()
            code_torch = Path(code_path_torch).read_text()
            content = f"""
'''\n
{bug_info}
'''\n
{code_tf}
{code_torch}
"""
            Path(dest_path).write_text(content)

            

    def main_testing(self, code_path_list: list[str], runtime_path_list: list[str], api_name, argument_dict: dict, bug_dir, timeout=30) -> (int, [str, str]):
        tf_code_path, torch_code_path = code_path_list
        tf_runtime_path, torch_runtime_path = runtime_path_list
        tf_status, tf_err_msg = self._test(tf_code_path, timeout_seconds=timeout)
        torch_status, torch_err_msg = self._test(torch_code_path, timeout_seconds=timeout)
        err_msg_dict = {"tensorflow": tf_err_msg, "pytorch": torch_err_msg}
        code_path_dict = {"tensorflow": tf_code_path, "pytorch": torch_code_path}
        if tf_status != 0:
            print(f"Error message for tf: \n{tf_err_msg}\n")
        if torch_status != 0:
            print(f"Error message for torch: \n{torch_err_msg}\n")
        # After running the generated program, we then compare their result
        if tf_status != torch_status:
            logger.info(f"Inconsistent running status. TensorFlow: {tf_status}, PyTorch: {torch_status}")
            if tf_status == -100:
                self.bug_report(bug_dir, argument_dict, "tensorflow", api_name, "crash", code_path_dict, err_msg_dict)
            if torch_status == -100:
                self.bug_report(bug_dir, argument_dict, "pytorch", api_name, "crash", code_path_dict, err_msg_dict)
            if tf_status != -100 and torch_status != 100:
                self.bug_report(bug_dir, argument_dict, "both",  api_name, "exception", code_path_dict, err_msg_dict)
            res = 0
        elif tf_status == 0 and torch_status == 0:
            try:
                tf_res = np.load(tf_runtime_path, allow_pickle=True)
            except:
                tf_res = None
            try:
                torch_res = np.load(torch_runtime_path, allow_pickle=True)
            except:
                torch_res = None
            if not compare_res(tf_res, torch_res):
                self.bug_report(bug_dir, argument_dict, "both", api_name, "inconsistent", code_path_dict, err_msg_dict, 
                                {"tensorflow": tf_res, "pytorch": torch_res})
                logger.info(f"Inconsistent result on function: {api_name}")
                res = 0
            else:
                logger.info(f"Consistent result on function: {api_name}. Equivalent API Found!")
                res = 1
        else:
            logger.info(f"Both program crashes")
            res = -1
        return res,err_msg_dict 


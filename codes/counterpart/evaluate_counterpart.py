import os
import subprocess
import traceback

import numpy as np

from codes.catcher.type_catcher_from_doc import TypeCatcher
from codes.logger.logger import logger
from codes.prompt_text.sample_input_prefix_1 import sample_input_query_1
from codes.prompting.Message import Message
from codes.constraints.constraint_solver import Tensor
from utils.utils import clean_code, get_api_name, monitor_memory_and_time, deprecated
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def convert_to_numpy(array):
    import tensorflow as tf
    import torch
    if isinstance(array, (tf.Tensor, torch.Tensor)):
        if isinstance(array, torch.Tensor) and array.requires_grad is True:
            return array.detach().numpy()
        return np.array(array)
    else:
        return array


def sparse_to_dense(array):
    import tensorflow as tf
    import torch
    if isinstance(array, tf.SparseTensor):
        return tf.sparse.to_dense(array)
    elif isinstance(array, torch.Tensor) and array.is_sparse:
        return array.to_dense()
    else:
        return array


def compare_res(res1, res2) -> bool:
    if isinstance(res1, (list, tuple)) and isinstance(res2, (list, tuple)) and len(res1) == len(res2):
        for r1, r2 in zip(res1, res2):  # handle tuple and list
            comp_res = compare_res(r1, r2)
            if comp_res is False:
                return False
        return True
    res1 = sparse_to_dense(res1)
    res2 = sparse_to_dense(res2)

    res1 = convert_to_numpy(res1)
    res2 = convert_to_numpy(res2)
    res = False
    if isinstance(res1, np.ndarray) and isinstance(res2, np.ndarray):
        res1 = np.squeeze(res1).flatten()
        res2 = np.squeeze(res2).flatten()
        if res1.size != res2.size:
            return False
        if res1.shape != res2.shape:
            return False
    try:
        res = np.allclose(res1, res2, atol=1e-1, equal_nan=True)
    except:
        res = False
    return res


def load_and_compare(tf_runtime_path, torch_runtime_path, function_name, silent=False):
    """
    compare the result of two runtime path.
    :param tf_runtime_path:
    :param torch_runtime_path:
    :param function_name:
    :return: 1: consistent result, 0: inconsistent result
    """
    try:
        tf_res = np.load(tf_runtime_path, allow_pickle=True)
    except:
        tf_res = None
    try:
        torch_res = np.load(torch_runtime_path, allow_pickle=True)
    except:
        torch_res = None
    try:
        if not compare_res(tf_res, torch_res):
            if silent is False:
                logger.info(f"Inconsistent result on function: {function_name}")
            res = 0
        else:
            if silent is False:
                logger.info(f"Consistent result on function: {function_name}. Equivalent API Found!")
            res = 1
    except Exception:
        print(
            f"Error happens when comparing two results for function: {function_name}. The error message is: {traceback.format_exc()}")
        res = 0
    return res


def evaluate_function_synonyms(code_path_list: [str], runtime_path_list: [str], function_name) -> (int, [str, str]):
    """
    Differential testing on different libraries on the same function.
    See if we can use a main process to handle cold boot overhead.
    :param code_path_list: list of programs written in different libraries implementing the same function.
    :param runtime_path_list: list of runtime path storing each library's final result for differential testing.
    :param function_name: string, name of the function.
    :return: int, 1: equivalent; 0: not equivalent; -1: invalid input
    """
    # evaluate equivalent function
    for runtime_path in runtime_path_list:
        assert runtime_path.endswith(".npy")
    tf_code_path, torch_code_path = code_path_list
    tf_runtime_path, torch_runtime_path = runtime_path_list
    tf_py_bin = "/opt/conda/envs/tensorflow/bin/python" if 'HOSTNAME' in os.environ and os.environ[
        'HOSTNAME'] == "sccpu6.cse.ust.hk" else "python"
    torch_py_bin = "/opt/conda/envs/pytorch/bin/python" if 'HOSTNAME' in os.environ and os.environ[
        'HOSTNAME'] == "sccpu6.cse.ust.hk" else "python"
    tf_call = f"TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='' COVERAGE_FILE=.coverage {tf_py_bin} -m coverage run -a --rcfile ./config/tensorflow_coveragepy.conf --branch {tf_code_path}"
    torch_call = f"TF_CPP_MIN_LOG_LEVEL=3 MKL_THREADING_LAYER=GNU {torch_py_bin} {torch_code_path}"
    tf_process: subprocess.Popen = subprocess.Popen(tf_call, shell=True, stderr=subprocess.PIPE)
    tf_err_msg, tf_status = monitor_memory_and_time(tf_process)
    torch_process: subprocess.Popen = subprocess.Popen(torch_call, shell=True, stderr=subprocess.PIPE)
    torch_err_msg, torch_status = monitor_memory_and_time(torch_process)
    if tf_status != 0:
        print(f"Error message for tf: \n{tf_err_msg}\n")
    if torch_status != 0:
        print(f"Error message for torch: \n{torch_err_msg}\n")
    # After running the generated program, we then compare their result
    if tf_status != torch_status:
        logger.info(f"Inconsistent running status. TensorFlow: {tf_status}, PyTorch: {torch_status}")
        res = 0
    elif tf_status == 0 and torch_status == 0:
        res = load_and_compare(tf_runtime_path, torch_runtime_path, function_name)
    else:
        logger.info(f"Both program crashes")
        res = -1
    return res, {"tensorflow": tf_err_msg, "pytorch": torch_err_msg}


class InputConvertor(object):
    """
    This is a utility class that can convert inputs across different library format.
    """

    @staticmethod
    def convert_input(input_str: str, src_lib: str, dst_lib: str) -> str:
        if src_lib == "tensorflow" and dst_lib == "pytorch":
            return InputConvertor.convert_tf_to_torch(input_str)
        elif src_lib == "pytorch" and dst_lib == "tensorflow":
            return InputConvertor.convert_torch_to_tf(input_str)
        else:
            raise NotImplementedError(f"Not implemented to convert {src_lib} input to {dst_lib} input")

    @staticmethod
    def convert_np_to_tf(input_expr: str) -> str:
        pass

    @staticmethod
    def convert_np_to_torch(input_expr: str) -> str:
        pass

    @staticmethod
    def convert_tf_to_np(tf_input_expr: str) -> str:
        """
        Some rules applied when using tensorflow's input to test numpy program.
        This function is to help generating valid input for function.
        :param tf_input_expr:
        :return:
        """
        np_input_expr = tf_input_expr
        # rule 3: tf.dtype -> np.dtype.
        target_dtype_list: [str] = ['bfloat16', 'bool', 'complex128', 'complex64', 'double', 'float16', 'float32',
                                    'float64', 'half', 'int16', 'int32', 'int64', 'string']
        for dtype in target_dtype_list:
            if f"tf.{dtype}" in np_input_expr:
                np_input_expr = np_input_expr.replace(f"tf.{dtype}", f"np.{dtype}")
        return np_input_expr

    @staticmethod
    def convert_tf_to_torch(tf_input_expr: str) -> str:
        """
        Some rules applied when using tensorflow's input to test pytorch program.
        This function is to help generating valid input for pytorch function.
        :param tf_input_expr:
        :return:
        """
        # we add some rules to convert tensorflow's input to pytorch's input.
        # rule 1: tf.convert_to_tensor -> torch.tensor
        torch_input_expr = tf_input_expr.replace("tf.convert_to_tensor", "torch.tensor")
        # rule 2: tf.constant-> torch.tensor
        torch_input_expr = torch_input_expr.replace("tf.constant", "torch.tensor")
        torch_input_expr = torch_input_expr.replace("tf.Variable", "torch.tensor")
        torch_input_expr = torch_input_expr.replace("tf.TensorShape", "torch.tensor")
        torch_input_expr = torch_input_expr.replace("tf.dtypes.", "tf.")
        # rule 3: tf.dtype -> torch.dtype.
        for dtype in Tensor.dtype_list:
            if f"tf.{dtype}" in torch_input_expr:
                torch_input_expr = torch_input_expr.replace(f"tf.{dtype}", f"torch.{dtype}")
            if f"\'{dtype}\'" in torch_input_expr:
                torch_input_expr = torch_input_expr.replace(f"\'{dtype}\'", f"torch.{dtype}")
            if f"\"{dtype}\"" in torch_input_expr:
                torch_input_expr = torch_input_expr.replace(f"\"{dtype}\"", f"torch.{dtype}")
            if "uint" in dtype:  # special handling, pytorch does not support uint16/32
                torch_input_expr = torch_input_expr.replace(f"torch.{dtype}", f"torch.uint8")
        return torch_input_expr
    
    @staticmethod
    def remove_uint_in_input(input_expr: str) -> str:
        """
        Remove uint type in the input expression.
        :param input_expr:
        :return:
        """
        for dtype in Tensor.dtype_list:
            if "uint" in dtype:
                if f"\'{dtype}\'" in input_expr:
                    input_expr = input_expr.replace(f"\'{dtype}\'", f"\'float32\'")
                if f"\"{dtype}\"" in input_expr:
                    input_expr = input_expr.replace(f"\"{dtype}\"", f"\'float32\'")
                if f"torch.{dtype}" in input_expr:
                    input_expr = input_expr.replace(f"torch.{dtype}", f"torch.float32")
                if f"tf.{dtype}" in input_expr:
                    input_expr = input_expr.replace(f"tf.{dtype}", f"tf.float32")
        return input_expr

    @staticmethod
    def convert_torch_to_np(input_expr: str) -> str:
        pass

    @staticmethod
    def convert_torch_to_tf(torch_input_expr: str) -> str:
        """
        Some rules applied when using pytorch's input to test tensorflow program.
        This function is to help generating valid input for tensorflow function.
        :param torch_input_expr:
        :return:
        """
        # we add some rules to convert tensorflow's input to pytorch's input.
        # rule 1: torch.tensor -> tf.constant
        tf_input_expr = torch_input_expr.replace("torch.tensor", "tf.constant")
        # rule 2: torch.from_numpy -> tf.constant
        tf_input_expr = tf_input_expr.replace("torch.from_numpy", "tf.constant")
        # rule 3: tf.dtype -> torch.dtype.
        target_dtype_list: [str] = ['bfloat16', 'bool', 'complex128', 'complex64', 'double', 'float16', 'float32',
                                    'float64', 'half', 'int16', 'int32', 'int64', 'string']
        for dtype in target_dtype_list:
            if f"torch.{dtype}" in tf_input_expr:
                tf_input_expr = tf_input_expr.replace(f"torch.{dtype}", f"tf.{dtype}")
        return tf_input_expr


class CounterpartEvaluator(object):
    """
    This class is used to evaluate whether implementations inside a counterpart are equivalent.
    Example usage:
    '''
    evaluator = CounterpartEvaluator(test_counterpart)
    evaluator.generate_input()
    evaluation_result: bool = evaluator.evaluate()  # True or False
    '''
    """

    def __init__(self, ):
        self.counterpart = {}
        self.tf_input_expr: str = ""
        self.torch_input_expr: str = ""
        self.np_input_expr: str = ""
        self.scipy_input_expr: str = ""

    @staticmethod
    def _get_inputs(input_expr: str) -> list[str]:
        """
        Collect all inputs assigned from the input expression.
        :param input_expr:
        :return:
        """
        import ast
        input_list = []
        for assign_expr in input_expr.strip().split("\n"):
            try:
                node = ast.parse(assign_expr).body[0]
            except:
                continue
            if not isinstance(node, ast.Assign) and not isinstance(node, ast.AugAssign):
                continue
            if isinstance(node, ast.Assign):
                for var in node.targets:
                    input_list.append(ast.unparse(var))
            else:
                input_list.append(ast.unparse(node.target))
        input_list = list(set(input_list))  # the input_list should follow the exact order or function argument call.
        return input_list

    def instantiate_inputs(self, input_expr, lib_format):
        """
        Instantiate self.tf_input_expr, self.np_input_expr, self.torch_input_expr
        :param input_expr:
        :param lib_format:
        :return:
        """
        if lib_format == "tensorflow":
            self.tf_input_expr = input_expr
            self.np_input_expr = InputConvertor.convert_tf_to_np(input_expr)
            self.torch_input_expr = InputConvertor.convert_tf_to_torch(input_expr)
        elif lib_format == "pytorch":
            self.tf_input_expr = InputConvertor.convert_torch_to_tf(input_expr)
            self.np_input_expr = InputConvertor.convert_torch_to_np(input_expr)
            self.torch_input_expr = input_expr
        else:
            raise NotImplementedError(f"Not implemented input convertor for library: {lib_format}")

    def _generate_tf_code(self, runtime_path: str, arg_type: [str, str]) -> str:
        func_impl: str = self.counterpart['counterparts']['tensorflow']
        code = f"""
import tensorflow
import tensorflow as tf
import numpy as np
np.random.seed(1234)
{func_impl}\n
"""
        code += self.tf_input_expr + "\n"
        input_list = self._get_inputs(self.tf_input_expr)
        # for arg in input_list:
        #     if arg not in arg_type or arg_type[arg] is None or not isinstance(arg_type[arg], str):
        #         continue
        #     if arg in arg_type:
        #         if arg_type[arg].lower() == "[tensor]":
        #             code += f"{arg} = [tf.constant(i) for i in {arg}]\n"  # write in comprehension
        #         elif "tensor" in arg_type[arg].lower():
        #             code += f"{arg} = tf.constant({arg})\n"
        arg_list = []  # arg_list follow the specific order as counterpart's inputs
        for i in self.counterpart['inputs']:
            if i in input_list:
                arg_list.append(i)
        code += f"out = tensorflow_call({','.join(arg_list)})\n"
        code += f"import pickle\n"
        code += f"pickle.dump(out, open('{runtime_path}', 'wb'))\n"
        return code

    def _generate_torch_code(self, runtime_path: str, arg_type: [str, str]) -> str:
        func_impl: str = self.counterpart['counterparts']['pytorch']
        code = f"""
import torch
import torch.nn.functional as F
import numpy as np
np.random.seed(1234)
{func_impl}\n
"""
        code += self.torch_input_expr + "\n"
        input_list = self._get_inputs(self.torch_input_expr)
        # for arg in input_list:
        #     if arg not in arg_type or arg_type[arg] is None or not isinstance(arg_type[arg], str):
        #         continue
        #     if arg in arg_type:
        #         if arg_type[arg].lower() == "[tensor]":
        #             code += f"{arg} = [torch.tensor(i) for i in {arg}]\n"  # write in comprehension
        #         elif "tensor" in arg_type[arg].lower():
        #             code += f"{arg} = torch.tensor({arg})\n"
        arg_list = []  # arg_list follow the specific order as inputs
        for i in self.counterpart['inputs']:
            if i in input_list:
                arg_list.append(i)
        code += f"out = pytorch_call({','.join(arg_list)})\n"
        code += f"import pickle\n"
        code += f"pickle.dump(out, open('{runtime_path}', 'wb'))\n"
        return code

    def _generate_np_code(self, runtime_path: str, arg_type: [str, str]) -> str:
        func_impl: str = self.counterpart['counterparts']['numpy']
        code = f"""
import numpy as np
{func_impl}\n
"""
        code += self.np_input_expr
        code += f"out = numpy_call({','.join(self.counterpart['inputs'])})\n"
        code += f"np.save('{runtime_path}', out, allow_pickle=True)\n"
        return code

    def _generate_scipy_code(self, runtime_path: str, arg_type: [str, str]) -> str:
        func_impl: str = self.counterpart['counterparts']['scipy']
        code = f"""
import scipy
import numpy as np
{func_impl}\n
"""
        code += self.scipy_input_expr
        code += f"out = scipy_call({','.join(self.counterpart['inputs'])})\n"
        code += f"np.save('{runtime_path}', out, allow_pickle=True)\n"
        return code

    def evaluate(self, counterpart: dict):
        """
        Entry method, this method will evaluate the counterpart holds or not.
        Currently, this method will generate one input and test multiple implementations,
        compare the result to check if it has the same behavior.
        :param counterpart:
        :return:
        """
        self.counterpart = counterpart
        func_name = self.counterpart['function_name']
        if func_name.startswith("tf."):
            library_name = "tensorflow"
        elif func_name.startswith("torch."):
            library_name = "pytorch"
        else:
            raise NotImplementedError(
                f"API: {func_name} Is not implemented. Currently the code only support handling tensorflow API")
        arg_type = TypeCatcher.catch(api_signature=func_name, library_name=library_name)

        # it is possible that the input expression cannot be successfully generated, if so, we generate empty input expr for it
        try:
            input_expr = self.collect_sample_inputs_from_llm()
            self.instantiate_inputs(input_expr, library_name)
        except Exception:
            pass
        code_dict: [str, str] = {}
        code_path_list = []
        runtime_path_list = []
        api_name = get_api_name(func_name)
        for lib_name in self.counterpart["counterparts"]:
            file_path = f"runtime/{lib_name}_{api_name}_test.py"
            runtime_path = f"runtime/{lib_name}_{api_name}_test.npy"
            runtime_path_list.append(runtime_path)
            code_path_list.append(file_path)
            os.system(f"rm {file_path} {runtime_path}")
            if lib_name == "tensorflow":
                code_dict[lib_name] = self._generate_tf_code(runtime_path, arg_type)
            elif lib_name == "pytorch":
                code_dict[lib_name] = self._generate_torch_code(runtime_path, arg_type)
            elif lib_name == "numpy":
                code_dict[lib_name] = self._generate_np_code(runtime_path, arg_type)
            elif lib_name == "scipy":
                code_dict[lib_name] = self._generate_scipy_code(runtime_path, arg_type)
            else:
                raise NotImplementedError(f"The code generator for library: {lib_name} is not implemented")

            with open(file_path, "w") as file:
                file.write(code_dict[lib_name])
            logger.info(f"The {lib_name}'s code is: {code_dict[lib_name]}")
        logger.info(f"Start comparing result")
        res, err_msg = evaluate_function_synonyms(code_path_list, runtime_path_list, func_name)
        return res


if __name__ == "__main__":
    pass

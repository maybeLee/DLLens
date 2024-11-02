from __future__ import annotations
from utils.utils import get_default_value, get_api_name, get_args_list
import ast


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

class DynamicTypeCatcher(object):
    """
    Catch the counterpart argument type from the sample inputs.
    Example Usage:
    api_signature=""
    DynamicTypeCatcher().catch(
      {
        "function_name": "tf.nest.is_nested(seq)",
        "inputs": [
          "seq"
        ],
        "sample_inputs": "seq = [1, [2, [3, [4]]]]",
        "counterparts": {
          "tensorflow": "def tensorflow_call(seq):\n  return tf.nest.is_nested(seq)",
        }
      },
    ) -> ["list"]
    """

    @staticmethod
    def prepare_inspection_func(counterpart):
        """
        Construct the wrapper function to return the value of received arguments.
        :param counterpart:
        :return:
        """
        keywords: [str, str] = get_default_value(counterpart["function_name"])
        args_list = counterpart["inputs"]
        lib_name = list(counterpart["counterparts"].keys())[0]
        args_str_list = []
        for arg in args_list:
            if arg in keywords:
                if isinstance(keywords[arg], str):
                    args_str_list.append(f"{arg}=\'{keywords[arg]}\'")
                elif isinstance(keywords[arg], dict) and ['obj'] == list(keywords[arg].keys()):
                    args_str_list.append(f"{arg}={keywords[arg]['obj']}")
                else:
                    args_str_list.append(f"{arg}={keywords[arg]}")
            else:
                args_str_list.append(arg)
        return f"def {lib_name}_call({','.join(args_str_list)}):\n  return {','.join(args_list)}"

    @staticmethod
    def collect_inputs_from_counterpart(counterpart: dict, type: str) -> list[dict[str, object]]:
        """
        Collect the input arguments from the counterpart
        :param counterpart:
        :param type: sample_inputs or llm_inputs
        :return:
        """
        inputs = []
        for sample_input in counterpart[type]:
            obj_dict = {}  # {arg_name: obj, ...}
            args_list = counterpart["inputs"]
            lib_name = list(counterpart["counterparts"].keys())[0]
            inspect_func = DynamicTypeCatcher.prepare_inspection_func(counterpart)
            actual_input_list = _get_inputs(sample_input)
            sorted_input_list = []  # arg_list follow the specific order as counterpart's inputs
            for i in args_list:
                if i in actual_input_list:
                    sorted_input_list.append(i)

            if lib_name == "tensorflow":
                envir = """
import tensorflow
import tensorflow as tf
import numpy as np
"""
            else:
                envir = """
import torch
import torch.nn.functional as F
import numpy as np
"""
            code = f"""
{envir}
{inspect_func}
{sample_input}
{', '.join(args_list)} = {lib_name}_call({', '.join(sorted_input_list)})
"""
            namespace = {}
            try:
                exec(code, globals(), namespace)
            except Exception as e:
                # print(f"Error when collecting inputs from counterpart: {e}")
                continue
            for arg in args_list:
                obj_dict[arg] = namespace[arg]
            inputs.append(obj_dict)
        return inputs

    @staticmethod
    def catch(counterpart: dict) -> ([str, str], [str, any]):
        """
        Catch the type of counterpart's argument, also return the object of the argument
        Return:
        {
          "arg1": "type1",
          "arg2": "type2",
          ...
        },
        {
          "arg1": "object1",
          "arg2": "object2",
          ...
        }
        """
        obj_dict = {}
        sample_inputs: str = counterpart["sample_inputs"][0]
        args_list = counterpart["inputs"]
        # corner case for torch.range and torch.arange when its signature is syntax invalid
        if get_api_name(counterpart["function_name"]) in ["torch.range", "torch.arange"]:
            return {"start": "integer", "end": "integer", "step": "integer"}, {}
        if len(args_list) == 1 and args_list == ["*tensors"]:
            # we hard code the return type of *tensors as "tensor" for approximation
            return {"tensors": "tensor"}, {}
        if len(args_list) == 1 and args_list == ["*args"]:
            # we hard code the return type of *tensors as "tensor" for approximation
            return {"args": "tensor"}, {}
        lib_name = list(counterpart["counterparts"].keys())[0]
        inspect_func = DynamicTypeCatcher.prepare_inspection_func(counterpart)
        actual_input_list = _get_inputs(sample_inputs)

        sorted_input_list = []  # arg_list follow the specific order as counterpart's inputs
        for i in args_list:
            if i in actual_input_list:
                sorted_input_list.append(i)

        if lib_name == "tensorflow":
            envir = """
import tensorflow
import tensorflow as tf
import numpy as np
"""
        else:
            envir = """
import torch
import torch.nn.functional as F
import numpy as np
"""
        code = f"""
{envir}
{inspect_func}
{sample_inputs}
{', '.join(args_list)} = {lib_name}_call({', '.join(sorted_input_list)})
"""
        namespace = {}
        try:
            exec(code, globals(), namespace)
        except:
            return {}, {}
        type_dict = {}
        for arg in args_list:
            obj_dict[arg] = namespace[arg]
            type_str = str(type(namespace[arg]).__name__)

            def convert(origin_type: str) -> str:
                if origin_type in ["EagerTensor", "Tensor", "ndarray", "ResourceVariable"]:
                    tensor_obj = namespace[arg] if not isinstance(namespace[arg], (list,tuple)) else namespace[arg][0]
                    if "bool" in str(tensor_obj.dtype):
                        origin_type = "boolean_tensor"
                    else:
                        origin_type = "tensor"
                elif origin_type.lower() == "bool":
                    origin_type = "boolean"
                elif origin_type.lower() == "int":
                    origin_type = "integer"
                elif origin_type.lower() == "nonetype":
                    origin_type = "any"
                elif origin_type.lower() == "dtype":
                    origin_type = "string"
                elif origin_type.lower() == 'str':
                    origin_type = "string"
                elif origin_type.lower() == 'tuple':
                    origin_type = 'list'
                return origin_type

            type_str = convert(type_str)
            if type_str in ('list', 'tuple') and len(arg) > 0 and len(namespace[arg]) > 0:
                type_str = f"list_{convert(type(namespace[arg][0]).__name__)}"
            type_dict[arg] = type_str.lower()
            type_dict[arg] = type_str.lower()
        return type_dict, obj_dict

    @staticmethod
    def collect_inputstring_from_counterpart(counterpart: dict, type: str) -> list[dict[str, str]]:
        input_dict_list = []
        for sample_input in counterpart[type]:
            input_dict = {}
            for assign_expr in sample_input.strip().split("\n"):
                try:
                    node = ast.parse(assign_expr).body[0]
                except:
                    continue
                if not isinstance(node, ast.Assign) and not isinstance(node, ast.AugAssign):
                    continue
                if isinstance(node, ast.Assign):
                    if len(node.targets) > 1:
                        continue
                    input_dict[ast.unparse(node.targets[0])] = ast.unparse(node.value)
            input_dict_list.append(input_dict)
        return input_dict_list

if __name__ == "__main__":
    res = DynamicTypeCatcher.catch(
        {
            "function_name": "tf.nest.is_nested(seq)",
            "inputs": [
                "seq"
            ],
            "sample_inputs": "seq = [1, [2, [3, [4]]]]",
            "counterparts": {
                "tensorflow": "def tensorflow_call(seq):\n  return tf.nest.is_nested(seq)",
            }
        },
    )
    print(res)

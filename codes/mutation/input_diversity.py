# This script analyze the diversity of validation inputs
from __future__ import annotations
from codes.catcher.type_catcher_from_example import DynamicTypeCatcher
import random
import numpy as np
import json
import os


class Argument(object):
    def __init__(self, name=None, value=None):
        self.name = name
        self.value = value
        self.tf_sample_input = None
        self.torch_sample_input = None

    def load_sample_input(self, tf_sample_input: str, torch_sample_input: str):
        self.tf_sample_input = tf_sample_input
        self.torch_sample_input = torch_sample_input

    def instantiate_attributes(self):
        pass

class Tensor(Argument):
    dtype_list = [
        'bfloat16', 'bool', 'complex128', 'complex64',
        'double', 'float16', 'float32',
        'float64', 'half', 'int16', 'int32', 'int64', 'int8',
        # 'qint16', 'qint32', 'qint8', 'quint16', 'quint8',  # qint is not implemented
        'uint16', 'uint32', 'uint64', 'uint8'
    ]
    dtype_mapping = {
        "integer": ['int16', 'int32', 'int64', 'int8', 'uint16', 'uint32', 'uint64', 'uint8'],
        "float": ['bfloat16', 'float16', 'float32', 'float64', 'half'],
        "complex": ['complex128', 'complex64'],
        "boolean": ['bool'],
    }

    def __init__(self, name=None, ndims=None, shape_info: dict = None, dtype=None, num_element=None, value=None):
        """

        :param name:
        :param ndims:
        :param shape_info: shape_info is a dictionary specifying the necessary index's value, for instance, shape[2]=0 -> shape_info = {2:0}
        :param dtype:
        :param num_element
        :param value:
        """
        super().__init__(name, value)
        self.name = name
        self.ndims = ndims
        self.shape_info = {} if shape_info is None else shape_info
        self.shape = None
        self.dtype = dtype
        self.num_element = num_element
        self.value = value
        self.set_inf = random.random() < 0.05
        self.set_nan = random.random() < 0.05
        self.intmin = -10
        self.intmax = 10
        self.numpy_str = ""
        self.tf_sample_input = None
        self.torch_sample_input = None

    # def __repr__(self):
    #     return f"name: {self.name}, ndims={self.ndims}, shape_info={self.shape_info}, dtype={self.dtype}, value={self.value}"

    @staticmethod
    def solve_shape(ndims, shape_info, num_element) -> list[int]:
        """Given the number of dimensions, shape_info, and num_element, solve the shape of the tensor.

        Args:
            ndims (_type_): _description_
            shape_info (_type_): _description_
            num_element (_type_): _description_

        Returns:
            [int]: _description_
        """
        if num_element is None:
            # If the num_element is not specified, we encourage to generate diverse shape
            shape_list = np.random.randint(0, 5, ndims)
            for index, size in shape_info.items():
                shape_list[index] = size
            return shape_list
        shape_z3 = [z3.Int(f"shape_{i}") for i in range(ndims)]
        solver = z3.Solver()
        solver.add([0 <= shape_z3[i] for i in range(ndims)])
        for index, size in shape_info.items():
            solver.add(shape_z3[index] == size)
        if num_element is not None:
            solver.add(z3.Product(*shape_z3) == num_element)
        # Attempt to solve the constraints
        if solver.check() == z3.sat:
            model = solver.model()
            return [model[d].as_long() for d in shape_z3]
        else:
            print(f"WARNING Cannot solve the shape constraint! ndims: {ndims}, \
                shape_info: {shape_info}, num_element: {num_element}, \
                    we discard the constraint on num_element.")
            return Tensor.solve_shape(ndims, shape_info, None)

    def instantiate_attributes(self):
        # We first define ndims
        if self.ndims is None:
            min_ndim = 0
            if self.shape_info != {}:
                min_ndim = max(self.shape_info.keys()) + 1
            self.ndims = np.random.randint(
                min_ndim, 5) if min_ndim < 5 else min_ndim
        shape_list = self.solve_shape(
            self.ndims, self.shape_info, self.num_element)
        self.shape = list(shape_list)
        self.shape_info = {idx: shape for idx, shape in enumerate(shape_list)}
        self.num_element = int(np.prod(self.shape))
        if self.dtype is None:
            self.dtype = np.random.choice(self.dtype_list)
        self.numpy_str = self.numpy_array_str()

    def numpy_array_str(self):
        shape_list = [str(i) for i in self.shape]
        if "int" in self.dtype:
            # if we want to generate an integer tensor, we cannot use np.random.rand
            value = f"np.random.randint({self.intmin}, {self.intmax}, ({','.join(shape_list)}))"
            if self.set_inf is True:
                value = f"{value}*np.inf"
            elif self.set_nan is True:
                value = f"{value}*np.nan"
        elif "bool" not in self.dtype:
            value = f"np.random.randn({','.join(shape_list)})"
            if self.set_inf is True:
                value = f"{value}*np.inf"
            elif self.set_nan is True:
                value = f"{value}*np.nan"
        else:
            value = f"np.random.choice([True, False], size=({','.join(shape_list)}))"
        return value

    def build_tf_repr(self):
        if self.value is not None:
            return f"{self.name} = {self.value}"
        assert self.numpy_str != ""
        if self.use_sample_input and self.tf_sample_input is not None:
            return f"{self.name} = {self.tf_sample_input}"
        return f"{self.name} = tf.constant({self.numpy_str}, dtype='{self.dtype}')"

    def build_np_repr(self):
        if self.value is not None:
            return f"{self.name} = {self.value}"
        assert self.numpy_str != ""
        return f"{self.name} = np.array({self.numpy_str}, dtype='{self.dtype}')"

    def build_torch_repr(self):
        if self.value is not None:
            return f"{self.name} = {self.value}"
        assert self.numpy_str != ""
        shape_list = [str(i) for i in self.shape]
        if "uint" in self.dtype:
            dtype = "uint8"
        else:
            dtype = self.dtype
        if self.use_sample_input and self.torch_sample_input is not None:
            return f"{self.name} = {self.torch_sample_input}"
        return f"{self.name} = torch.tensor({self.numpy_str}, dtype=torch.{dtype})"

    def build_tf_obj(self):
        import tensorflow as tf
        if self.value is None:
            expr = self.build_tf_repr()
            exec(f"{expr}", globals(), locals())
            return locals()[self.name]
        else:
            if isinstance(self.value, tf.Tensor):
                return self.value
            elif isinstance(self.value, np.ndarray):
                return tf.constant(self.value)
            else:
                raise NotImplementedError(
                    f"data type of value {type(self.value)} is not implemented")

    def build_torch_obj(self):
        import torch
        if self.value is None:
            expr = self.build_torch_repr()
            exec(f"{expr}", globals(), locals())
            return locals()[self.name]
        else:
            import tensorflow as tf
            if isinstance(self.value, tf.Tensor):
                return torch.tensor(self.value.numpy())
            elif isinstance(self.value, np.ndarray):
                return torch.tensor(self.value)
            else:
                raise NotImplementedError(
                    f"data type of value {type(self.value)} is not implemented")

    def create_from_obj(self, obj):
        if str(type(obj).__name__) not in ["EagerTensor", "Tensor", "ndarray", "ResourceVariable", "int", "float", "bool", "list"]:
            raise NotImplementedError(
                f"Cannot create tensor from object type: {str(type(obj).__name__)}")
        if str(type(obj).__name__) in ["float", "bool", "list"]:  # convert it to tensor
            obj = np.array(obj)
        elif str(type(obj).__name__) == "int":
            obj = np.array(obj, dtype="int32")
        self.ndims = len(obj.shape)
        self.shape_info = {idx: list(obj.shape)[idx]
                           for idx in range(self.ndims)}
        self.dtype: str = obj.dtype.name if hasattr(
            obj.dtype, "name") else str(obj.dtype).split(".")[-1]
        if len(obj.shape) == 0:
            self.num_element = 1
        else:
            self.num_element = np.prod(list(obj.shape))


class Analyzer(object):
    def _init__(self):
        pass
    

class TensorAnalyzer(Analyzer):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.dtype = set()
        self.shape = set()
        self.ndims = set()
        self.num_element = set()
        self.name = name
    
    def analyze(self, obj, arg_type="") -> None:
        tensor = Tensor(name=self.name)
        tensor.create_from_obj(obj)
        self.dtype.add(str(tensor.dtype))
        self.shape.add(str(list(tensor.shape_info.values())))
        self.ndims.add(str(tensor.ndims))
        self.num_element.add(tensor.num_element)

    def report_diversity(self,):
        return {"dtype": len(self.dtype), "shape": len(self.shape), "ndims": len(self.ndims), "num_element": len(self.num_element)}
        # print(f"Number of unique dtypes for {self.name}: {len(self.dtype)}")
        # print(f"Number of unique shapes for {self.name}: {len(self.shape)}")
        # print(f"Number of unique ndims for {self.name}: {len(self.ndims)}")

class OtherAnalyzer(Analyzer):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.value = set()
    
    def analyze(self, obj, arg_type: str) -> None:
        type_str = str(type(obj).__name__)
        if type_str in ["EagerTensor", "Tensor", "ndarray", "ResourceVariable"]:
            if arg_type == "boolean":
                obj = bool(obj)
                type_str = "bool"
            elif arg_type == "integer":
                obj = int(obj)
                type_str = "int"
            elif arg_type == "float":
                obj = float(obj)
                type_str = "float"
        if type_str in ["bool", "int", "float", "str"]:
            self.value.add(obj)
    
    def report_diversity(self,):
        return {"value": len(self.value)}


analyzer_dict = {"tensor": TensorAnalyzer, "integer": OtherAnalyzer, "float": OtherAnalyzer, "boolean": OtherAnalyzer, "string": OtherAnalyzer}

def analyze_counterpart(counterpart: dict, input_type: str = "llm_inputs") -> dict[str, dict[str, int]]:
    """_summary_

    Args:
        counterpart (dict): counterpart to be analyzed
        input_type (str, optional): source of input, can be "sample_inputs" or "llm_inputs". Defaults to "llm_inputs".

    Returns:
        dict[str, dict[str, int]]: _description_
    """
    inputs = DynamicTypeCatcher.collect_inputs_from_counterpart(counterpart, input_type)
    lib_name = list(counterpart['counterparts'].keys())[0]
    type_dict, obj_dict = DynamicTypeCatcher.catch(counterpart)
    total = {}
    for arg_name in type_dict:
        arg_type = type_dict[arg_name]
        if arg_name not in obj_dict:
            continue
        if arg_type not in analyzer_dict:
            # print(f"Find unsupported type {arg_type}")
            continue
        analyzer = analyzer_dict[arg_type](arg_name)
        for sample_input in inputs:
            obj = sample_input[arg_name]
            if obj is None:
                continue
            analyzer.analyze(obj, arg_type)
        total[arg_name] = analyzer.report_diversity()
        if arg_type == "boolean" and total[arg_name]["value"] > 2:
            print(f"Find more than 2 boolean values for {arg_name}")
            breakpoint()
        total[arg_name]["argument_type"] = arg_type
    return total

if __name__ == "__main__":
    counterpart_dir_list = [
    "./data/working_dir/counterpart/augmented/tensorflow_pytorch_gpt-4o-mini_3_seeds/counterparts",
    "./data/working_dir/counterpart/augmented/pytorch_tensorflow_gpt-4o-mini_3_seeds/counterparts",
    ]
    input_type_list = ["sample_inputs", "llm_inputs"]

    def load_res(counterpart_dir):
        res = {}
        for file_name in os.listdir(counterpart_dir):
            if not file_name.endswith(".json"):
                continue
            api_name = file_name.split(".json")[0]
            print(f"Working on api: {api_name}")
            res[api_name] = {}
            file_path = os.path.join(counterpart_dir, file_name)
            with open(file_path, "r") as f:
                counterpart = json.load(f)
            for input_type in input_type_list:
                res[api_name][input_type] = analyze_counterpart(counterpart, input_type)
        return res

    res = load_res(counterpart_dir_list[0])

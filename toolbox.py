from pathlib import Path
import numpy as np
from codes.program_generator import ProgramGenerator
import os
import yaml
from utils.utils import load_apis, get_api_name, get_args_list, tf_api_to_signature, torch_api_to_signature
from codes.constraints.constraint_solver import ConstraintSolver
import json
from collections import defaultdict
from codes.mutation.input_diversity import analyze_counterpart


lib_list = ['pytorch', 'tensorflow']


def rq1_load_tensorscope_counterparts():
    tf2onnx = json.load(open("data/working_dir/rq1/tensorscope/tf2onnx.json", "r"))
    api_list = []
    for value in tf2onnx.values():
        for i in value:
            api_list.append(i)
    api_list = [f"tf.raw_ops.{i}" for i in api_list]
    t_tf2onnx = set(api_list)

    tf2paddle = json.load(open("data/working_dir/rq1/tensorscope/tf2paddle.json"))
    api_list = [f"tf.raw_ops.{i}" for i in tf2paddle.keys()]
    t_tf2paddle = set(api_list)

    tf2tflite = json.load(open("data/working_dir/rq1/tensorscope/tf2tflite.json"))
    tf2tflite = [f"tf.raw_ops.{i[1:]}" for i in tf2tflite]  # tf.raw_ops.kOP -> tf.raw_ops.OP
    t_tf2tflite = set(tf2tflite)

    torch2ms = json.load(open("data/working_dir/rq1/tensorscope/mindconverter2.json"))
    api_list = [i.replace("aten::", "torch.") for i in torch2ms.keys()]
    t_torch2ms = set(api_list)

    torch2onnx = json.load(open("data/working_dir/rq1/tensorscope/torch.onnx.export.json"))
    api_list = []
    for export_str in torch2onnx:
        api_name = export_str.split('@_onnx_symbolic(\"')[-1].rsplit("\")", 1)[0].split("\",decorate")[0]
        api_list.append(api_name.replace("aten::", "torch."))
    t_torch2onnx = set(api_list)

    torch2paddle = json.load(open("data/working_dir/rq1/tensorscope/torch2paddle.json"))
    api_list = [i for i in torch2paddle.keys() if i.startswith("torch.")]
    t_torch2paddle = set(api_list)

    t_tf = t_tf2onnx.union(t_tf2tflite).union(t_tf2paddle)
    t_torch = t_torch2ms.union(t_torch2onnx).union(t_torch2paddle)
    
    tf_total_list = load_apis("", "tensorflow")
    torch_total_list = load_apis("", "pytorch")

    tf_total_list = [get_api_name(i) for i in tf_total_list]
    torch_total_list = [get_api_name(i) for i in torch_total_list]

    t_tf = set([i for i in list(t_tf) if not i.startswith("tf.raw_ops.TFL_")])
    t_torch = set([i for i in list(t_torch) if i.startswith("torch.") and (not i.startswith("torch._"))])
    t_torch_target = []
    for i in list(t_torch):
        if i in torch_total_list:
            t_torch_target.append(i)
        else:
            find_api = False
            for api_name in torch_total_list:
                if api_name.split(".")[-1] == i.split(".")[-1]:
                    t_torch_target.append(api_name)
                    find_api = True
                    break
            if find_api == False:
                t_torch_target.append(i)
    t_torch = set(t_torch_target)
    return t_tf, t_torch

def rq1_load_unique_input_properties():
    counterpart_dir_list = [
    "./data/working_dir/rq1/dllens/tensorflow/counterparts",
    "./data/working_dir/rq1/dllens/pytorch/counterparts",
    ]
    input_type_list = ["sample_inputs", "llm_inputs"]

    def load_res(counterpart_dir):
        res = {}
        for file_name in os.listdir(counterpart_dir):
            if not file_name.endswith(".json"):
                continue
            api_name = file_name.split(".json")[0]
            # print(f"Working on api: {api_name}")
            res[api_name] = {}
            file_path = os.path.join(counterpart_dir, file_name)
            with open(file_path, "r") as f:
                counterpart = json.load(f)
            for input_type in input_type_list:
                res[api_name][input_type] = analyze_counterpart(counterpart, input_type)
        return res

    def check_difference(result, difference):
        for api_name, value in result.items():
            for input_type, arg_res in value.items():
                for arg_name, arg in arg_res.items():
                    argument_type = arg["argument_type"]
                    for attr in arg:
                        if attr == "argument_type":
                            continue
                        arg[attr] = 15 if arg[attr] > 15 else arg[attr]  # remove outlier
                    if argument_type == "tensor":
                        difference[input_type]["dtype"].append(arg["dtype"])
                        difference[input_type]["shape"].append(arg["shape"])
                        difference[input_type]["ndims"].append(arg["ndims"])
                        difference[input_type]["num_element"].append(arg["num_element"])
                    else:
                        difference[input_type][f"{argument_type}_value"].append(arg["value"])
        return difference
    result0 = load_res(counterpart_dir_list[0])
    result1 = load_res(counterpart_dir_list[1])

    difference = defaultdict(dict)
    difference = {
        "sample_inputs": {"dtype": [], "shape": [], "ndims": [], "num_element": [], "integer_value": [], "float_value": [], "boolean_value": [], "string_value": []}, 
        "llm_inputs": {"dtype": [], "shape": [], "ndims": [], "num_element": [], "integer_value": [], "float_value": [], "boolean_value": [], "string_value": []}
        }
    difference = check_difference(result0, difference)
    difference = check_difference(result1, difference)
    return difference


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")

def rq2_analyze_path_constraint(target_dir, name: str):
    target_dir = Path(target_dir)
    def get_cons_res(target_dir):
        api_cons_res = {}  # {"tf.math.sin": {"cons1": True, "cons2": False, ...}, ...}
        api_path_res = {}  # {"tf.math.sin": 10, ...}
        api_conds_per_path_res = {}  # {'tf.math.sin': 10, ...}
        for api_name in os.listdir(target_dir):
            if api_name.startswith('constraints') or api_name.endswith('.json') or api_name == "logs":
                continue
            cons_res = {}
            conds_per_path_list = []
            valid_path_cons = []
            if os.path.isdir(target_dir/api_name):
                # get number of path constraint per api
                tf_constraints: [[str]] = ProgramGenerator.load_constraints(target_dir/api_name/ "tensorflow")
                torch_constraints: [[str]] = ProgramGenerator.load_constraints(target_dir/api_name/ "pytorch")
                valid_path_cons = [sorted(t_cons) for t_cons in tf_constraints]
                for t_cons in torch_constraints:
                    t_cons = sorted(t_cons)
                    if t_cons not in valid_path_cons:
                        valid_path_cons.append(t_cons)
                conds_per_path_list = [len(path_cons) for path_cons in valid_path_cons]
            if len(conds_per_path_list) == 0:
                pass
            else:
                api_conds_per_path_res[api_name] = np.mean(conds_per_path_list)
            api_path_res[api_name] = len(valid_path_cons)
            api_cons_res[api_name] = cons_res
        return api_cons_res, api_path_res, api_conds_per_path_res

    def get_num_valid_cons(api_cons):
        # number of valid unique constraints in each API
        unique_cons_list = {}
        valid_cons_list = {}
        for api, cons in api_cons.items():
            unique_cons_list[api] = len(cons)
            valid_cons = 0
            valid_c_list = []
            for c in cons:
                if cons[c] == "True":
                    valid_cons += 1
                else:
                    pass
            valid_cons_list[api] = valid_cons
        unique_cons_count = list(unique_cons_list.values())
        return valid_cons_list

    def evaluate(api_cons, api_path, api_conds_per_path): 
        path_list = list(api_path.values())
        conds_per_path_list = list(api_conds_per_path.values())
        print(f"======== {name} ==========")
        print(f"Number of path constraints per api: {np.mean(path_list)}")
        print(f"Number of input constraints per path per api: {np.mean(conds_per_path_list)}\n")

    llm_api_cons, llm_api_path, llm_conds_per_path = get_cons_res(target_dir)
    llm_api_num_valid_cons = get_num_valid_cons(llm_api_cons)
    
    evaluate(llm_api_num_valid_cons, llm_api_path, llm_conds_per_path)

def rq2_evaluate_docter_constraint():
    num_cons_list = []
    dtype_constraint = {}
    structure_constraint = {}
    shape_constraint = {}
    value_constraint = {}
    for lib_name in lib_list:
        docter_dir = {
            "tensorflow": Path("data/working_dir/rq2/docter/constraints_extracted/tensorflow"),
            "pytorch": Path("data/working_dir/rq2/docter/constraints_extracted/pytorch"),
        }[lib_name]
        target_dir = {
            "tensorflow": Path("./data/working_dir/rq2/with-icf/tensorflow"),
            "pytorch": Path("./data/working_dir/rq2/with-icf/pytorch")
        }[lib_name]
        api_list = os.listdir(target_dir)
        for api_name in api_list:
            docter_cons_yaml = docter_dir / f"{api_name}.yaml"
            if os.path.exists(docter_cons_yaml):
                dtype_constraint[api_name] = 0
                structure_constraint[api_name] = 0
                shape_constraint[api_name] = 0
                value_constraint[api_name] = 0
                yaml_data = load_yaml(docter_cons_yaml)
                api_cons = yaml_data['constraints']
                for param in api_cons:
                    if param == "name":
                        continue
                    cons_dict = api_cons[param]
                    for field in cons_dict:
                        if field in ["dtype"]:
                            dtype_constraint[api_name] += 1
                        elif field in ["shape", "ndim"]:
                            shape_constraint[api_name] += 1
                        elif field in ["range", "enum"]:
                            value_constraint[api_name] += 1
                        elif field in ["structure", "tensor_t"]:
                            structure_constraint[api_name] += 1
                num_cons = dtype_constraint[api_name] + shape_constraint[api_name] + value_constraint[api_name] + \
                        structure_constraint[
                            api_name]
                num_cons_list.append(num_cons)
        print(f"Overall, DocTer can find constraints for {len(num_cons_list)} {lib_name} APIs")
    return dtype_constraint, shape_constraint, value_constraint, structure_constraint, num_cons_list

def rq2_evaluate_dllens_constraint():
    num_cons_list = []
    dtype_constraint = {}
    structure_constraint = {}
    shape_constraint = {}
    value_constraint = {}
    for lib_name in lib_list:
        docter_dir = {
            "tensorflow": Path("data/working_dir/rq2/docter/constraints_extracted/tensorflow"),
            "pytorch": Path("data/working_dir/rq2/docter/constraints_extracted/pytorch"),
        }[lib_name]
        target_dir = {
            "tensorflow": Path("./data/working_dir/rq2/with-icf/tensorflow"),
            "pytorch": Path("./data/working_dir/rq2/with-icf/pytorch")
        }[lib_name]
        sig_fetcher = {"tensorflow": tf_api_to_signature, 
                    "pytorch": torch_api_to_signature}
        api_list = os.listdir(target_dir)
        for api_name in api_list:
            docter_cons_yaml = docter_dir / f"{api_name}.yaml"
            if not os.path.exists(docter_cons_yaml):
                continue
            api_sig = sig_fetcher[lib_name](api_name)
            args_list = get_args_list(api_sig)
            dtype_constraint[api_name] = 0
            structure_constraint[api_name] = len(args_list)
            shape_constraint[api_name] = 0
            value_constraint[api_name] = 0
            our_cons: [[str]] = ProgramGenerator.load_constraints(target_dir/api_name/"tensorflow")
            our_cons += ProgramGenerator.load_constraints(target_dir/api_name/"pytorch")
            properties = ConstraintSolver.return_properties(our_cons)
            for pro in properties:
                if pro.endswith("_dtype") and pro.rsplit("_dtype",1)[0] in args_list:
                    dtype_constraint[api_name] += 1
                elif pro.endswith("_shape") or pro.endswith("_num_element") or pro.endswith("_ndims"):
                    if pro.rsplit("_shape", 1)[0] in args_list:
                        shape_constraint[api_name] += 1
                    elif pro.rsplit("_num_element", 1)[0] in args_list:
                        shape_constraint[api_name] += 1
                    elif pro.rsplit("_ndims", 1)[0] in args_list:
                        shape_constraint[api_name] += 1
                else:
                    if pro in args_list:
                        value_constraint[api_name] += 1
            num_cons = dtype_constraint[api_name] + shape_constraint[api_name] + value_constraint[api_name] + \
                        structure_constraint[
                            api_name]
            num_cons_list.append(num_cons)
    return dtype_constraint, shape_constraint, value_constraint, structure_constraint


def load_tool_cov(cov_path:str):
    with open(cov_path, 'r') as file:
        cov_data = file.read().strip().split("\n")
    cov_dict = {"tensorflow": [], "pytorch": []}
    for line in cov_data:
        lib_name, cov = line.split(":")
        cov_dict[lib_name].append(float(cov.strip()))
    return cov_dict

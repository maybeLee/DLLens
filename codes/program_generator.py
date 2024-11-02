from utils import utils
from pathlib import Path
from codes.mutation.input_diversity import Argument
import argparse
import ast
import inspect
import json
import os
import sys
import time
import traceback
from collections import defaultdict
import random

import numpy as np

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


VALID_CONS_LIMIT = 2000
SOLVABLE_CONS_LIMIT = 100


def code_add_res(code: str, runtime_path: str) -> str:
    new_code = f"{code}\nimport numpy as np\nnp.save('{runtime_path}', out, allow_pickle=True)\n"
    return new_code



class ConstraintSelector(object):
    class Constraint:
        def __init__(self, name: str, constraint: list[str]):
            self.constraint = constraint
            self.error_constraint = []
            self.cover_constraint = []
            self.empty_tensor_constraint = []
            self.sample_constraint = []
            self.name = name
            self.count = 0
        
        def get_constraint(self):
            self.cover_constraint = list(set(self.cover_constraint))
            self.error_constraint = list(set(self.error_constraint))
            self.empty_tensor_constraint = list(set(self.empty_tensor_constraint))
            while len(self.cover_constraint) >= 20:  # avoid too many constraints
                self.cover_constraint.remove(np.random.choice(self.cover_constraint))
            sample_constraint = self.sample_constraint if random.random() < sample_input_prob else []
            return self.constraint + self.error_constraint + self.cover_constraint + self.empty_tensor_constraint + sample_constraint
        
        def add_count(self):
            self.count += 1

        @property
        def score(self) -> float:
            return 1 / (1 + self.count)

    def __init__(self, constraints):
        self.constraints: list = [self.Constraint(name, constraint) 
                            for name, constraint in constraints.items()]

    @property
    def probabilities(self):
        _sum = sum([constraint.score for constraint in self.constraints])
        return [constraint.score / _sum for constraint in self.constraints]

    def pick_constraint(self):
        picked_constraint = np.random.choice(self.constraints, p=self.probabilities)
        picked_constraint.add_count()
        return picked_constraint
    

class ProgramGenerator(object):
    """
    Generate program based on the given constraints.
    We need to do two things:
    - one parser that can parse the valid constraint and then use constraint solver to solve an input instance.
    - generate the input for fuzzing different libraries.
    """

    def __init__(self, flags):
        # self.llm: ChatGPTCall = ChatGPTCall()
        self.i_start = flags.i_start
        self.i_end = flags.i_end

    @staticmethod
    def load_constraints(constraint_dir) -> list[str]:
        if not os.path.exists(constraint_dir):
            return []
        cons_list: list[[str]] = []
        for cons in os.listdir(constraint_dir):
            file_path = os.path.join(constraint_dir, cons)
            if os.path.isfile(file_path) and file_path.endswith(".txt"):
                with open(file_path, "r") as file:
                    cons: list[str] = file.read().split("Valid Constraint:")[-1].strip().split("Invalid Constraints:")[
                        0].strip().splitlines()
                cons = sorted(cons)
                if cons not in cons_list:
                    cons_list.append(cons)
        return cons_list

    # @utils.deprecated
    # def _constraint_to_program(self, constraint: str, signature: str) -> str:
    #     message = Message()
    #     args_list = utils.get_args_list(signature)
    #     api_name = utils.get_api_name(signature)
    #     query = f"Your task is to generate a program exemplifying the usage of API: {api_name}. " \
    #             f"To do so, you need to complete the following code. " \
    #             f"```\n" \
    #             f"import tensorflow as tf\n" \
    #             f"# Instantiate the input arguments: {','.join(args_list)}. Input constraint: {constraint} \n" \
    #             f"# Call {api_name}\n" \
    #             f"```\n" \
    #             f"When generating the input arguments, you also need to generate input arguments that follow the following constraints:\n" \
    #             f"```\n" \
    #             f"{constraint}\n" \
    #             f"```\n" \
    #             f"DO NOT add any assertion or the if branch on input arguments to validate these constraints. \n" \
    #             f"If the input is an image, use a randomly generated tensor instead. " \
    #             f"If the input is randomly generated, use numpy to generate the input and set the random seed to 1234. \n" \
    #             f"You also need to use the variable `out` to save the output of the result. " \
    #             f"Remember to quote the generated code with ``` symbol."
    #     message.update_query(query)
    #     answer = self.llm.ask_gpt(message.message)
    #     print(answer)
    #     code = utils.clean_code(answer)
    #     if isinstance(code, list):
    #         code = code[0]
    #     constraint = constraint.replace('\n', '')
    #     code += f"\n# {constraint}\n"
    #     return code

    @staticmethod
    def cons_to_arguments(constraints: dict, args_list: list[str], args_type: dict[str, str],
                          default_value: dict[str, str] = {}, sample_inputs: dict[str, str] = {}) -> dict[str, Argument]:
        """
        Core function that parse constraints, solve constraints and generate inputs based on constraints.
        constraint_to_input_expr({"image_dtype": "float32"}, ['image']) == "np.random.rand(*shape).astype('float32')"
        :param constraints: a dictionary specifying some specific values on the API argument attributes.
        example of the constraints: {"image_dtype": "float32"}
        :param args_list:
        :param args_type:
        :param default_value: if there are default value from string, and no constraint on string, we use it
        :return:
        """
        args_type_dict = args_type.copy()  # avoid changing the original dict
        argument_dict: dict[str, Argument] = {}

        def update_tensor(argument: str, attr: str, attr_value):
            # we first create a tensor instance if the argument_dict
            # tensor can be Tensor or a LIST of tensor
            tensor: Tensor = argument_dict[argument]  # type: ignore
            if attr == f"{argument}_dtype":
                tensor.dtype = attr_value
            elif attr.startswith(f"{argument}_shape_"):
                index = int(attr.split("shape_")[-1])
                tensor.shape_info[index] = attr_value
            elif attr == f"{argument}_ndims":
                tensor.ndims = attr_value
            elif attr == f"{argument}_num_element":
                tensor.num_element = value
            elif attr == f"{argument}_intmin":
                tensor.intmin = value
            elif attr == f"{argument}_intmax":
                tensor.intmax = value
            elif attr == argument:
                # this branch may not be reached
                tensor.value = attr_value
            else:
                raise ValueError(f"Invalid attribute: {attr}")

        # infer argument type by argument constraint:
        valid_attr_name = list(inspect.signature(Tensor).parameters.keys())
        # Add another rule: if all sample inputs for an argument is scalar tensor, we set the argument's dtype to be integer.
        for attr in constraints.keys():
            for attr_name in valid_attr_name:
                if attr.endswith(attr_name) and attr[:-len(attr_name) - 1] in args_type_dict:
                    arg_name = attr[:-len(attr_name) - 1]
                    origin_type = args_type_dict[arg_name]
                    # We handle special case for list type
                    origin_type = origin_type.split(
                        "_")[-1] if origin_type.startswith("list_") else origin_type
                    if origin_type in ["boolean_tensor", "tensor"]:
                        continue
                    # if original dtype is integer and constraint requires value is integer, we need to convert it to int
                    # same for other scalars
                    if origin_type in ["integer", "boolean", "float"]:
                        if attr_name == "dtype" and \
                                constraints[attr] in Tensor.dtype_mapping[origin_type]:
                            continue
                        if attr_name == "num_element" and str(constraints[attr]) == "1":
                            continue
                        if attr_name == "ndims" and str(constraints[attr]) == "0":
                            continue
                    args_type_dict[arg_name] = "tensor"
        for arg in args_list:
            if arg in default_value and isinstance(default_value[arg], str):  # if default value's type is string, we change arg type to string
                args_type_dict[arg] = "string"
            arg_type = args_type_dict[arg].lower()
            if arg_type == "tensor":
                argument_dict[arg] = Tensor(name=arg)
            elif arg_type == "boolean_tensor":
                argument_dict[arg] = Tensor(name=arg, dtype="bool")
            elif arg_type == "boolean" or arg_type == "bool":
                argument_dict[arg] = Boolean(name=arg)
            elif arg_type == "int" or arg_type == "integer":
                argument_dict[arg] = Int(name=arg)
            elif arg_type == 'float':
                argument_dict[arg] = Float(name=arg)
            elif arg_type == "string":
                if arg in default_value:
                    if isinstance(default_value[arg], str):
                        argument_dict[arg] = String(name=arg, value=default_value[arg])
                    # we discard argument if it's default value is not a string
                else:
                    argument_dict[arg] = String(name=arg)
            elif arg_type == "any":
                argument_dict[arg] = NONE(name=arg)
            elif arg_type.startswith("list_"):
                element_type = arg_type.split("list_")[-1]
                argument_dict[arg] = LIST(name=arg, element_type=element_type)
            else:
                raise NotImplementedError(
                    f"The argument type {arg_type} is not implemented.")
        for attr, value in constraints.items():
            if attr in args_list and attr in argument_dict:
                if argument_dict[attr] != "tensor":
                    argument_dict[attr].value = value
            else:
                # constraint on list tensor should be propagated to all tensors in the list
                mutable = False
                for arg in args_list:
                    if args_type_dict[arg] in ["tensor", "boolean_tensor", "list_tensor"]:
                        if f"{arg}_dtype" == attr or f"{arg}_shape" == attr or (attr.startswith(
                            f"{arg}_shape_") and attr.split("_")[-1].isdigit()) or \
                                f"{arg}_ndims" == attr or f"{arg}_num_element" == attr \
                                or f"{arg}_intmin" == attr or f"{arg}_intmax" == attr:
                            update_tensor(arg, attr, value)
                            mutable = True
                if mutable is False:
                    print(
                        f"Find tensor-related constraints on non-tensor variable: {attr}")

        for arg in argument_dict:
            argument_dict[arg].instantiate_attributes()
            if arg not in sample_inputs:
                continue
            lib_name = "tensorflow" if "tf" in sample_inputs[arg] else "pytorch"
            tf_sample_input = sample_inputs[arg] if lib_name == "tensorflow" else InputConvertor.convert_torch_to_tf(sample_inputs[arg])
            torch_sample_input = sample_inputs[arg] if lib_name == "pytorch" else InputConvertor.convert_tf_to_torch(sample_inputs[arg])
            argument_dict[arg].load_sample_input(tf_sample_input, torch_sample_input)
        return argument_dict

    @staticmethod
    def cons_to_program(solution: dict, argument_dict: dict, args_list: list[str], library_impl: dict, cons_str):
        code_list = {}
        np_seed = np.random.randint(0, 100)
        envir_str = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed({np_seed})
"""
        for library_name, func_impl in library_impl.items():
            code = ""
            if library_name == "tensorflow":
                code = f"""
{envir_str}\n
import tensorflow
import tensorflow as tf
{func_impl}\n
"""
            elif library_name == "pytorch":
                code = f"""
{envir_str}\n
import torch
import torch.nn.functional as F
{func_impl}\n
"""
            input_expr = f"# Constraint Expression: {cons_str}\n"
            input_expr += f"# Solved constraints: {solution}\n"
            for arg in argument_dict.values():
                if library_name == "tensorflow":
                    input_expr += arg.build_tf_repr() + "\n"
                elif library_name == "pytorch":
                    input_expr += arg.build_torch_repr() + "\n"

            code += input_expr + "\n"
            necessary_args_list = [arg for arg in args_list if arg in argument_dict]
            code += f"out = {library_name}_call({','.join(necessary_args_list)})\n"
            if library_name == "pytorch":
                code += """
if type(out).__name__ == "Tensor" and out.dtype == torch.bfloat16:
    out = out.to(torch.float)
"""
            elif library_name == "tensorflow":
                code += """
if type(out).__name__ in ["EagerTensor", "Tensor", "ndarray"] and out.dtype == tf.bfloat16:
    out = tf.cast(out, tf.float32)
"""
            code += "print(out)\n"
            code_list[library_name] = code
        return code_list['tensorflow'], code_list['pytorch']

    @staticmethod
    def is_arg_type_valid(args_type: dict[str, str]) -> bool:
        # check if the collected argument types are valid.
        valid = True
        for arg in args_type:
            arg_type = args_type[arg].lower()
            if arg_type not in ['tensor', 'boolean', 'int', 'integer', 'float', 'string', 'any', 'boolean_tensor']:
                if arg_type.startswith("list_") and arg_type.split("list_")[-1] in ['tensor', 'boolean', 'int', 'integer',
                                                                                    'float', 'string']:
                    continue
                valid = False
                print(
                    f"The argument type {arg_type} of {arg} is invalid, we skip this API.")
        return valid

    @staticmethod
    def is_all_tensor(args_type: dict[str, str]) -> bool:
        # check if the collected argument types are tensors.
        all_tensor = True
        for arg in args_type:
            arg_type = args_type[arg].lower()
            if arg_type != "tensor":
                all_tensor = False
                print(
                    f"The argument type {arg_type} of {arg} is not all tensor, we skip this API.")
        return all_tensor

    @staticmethod
    def _get_solvable_constraints(valid_cons: list[[str]], args_type: dict[str, str]) -> list[[str]]:
        """
        Filter the solvable constraints from the valid constraints
        :param valid_cons: total valid constraints to be filtered.
        :param args_type: type of the arguments for solving.
        :return:
        """
        solved_cons = []
        for cons in valid_cons:
            if len(cons) == 0:
                continue
            # print(f"Solving path constraint: {cons}")
            if ConstraintSolver(args_type_dict=args_type).solve(cons) == {}:
                # print(f"The constraint: {cons} cannot be solved")  # [DEBUG]
                continue
            if cons not in solved_cons:
                solved_cons.append(cons)
        return solved_cons

    @staticmethod
    def _create_program_structure(api_input_dir) -> (str, str, str):
        os.makedirs(api_input_dir, exist_ok=True)
        tf_save_dir = api_input_dir / "tensorflow"
        os.makedirs(tf_save_dir, exist_ok=True)
        torch_save_dir = api_input_dir / "pytorch"
        os.makedirs(torch_save_dir, exist_ok=True)
        runtime_dir = api_input_dir / "runtime"
        os.makedirs(runtime_dir, exist_ok=True)
        runtime_constraint_dir = api_input_dir / "constraints"
        os.makedirs(runtime_constraint_dir, exist_ok=True)
        return tf_save_dir, torch_save_dir, runtime_dir, runtime_constraint_dir

    @staticmethod
    def _check_and_remove_runtime(runtime_path):
        if os.path.exists(runtime_path):
            os.system(f"rm {runtime_path}")

    def construct_counter_list(self, counterpart_dir: str, api_source: str) -> list[[]]:
        total_counter_list = os.listdir(counterpart_dir)
        if self.i_start is not None and self.i_end is not None:
            total_counter_list = total_counter_list[self.i_start: self.i_end]
        counter_list = []
        for counterpart in total_counter_list:
            counterpart = json.load(
                open(os.path.join(counterpart_dir, counterpart), 'r'))
            api_sig = counterpart['function_name']
            api_name = utils.get_api_name(api_sig)
            prefix_mapping = {"tf.": "tensorflow", "torch.": "pytorch"}
            library_name = prefix_mapping.get(
                api_name.split(".")[0] + ".", None)
            if library_name is None:
                raise ValueError(
                    "Invalid func name! Either tf. or torch. is allowed.")
            if api_source == "sampled":
                sample_api_path = {"tensorflow": "./data/samples/rq4_tf_100.txt",
                                   "pytorch": "./data/samples/rq4_torch_100.txt"}
                sample_api_list = open(
                    sample_api_path[library_name], "r").read().split("\n")
                sample_api_list = [utils.get_api_name(
                    _sig) if "(" in _sig else _sig for _sig in sample_api_list]
                if api_name not in sample_api_list:
                    continue
            elif api_source.startswith("torch.") or api_source.startswith("tf."):
                # it is a specific name
                if api_name != api_source:
                    continue
            counter_list.append(counterpart)
        final_counter_list = []
        for counterpart in counter_list:
            api_sig = counterpart['function_name']
            api_name = utils.get_api_name(api_sig)
            prefix_mapping = {"tf.": "tensorflow", "torch.": "pytorch"}
            library_name = prefix_mapping.get(
                api_name.split(".")[0] + ".", None)
            args_list = utils.get_args_list(api_sig)
            args_list = [arg.replace("*tensors", "tensors")
                         for arg in args_list]
            if self.has_invalid_arg(args_list):
                continue
            try:
                args_type, obj_dict = DynamicTypeCatcher.catch(counterpart)
            except Exception as e:
                print(
                    f"Failed to catch the argument type for {api_name}, the error is: {e}, we skip it.")
                continue
            if self.is_arg_type_valid(args_type) is False:
                continue
            final_counter_list.append(counterpart)
        return final_counter_list

    @staticmethod
    def distribute_programs(total_programs, num_constraints):
        base_programs = total_programs // num_constraints
        extra_programs = total_programs % num_constraints
        programs_per_constraint = [
            base_programs + (1 if i < extra_programs else 0) for i in range(num_constraints)]
        return programs_per_constraint

    @staticmethod
    def has_invalid_arg(args_list: list[str]) -> bool:
        for arg in args_list:
            if arg.startswith("*shape"):  # this is ad-hoc
                print(
                    f"WARNING!!! The argument {arg} is not supported, we skip this API.")
                return True
        return False

    def run(self, constraints_dir: str = "./data/working_dir/constraints",
            counterpart_dir: str = "./data/working_dir/counterpart/tensorflow_torch_gpt35_3_seeds/counterparts",
            program_dir: str = "./data/working_dir/test_programs", num_program=2000, total_time=60*60, api_source="sampled"):
        """
        Load the current constraint, parse the constraint, generate test program for both tensorflow and pytorch fuzzing
        :param constraints_dir:
        :param counterpart_dir:
        :param program_dir:
        :param num_program:
        :return:
        """
        constraints_dir = Path(constraints_dir)  # type: ignore
        program_dir = Path(program_dir)  # type: ignore
        bug_dir = program_dir / "bugs"  # type: ignore
        counter_list = self.construct_counter_list(counterpart_dir, api_source)
        main_tester = MainTester()
        total_start_time = time.time()
        time_limit_per_api = total_time / len(counter_list)
        total_time_generation = 0
        total_time_constraint_solving = 0
        total_time_testing = 0
        for counterpart in counter_list:
            api_sig = counterpart['function_name']
            api_name = utils.get_api_name(api_sig)
            prefix_mapping = {"tf.": "tensorflow", "torch.": "pytorch"}
            library_name = prefix_mapping.get(
                api_name.split(".")[0] + ".", None)
            print(f"===== Working on api: {api_sig} =====")
            api_input_dir = program_dir / api_name  # type: ignore
            api_constraint_dir = constraints_dir / api_name  # type: ignore
            default_value: dict[str, str] = utils.get_default_value(api_sig)
            args_list = utils.get_args_list(api_sig)
            args_list = [arg.replace("*tensors", "tensors")
                         for arg in args_list]
            if self.has_invalid_arg(args_list):
                continue
            try:
                args_type, obj_dict = DynamicTypeCatcher.catch(counterpart)
            except Exception as e:
                print(
                    f"Failed to catch the argument type for {api_name}, the error is: {e}, we skip it.")
                continue
            if self.is_arg_type_valid(args_type) is False:
                continue
            # {"tensorflow": "def tensorflow_call(x, y):", "pytorch": "def pytorch_call(x, y):"}
            library_impl = counterpart['counterparts']
            tf_save_dir, torch_save_dir, runtime_dir, runtime_constraint_dir = self._create_program_structure(
                api_input_dir)
            tf_constraints: list[[str]] = self.load_constraints(
                api_constraint_dir / "tensorflow")
            torch_constraints: list[[str]] = self.load_constraints(
                api_constraint_dir / "pytorch")
            valid_cons = []  # fix minor list override issue, won't influence results
            for tf_cons in tf_constraints:
                tf_cons = sorted(tf_cons)
                if tf_cons not in valid_cons:
                    valid_cons.append(tf_cons)
            for t_cons in torch_constraints:
                t_cons = sorted(t_cons)
                if t_cons not in valid_cons:
                    valid_cons.append(t_cons)
            # Extract possible constraints from validation inputs
            validation_constraints = SampleConstraintCatcher(counterpart, args_type).extract_constraints()
            sample_inputs = SampleConstraintCatcher(counterpart, args_type).extract_sample_inputs()
            print(f"The extracted validation constraints are: \n{validation_constraints}")
            print(f"The extracted sample objects are: \n{sample_inputs}")
            if [] not in valid_cons:
                valid_cons += [[]]
            print(f"Total solvable constraints: {len(valid_cons)}")
            # For each API, we apply the roulette wheel selection to pick the constraint
            selector = ConstraintSelector({f"{i}": cons for i, cons in enumerate(valid_cons)})
            num_valid_input = 0
            i = 0
            # Add validation constraints to constraint selector
            for constraint in selector.constraints:
                # only add the validation constraint to original constraint if it won't break existing PCs.
                if ConstraintSolver(args_type_dict=args_type).solve(constraint.constraint + validation_constraints) != {}:
                    constraint.sample_constraint = validation_constraints
            s_time = time.time()
            while time.time() - s_time < time_limit_per_api and i < num_program:
                si_time = time.time()
                picked_cons = selector.pick_constraint()
                cons_in_this_run = picked_cons.get_constraint()
                print("============== Start Solving The Constraint =================")
                # print(f"The constraint is: {cons_in_this_run}")
                solution: dict = ConstraintSolver(
                    args_type_dict=args_type).solve(cons_in_this_run)
                total_time_constraint_solving += time.time() - si_time
                if solution == {}:
                    print(f"[WARNING] Cannot solve the constraint, we have to randomly generate the argument")
                print(f"============== Start Generating The Program ==================")
                argument_dict = self.cons_to_arguments(solution, args_list, args_type,
                                                        default_value, sample_inputs)  # instantiate arguments
                tf_code, torch_code = self.cons_to_program(
                    solution, argument_dict, args_list, library_impl, cons_in_this_run)
                gen_end_time = time.time()
                print(f"Time spent in generation: {gen_end_time - si_time}")
                total_time_generation += gen_end_time - si_time
                print("============== Start Testing The Program =====================")
                program_name = f'{picked_cons.name}_code{picked_cons.count}'
                tf_code_path = tf_save_dir / f"{program_name}.py"
                torch_code_path = torch_save_dir / f"{program_name}.py"
                tf_runtime_path = os.path.join(runtime_dir, f"tf_{program_name}.npy")
                torch_runtime_path = os.path.join(runtime_dir, f"torch_{program_name}.npy")
                tf_code = code_add_res(tf_code, runtime_path=tf_runtime_path)
                torch_code = code_add_res(torch_code, runtime_path=torch_runtime_path)
                utils.save_file(torch_code, torch_code_path)
                utils.save_file(tf_code, tf_code_path)

                print(f"Running code: {tf_code_path}, {torch_code_path}")
                print(f"Runtime path is: ", tf_runtime_path, " ", torch_runtime_path)
                _, error_msg_dict = main_tester.main_testing([tf_code_path, torch_code_path],
                                                                [tf_runtime_path,
                                                                    torch_runtime_path],
                                                                api_name=api_name,
                                                                argument_dict=argument_dict,
                                                                bug_dir=bug_dir,
                                                                timeout=5)
                print(f"Time spent in testing: {time.time() - gen_end_time}")
                total_time_testing += time.time() - gen_end_time
                self._check_and_remove_runtime(tf_runtime_path)
                self._check_and_remove_runtime(torch_runtime_path)
                if error_msg_dict[library_name] == "":
                    num_valid_input += 1
                cover_list = []
                for name in argument_dict:
                    if isinstance(argument_dict[name], (Float, Int, Boolean, String)):
                        cover_list.append(f"({name} == {argument_dict[name].value})")
                        continue
                    elif isinstance(argument_dict[name], Tensor):
                        tensor: Tensor = argument_dict[name]  # type: ignore
                        cover_list.append(f"({name}.ndims == {tensor.ndims} and {name}.dtype == '{tensor.dtype}')")
                        if tensor.num_element == 0:
                            picked_cons.empty_tensor_constraint.append(f"assert ({name}.num_element != 0) is True")
                if cover_list != []:
                    cover_condition = " and ".join(cover_list)
                    picked_cons.cover_constraint.append(f"assert ({cover_condition}) is False")
                for lib_name, err_msg in error_msg_dict.items():
                    try:
                        cons_from_error = ErrorConstraintCatcher.catch(argument_dict, err_msg)
                    except Exception:
                        print(f"Error when handling the error message, need to be refined.")
                        print(traceback.format_exc())
                        cons_from_error = ""
                    if cons_from_error.strip() != "":
                        print(f"Extracted constraint is: {cons_from_error}")
                        # We need to make sure this constraint will not make the whole constraint set solvable before we adding it.
                        _cons_in_this_run = list(
                            cons_in_this_run + [cons_from_error])
                        if solution != {} and ConstraintSolver(
                            args_type_dict=args_type).solve(_cons_in_this_run) == {}:
                            print(f"The extracted constraint has conflict with existing ones, we discard it.")
                            continue
                        cons_from_error_path = runtime_constraint_dir / \
                            f"{program_name}_{lib_name}.txt"
                        utils.save_file(
                            err_msg + "\n" + cons_from_error, cons_from_error_path)
                        picked_cons.error_constraint.append(cons_from_error)
                i += 1
            if num_valid_input == 0:
                print(
                    f"WARNING!!! No Valid Input Can be Generated For API: {api_name}")
        print(f"Total Time Spent In Generating Programs: {total_time_generation} Seconds")
        print(f"Total Time Spent In Solving Constraints: {total_time_constraint_solving} Seconds")
        print(f"Total Time Spent In Testing: {total_time_testing} Seconds")
        print(f"Total Time Spent: {time.time() - total_start_time} Seconds")
        main_tester.stop_child()


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--api_source", type=str, default="sampled",
                       help="The source of API. It can be sampled, all, or a specific API name")
    parse.add_argument("--library_name", type=str,
                       default="tensorflow", help="The name of library")
    parse.add_argument("--constraints_dir", type=str, default="./data/working_dir/constraints/rules/",
                       help="the directory of collected constraints")
    parse.add_argument("--counterpart_dir", type=str,
                       default="./data/working_dir/counterpart/pytorch_tensorflow_gpt35_3_seeds/counterparts",
                       help="The directory of saved constraints")
    parse.add_argument("--program_dir", type=str,
                       default="./data/working_dir/test_programs/rules")
    parse.add_argument("--time_limit", type=int, default=60*60, help="The time limit for generating programs")
    parse.add_argument("--num_program", type=int, default=2000, help="The number of program generated")
    parse.add_argument('--i_start', type=int, default=None,
                       help='The start idx of api')
    parse.add_argument('--i_end', type=int, default=None,
                       help='The end idx of api')
    flags, _ = parse.parse_known_args(sys.argv[1:])
    generator = ProgramGenerator(flags)
    generator.run(constraints_dir=flags.constraints_dir.rstrip("/"),
                  program_dir=flags.program_dir.rstrip("/"),
                  counterpart_dir=flags.counterpart_dir.rstrip("/"),
                  api_source=flags.api_source,
                  total_time=flags.time_limit,
                  num_program=flags.num_program
                )

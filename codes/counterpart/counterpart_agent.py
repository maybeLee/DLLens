# This script will use ChatGPT to extract the counterpart APIs between one library (e.g., TensorFlow) and another library (e.g., PyTorch)
# It is designed with the following steps:
# 1. generate sample inputs (from either GPT or mine from existing test suite)
# 2. evaluate the sample inputs on the target API
# 3. request GPT to generate program using another library's API to implement the same function.
# 4. dynamically run the code, use runtime error as the feedback to fix the function.

import argparse
import json
import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import tensorflow as tf
import utils.static_analysis_utils as static_tools
from codes.catcher.type_catcher_from_doc import TypeCatcher
from codes.chatgpt import ChatGPTCall, NotEnoughBudgetError
from codes.constraints.constraint_solver import Tensor
from codes.counterpart.evaluate_counterpart import CounterpartEvaluator, load_and_compare, InputConvertor
from codes.prompt_text.counterpart_collection_prefix import construct_feedback_query, \
    counterpart_query, system_msg, \
    input_error_feedback_query
from codes.prompt_text.sample_input_prefix_1 import sample_input_query_1
from codes.prompting.Message import Message
from codes.testing.testing_process import MainTester
from codes.logger.logger import Logger
from codes.catcher.type_catcher_from_example import DynamicTypeCatcher
from utils import utils
from utils.utils import clean_code, get_args_list, get_api_name, get_pkg_name, string_startswith, deprecated
import time
import ast
from pathlib import Path

# MACRO
NUM_COUNTERPART = 3  # the number of counterpart for GPT to generate
NUM_SAMPLE_INPUTS = 3  # the number of sample inputs for GPT to generate
TEMPERATURE = 0.4  # the temperature for GPT to generate
MAX_TOKEN = 2000  # the maximum token for GPT to generate
logger = Logger()

# we use these variables to understand what is the most time-consuming part.
timer_prompt_input = 0
timer_prompt_counterpart = 0
timer_testing = 0


def ask_gpt_timer(message: list, temperature=TEMPERATURE, num_choices=1, max_token=MAX_TOKEN) -> (str, int):
    gpt_timer_s_time = time.time()
    answer = llm.ask_gpt_openai(message, temperature=temperature, num_choices=num_choices, max_token=max_token)
    if answer == NotEnoughBudgetError:
        logger.warning('The budget is not enough to complete the task. Please add more budget.')
        exit(-1)
    return answer, time.time() - gpt_timer_s_time


def ask_gpt_input_timer(message: list, num_choices=NUM_SAMPLE_INPUTS) -> [str]:
    answer, _time_spent = ask_gpt_timer(message, num_choices=num_choices)
    global timer_prompt_input
    timer_prompt_input += _time_spent
    processed_answer = []
    for ans in answer:
        ans = ans.replace("```python", "```")
        code = clean_code(ans, silent=True)
        code = code[0] if isinstance(code, list) else code
        processed_answer.append(code)
    return processed_answer


def ask_gpt_counterpart_timer(message: list, num_choices=NUM_COUNTERPART) -> [str]:
    answer, _time_spent = ask_gpt_timer(message, num_choices=num_choices)
    global timer_prompt_counterpart
    timer_prompt_counterpart += _time_spent
    processed_answer = []
    for ans in answer:
        ans = ans.replace("```python", "```")
        code = clean_code(ans, silent=True)
        code = code[0] if isinstance(code, list) else code
        processed_answer.append(code)
    return processed_answer



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


def format_reference_func(api_sig: str, lib_name: str):
    """
    format_reference_func('tf.math.sin(x)', 'tensorflow') -> 'def tensorflow_call(x):\n  return tf.math.sin(x)'
    :param api_sig:
    :param lib_name:
    :return:
    """
    api_name = get_api_name(api_sig)
    args_list: [str] = get_args_list(api_sig)  # all argument names except the `name` argument
    keywords: [str, str] = utils.get_default_value(api_sig)
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
    if api_name.startswith("tf.raw_ops"):
        args_list = [f"{arg}={arg}" for arg in args_list]
    # we handle some special case such as class API or special API that returns non-tensor/scalar object
    if api_name in ['tf.nn.isotonic_regression']:
        code = f"def {lib_name}_call({','.join(args_str_list)}):\n  return {api_name}({','.join(args_list)})[0]"
    else:
        code = f"def {lib_name}_call({','.join(args_str_list)}):\n  return {api_name}({','.join(args_list)})"
    return code

def mutate_sample_inputs(c: dict, lib_name: str, tester) -> list[str]:
    """
    mutate the sample inputs to generate variance
    """

    def _replace_inputs(origin_inputs: str, arg_name, mutant: str):
        input_list = origin_inputs.strip().split("\n")
        for input_str in input_list:
            if input_str.strip().startswith(arg_name):
                input_list.remove(input_str)
                break
        input_list.append(mutant)
        return "\n".join(input_list)
    from codes.mutation.input_mutator import mutator_dict
    type_dict, obj_dict = DynamicTypeCatcher.catch(c)
    new_inputs = []
    for arg_name in type_dict:
        arg_type = type_dict[arg_name]
        if arg_name not in obj_dict or arg_type not in mutator_dict:
            continue
        obj = obj_dict[arg_name]
        mutator = mutator_dict[arg_type]
        mutants: list[str] = mutator().mutate(obj, arg_name, lib_name)
        for mutant in mutants:
            new_input = _replace_inputs(c['sample_inputs'][0], arg_name, mutant)
            s, error_msg = test_sample_inputs(c, inputs=new_input, arg_type=arg_type,
                                          lib_name=lib_name, tester=tester)
            if s == 0:  # The mutant is valid
                new_inputs.append(new_input)
    return list(set(new_inputs))
        
def test_sample_inputs(c: dict, inputs: str, arg_type: dict, lib_name: str, tester) -> (int, str):
    """
    Test the sample inputs on the target library
    :param c:
    :param inputs:
    :param arg_type:
    :param lib_name:
    :param tester:
    :return: (status, error_message): status: 0 for not-crash, 1 for crash
    """
    api_sig = c['function_name']
    api_name = get_api_name(api_sig)
    file_path = f"runtime/{lib_name}_{api_name}_test.py"
    runtime_path = f"runtime/{lib_name}_{api_name}_test.npy"
    counterpart = CounterpartEvaluator()
    counterpart.instantiate_inputs(inputs, lib_format=lib_name)
    counterpart.counterpart = c
    if lib_name == "tensorflow":
        code = counterpart._generate_tf_code(runtime_path, arg_type)
    else:
        assert lib_name == "pytorch"
        code = counterpart._generate_torch_code(runtime_path, arg_type)
    with open(file_path, "w") as file:
        file.write(code)
    testing_s_time = time.time()
    s, err_msg = tester._test(file_path, timeout_seconds=10)
    global timer_testing
    timer_testing += time.time() - testing_s_time
    return s, err_msg


def remove_imports_and_function_calls(code: str, func_name: str) -> str:
    # Remove import statement, function definition and call
    try:
        tree = ast.parse(code)
    except:
        return ""
    for node in ast.walk(tree):
        if node not in tree.body:  # some safety check
            continue
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            tree.body.remove(node)
        elif isinstance(node, ast.FunctionDef) and node.name == func_name:
            tree.body.remove(node)
        elif isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign)) and getattr(node, "value", None) is not None:
            call_node = getattr(node.value, "func", None)
            if isinstance(call_node, ast.Name) and call_node.id in [func_name, 'print']:
                tree.body.remove(node)
    return ast.unparse(tree)


def parse_and_check_sample_inputs(code: str, func_name: str) -> str:
    sample_inputs: str = remove_imports_and_function_calls(code, func_name)
    sample_inputs: str = InputConvertor.remove_uint_in_input(sample_inputs)
    # check if the sample inputs contains random generator, if so, return "" to discard this instance.
    if 'torch.rand(' in sample_inputs or 'torch.randn(' in sample_inputs:
        print(f"Find random generator in the sample inputs, we discard it.")
        return ""
    elif 'tf.random.uniform(' in sample_inputs or 'tf.random.normal(' in sample_inputs:
        print(f"Find random generator in the sample inputs, we discard it.")
        return ""
    return sample_inputs


def get_sample_inputs_from_llm(c: dict, lib_name: str, tester: MainTester, num_seeds: int) -> [str]:
    """
    Generate sample inputs using LLM
    :param c:
    :param lib_name:
    :param tester:
    :param num_seeds:
    :return:
    """
    api_sig = c['function_name']
    api_name = get_api_name(api_sig)
    arg_type = {}  # it won't be used in this version
    seed_func = c['counterparts'][lib_name]
    lib_info_mapping = {'tensorflow': 'TensorFlow 2.10.0', 'pytorch': 'PyTorch 2.1.0'}
    avoid_random_generator_mapping = {'tensorflow': 'tf.random.uniform and tf.random.normal',
                                      'pytorch': 'torch.rand and torch.randn'}
    lib_info = lib_info_mapping[lib_name.lower()]
    avoid_random_generator = avoid_random_generator_mapping[lib_name.lower()]
    query = sample_input_query_1.format(lib_info, avoid_random_generator, ','.join(c['inputs']), seed_func)
    # print(query)  # [DEBUG]
    message = Message()
    message.update_system_message(system_msg)
    message.update_query(query)
    answer: list[str] = ask_gpt_input_timer(message.message)
    crash_input_list = []
    valid_input_list = []
    for sample_input in answer:
        sample_input = parse_and_check_sample_inputs(sample_input, func_name=f"{lib_name.lower()}_call")
        # print(f"Analyzing input: \n{sample_input}")  # [DEBUG]
        if sample_input.strip() == "":  # skip testing if no input is collected
            continue
        s, error_msg = test_sample_inputs(c, inputs=sample_input, arg_type=arg_type,
                                          lib_name=lib_name, tester=tester)
        if s == 0:
            # print(f"Successfully generate sample inputs for api {api_name}")
            valid_input_list.append(sample_input)
            if len(valid_input_list) >= num_seeds:
                return valid_input_list
        else:
            # print(f"Crash, the error message is: {error_msg}")  # [DEBUG]
            crash_input_list.append((sample_input, error_msg))
    origin_query = query
    for i in range(len(crash_input_list)):
        sample_input, error_msg = crash_input_list[i]
        message = Message()
        message.update_context_by_query_answer(query=origin_query, answer=f"```\n{sample_input}\n'''")
        for j in range(2):
            query = input_error_feedback_query.format(error_msg)
            message.update_query(query)
            # print(message.message)  # [DEBUG]
            answer = ask_gpt_input_timer(message.message, num_choices=1)[0]
            # print(answer)  # [DEBUG]
            sample_input = parse_and_check_sample_inputs(answer, func_name=f"{lib_name.lower()}_call")
            if sample_input.strip() == "":
                continue
            s, error_msg = test_sample_inputs(c, inputs=sample_input, arg_type=arg_type,
                                              lib_name=lib_name, tester=tester)
            if s == 0:
                logger.info(f"Successfully generate sample inputs for api {api_name}, after repairing {j + 1} times.")
                valid_input_list.append(sample_input)
                if len(valid_input_list) >= num_seeds:
                    return valid_input_list
                break
            else:
                # logger.info(f"Crash, the error message is: {error_msg}")  # [DEBUG]
                message.update_context_by_query_answer(query=query, answer=f"```\n{sample_input}\n'''")
    return list(set(valid_input_list))


def obj_to_input_str(arg_name: str, arg) -> str:
    if isinstance(arg, (tf.Tensor, np.ndarray)):
        if not hasattr(arg, 'shape') or not hasattr(arg, 'dtype'):
            return f"{arg_name} = {arg}"
        shape = arg.shape
        dtype = arg.dtype.name
        shape_info = {index: value for index, value in enumerate(shape)}
        ndims = len(shape)
        sym_tensor = Tensor(name=arg_name, shape_info=shape_info, dtype=dtype, ndims=ndims)
        sym_tensor.set_inf = sym_tensor.set_nan = False
        sym_tensor.instantiate_attributes()
        return sym_tensor.build_np_repr()
    elif isinstance(arg, (list, tuple)):
        elt_list: [str] = []
        for elt in arg:
            if isinstance(elt, (tf.Tensor, np.ndarray)):
                shape = elt.shape
                dtype = elt.dtype.name
                shape_info = {index: value for index, value in enumerate(shape)}
                ndims = len(shape)
                sym_tensor = Tensor(name='xxx', shape_info=shape_info, dtype=dtype, ndims=ndims)
                sym_tensor.set_inf = sym_tensor.set_nan = False
                sym_tensor.instantiate_attributes()
                elt_expr = sym_tensor.build_np_repr().split("= ")[-1]
                elt_list.append(elt_expr)
            elif isinstance(elt, str):
                elt_list.append(f"'{elt}'")
        return f"{arg_name} = [{','.join(elt_list)}]"
    elif isinstance(arg, str):
        return f"{arg_name} = '{arg}'"
    return f"{arg_name} = {arg}"


class InputRes(object):
    def __init__(self, d_input, status, error_msg, src_output=None, dst_output=None):
        self.d_input= d_input.strip()
        self.status = status
        self.error_msg = error_msg.strip()
        self.src_output = src_output
        self.dst_output = dst_output
    
    def feedback_query(self):
        assert self.status != 0
        if self.status == 1:
            return f"[Crash On Input]:\n{self.d_input}\n[Error Message]:\n{self.error_msg}\n"
        else:
            return f"[Inconsistent On Input]:\n{self.d_input}\n[Expected Output]:\n{self.src_output}\n[Actual Output]:\n{self.dst_output}\n" 

def evaluate_counterpart(ans, cache):
    counterpart, source_lib, dest_lib, api_name, sample_inputs, dst_inputs, tester = cache
    arg_type = {}  # it won't be used in this version
    src_runtime_path = f"runtime/{source_lib}_{api_name}_test.npy"
    dst_runtime_path = f"runtime/{dest_lib}_{api_name}_test.npy"
    if ans.strip() == "":  # reject it if the extracted code is empty
        return "", [InputRes("", -1, "")]
    code = static_tools.find_func_code(f"{dest_lib}_call", ans)
    if code.strip() == "":
        return "", [InputRes("", -1, "")]
    counterpart['counterparts'][dest_lib] = code
    res_list: [InputRes] = []
    res_count = {0: 0, 1: 0, 2: 0}  # 0 for success, 1 for crash, 2 for fail, -1 for rejected
    for s_input, d_input in zip(sample_inputs, dst_inputs):
        src_status, src_err = test_sample_inputs(counterpart, inputs=s_input, arg_type=arg_type,
                                                    lib_name=source_lib,
                                                    tester=tester)
        if src_status != 0:
            # the original sample input is invalid or the API under test is non-deterministic
            return "", [InputRes(d_input, -1, src_err)]
        dst_status, dst_err = test_sample_inputs(counterpart, inputs=d_input, arg_type=arg_type, lib_name=dest_lib,
                                                    tester=tester)
        src_output = None
        dst_output = None
        if dst_status == 0:
            value_status = 0 if \
                load_and_compare(src_runtime_path, dst_runtime_path, api_name, silent=True) == 1 else 2
            try:
                src_output = np.load(src_runtime_path, allow_pickle=True)
            except:
                src_output = None
            try:
                dst_output = np.load(dst_runtime_path, allow_pickle=True)
            except:
                dst_output = None
        else:
            value_status = 1
        res_count[value_status] += 1
        res_list.append(InputRes(d_input, value_status, dst_err, src_output, dst_output))
    logger.info(f"Eval Results: {res_count[0]} passes; {res_count[1]} crashes; {res_count[2]} fails")
    return code, res_list

def counterpart_feedback_incon(message, dst_code, res_list: [InputRes], cache, crash_only=False, try_time=0) -> (bool, str):
    """
    Recursively apply the feedback to make the generate code applicable for all test inputs
    """
    bad_res: [InputRes] = [res for res in res_list if res.status != 0]
    inconsistent_res: [InputRes] = [res for res in res_list if res.status == 2]
    crash_res: [InputRes] = [res for res in res_list if res.status == 1]
    if crash_only is False and len(bad_res) == 0:
        logger.info(f"Successfully repair the counterpart after {try_time} iterations.")
        return True, dst_code
    if crash_only is True:
        if len(crash_res) == 0 and len(bad_res) == 0:
            logger.info(f"Successfully repair the counterpart after {try_time} iterations.")
            return True, dst_code
        elif len(crash_res) == 0 and len(bad_res) > 0:
            logger.info(f"Successfully repair the crash counterpart after {try_time} iterations, but there are still inconsistencies.")
            return False, dst_code
    if try_time >= 5:
        logger.info(f"Fail to repair the counterpart after {try_time} iterations.")
        return False, dst_code
    if crash_only is False:
        feedback_query = construct_feedback_query(crash_res, inconsistent_res)
    else:
        feedback_query = construct_feedback_query(crash_res, [])
    message.update_query(feedback_query)
    # message.show_message()  # [DEBUG]
    answer: str = ask_gpt_counterpart_timer(message.message, num_choices=1)[0]
    # logger.info(f"The answer is: \n{answer}")  # [DEBUG]
    # see if we need to keep this answer, if we want, replace previous answer with the new one.
    new_dst_code, new_res_list = evaluate_counterpart(answer, cache)
    if new_dst_code == "":
        return counterpart_feedback_incon(message, dst_code, res_list, cache, try_time=try_time+1)
    new_bad_res = [res for res in new_res_list if res.status != 0]
    new_crash_res = [res for res in new_res_list if res.status == 1]
    if len(new_bad_res) < len(bad_res) or len(new_crash_res) < len(crash_res):
        message.context.pop(-1)
        message.context.append({"role": "assistant", "content": f"```\n{new_dst_code}\n```"})
        dst_code, res_list = new_dst_code, new_res_list
    return counterpart_feedback_incon(message, dst_code, res_list, cache, try_time=try_time+1)

def check_external_package(code: str, lib_name: str) -> bool:
    """
    Check if the code contains external package
    """
    if lib_name == "tensorflow":
        string = 'torch.'
    elif lib_name == "pytorch":
        string = 'tf.'
    if string in code:
        return True
    return False

def clean_inputs(counterpart: dict, source_lib: str) -> dict:
    sample_inputs = counterpart['sample_inputs']
    filtered_inputs = []
    for s_input in sample_inputs:
        if not check_external_package(s_input, lib_name=source_lib):
            if s_input not in filtered_inputs:
                filtered_inputs.append(s_input)
    counterpart['sample_inputs'] = filtered_inputs
    return counterpart

def find_counterpart_from_llm(counterpart: dict, api_name: str, source_lib: str, dest_lib: str,
                              tester: MainTester) -> str:
    sample_inputs = counterpart['sample_inputs']
    dst_inputs = [
        InputConvertor.convert_input(input_str=s_input, src_lib=source_lib, dst_lib=dest_lib)
        for s_input in sample_inputs
    ]
    cache = (counterpart, source_lib, dest_lib, api_name, sample_inputs, dst_inputs, tester)
    dst_input_str = ""
    for i, d_input in enumerate(dst_inputs[:3]):  # we only include three sample inputs in the prompt
        dst_input_str += f"[Input {i + 1}]: \n{d_input}\n"
    query = counterpart_query.replace("[DST_LIB]", dest_lib)
    query = query.replace("[SRC_LIB]", source_lib)
    query = query.replace("[SAMPLE_INPUTS]", dst_input_str)
    query = query.replace("[SRC_LIB_FUNC]", counterpart['counterparts'][source_lib])
    message = Message()
    message.update_system_message(system_msg)
    message.update_query(query)
    # message.show_message()  # [DEBUG]
    answer_list: list[str] = ask_gpt_counterpart_timer(message.message, num_choices=NUM_COUNTERPART)
    for i, answer in enumerate(answer_list):
        # logger.info(f"The answer is: {answer}")
        # eval_res: 0 for success, 1 for crash, 2 for fail
        dst_code, res_list = evaluate_counterpart(answer, cache)
        if dst_code == "":
            logger.info(f"No code is found on the {i}th trial.")
            continue
        message = Message()
        message.update_system_message(system_msg)
        message.update_context_by_query_answer(query=query, answer=f"```\n{dst_code}\n```")
        # message.show_message()  # [DEBUG]
        result, repaired_dst_code = counterpart_feedback_incon(message, dst_code, res_list, cache)
        if result is True:
            return repaired_dst_code
    return ""


def report_efficiency():
    print(f"============== Let's take a coffee =============")
    print(f"Current time spent in input generation: {timer_prompt_input}")
    print(f"Current time spent in counterpart generation: {timer_prompt_counterpart}")
    print(f"Current time spent in testing: {timer_testing}")
    print(f"==============     Back to work    =============")


def main(api_list=None, source_lib: str = "tensorflow", dest_lib: str = "pytorch",
                        save_dir: str = "./data/working_dir/counterpart", num_seeds=3,
                        i_start: int = -1, i_end: int = -1, model_name = "gpt-3.5-turbo") -> None:
    save_dir = Path(save_dir)  # type: ignore
    work_dir = save_dir / f"{source_lib}_{dest_lib}_{model_name}_{num_seeds}_seeds"  # type: ignore
    os.makedirs(work_dir, exist_ok=True)
    counterpart_dir = work_dir / "counterparts"
    os.makedirs(counterpart_dir, exist_ok=True)
    api_sig_list = load_apis(package_name="", library_name=source_lib) if api_list is None else api_list
    # difficult_api = _load_difficult()
    logger.info(f"In total, we have {len(api_sig_list)} APIs")
    tester = MainTester()
    if i_start != -1 and i_end != -1:
        api_sig_list = api_sig_list[i_start:i_end]
    for i, api_sig in enumerate(api_sig_list):
        api_sig = api_sig.replace("\*", "")  # in case there are signature: (xxx, \*, xxx)
        api_name = get_api_name(api_sig)
        counterpart_path = counterpart_dir / f"{api_name}.json"
        logger.info(f"Working on api: {api_name}")
        if f"{api_name}.json" in os.listdir(counterpart_dir):
            logger.info(f"Correct counterpart has already been found for the API: {api_sig}. We skip it.")
            continue
        if (i + 1) % 10 == 0:
            report_efficiency()
            print('=========== We restart another testing process ===========')
            tester.stop_child()
            tester = MainTester()
        src_func_str = format_reference_func(api_sig, lib_name=source_lib)
        initial_counterpart: dict = {
            "function_name": api_sig,
            "inputs": get_args_list(api_sig),
            "sample_inputs": "todo",
            "counterparts": {
                source_lib: src_func_str,
                dest_lib: "todo",
            }
        }
        try_time = 0
        dest_func_str = ""
        while try_time < 5 and dest_func_str == "":
            counterpart = initial_counterpart.copy()
            try_time += 1
            sample_inputs = get_sample_inputs_from_llm(counterpart, lib_name=source_lib, tester=tester, num_seeds=num_seeds)
            assert isinstance(sample_inputs, list)
            if len(sample_inputs) == 0:
                logger.info(f"Fail to generate sample inputs for API: {api_sig}. We skip this API in this round.")
                continue
            else:
                counterpart['sample_inputs'] = sample_inputs.copy()
                counterpart['llm_inputs'] = sample_inputs.copy()  # for record only
                logger.info(f"Start mutating the sample inputs")
                mutant_inputs = mutate_sample_inputs(counterpart, source_lib, tester)
                counterpart['sample_inputs'] += mutant_inputs
            logger.info(f"LLM Generated {len(sample_inputs)} Sample Inputs; {len(mutant_inputs)} Mutant Inputs.")
            counterpart = clean_inputs(counterpart, source_lib=source_lib)
            logger.info(f"After cleaning, we have {len(counterpart['sample_inputs'])} sample inputs.")
            dest_func_str = find_counterpart_from_llm(counterpart, api_name, source_lib=source_lib, dest_lib=dest_lib,
                                                    tester=tester)
            if dest_func_str == "":
                logger.info(f"Fail to generate the counterpart for api: {api_name}. Try Time: {try_time}")
            else:
                logger.info(f"Successfully find counterpart for api: {api_name}. Try Time: {try_time}")
                counterpart['counterparts'][dest_lib] = dest_func_str
                with open(counterpart_path, "w") as file:
                    json.dump(counterpart, file, indent=2)
        cost = llm.current_cost()
        logger.info(f"The total cost for GPT querying is: {cost} USD")
    tester.stop_child()


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--source_lib', type=str, default="tensorflow")
    parse.add_argument('--dest_lib', type=str, default="pytorch")
    parse.add_argument('--api_key_path', type=str, default="./data/api_openai.key")
    parse.add_argument('--save_dir', type=str, default="./data/working_dir/counterpart")
    parse.add_argument('--model_name', type=str, default="gpt-3.5-turbo")
    parse.add_argument("--i_start", type=int, default=-1, help="The start index of the api name, -1 for all")
    parse.add_argument("--i_end", type=int, default=-1, help="The end index of the api name, -1 for all")
    parse.add_argument("--num_seed", type=int, default=1,
                       help="The number of sample seed used for counterpart collection")
    flags, _ = parse.parse_known_args(sys.argv[1:])
    print(flags)
    s_time = time.time()
    llm = ChatGPTCall(api_key_file=flags.api_key_path, model_name=flags.model_name)
    main(
        source_lib=flags.source_lib,
        dest_lib=flags.dest_lib,
        save_dir=flags.save_dir.rstrip("/"),
        num_seeds=flags.num_seed,
        i_start=flags.i_start,
        i_end=flags.i_end,
        model_name=flags.model_name
    )
    e_time = time.time()
    print(f"Total time cost is: {e_time - s_time}")

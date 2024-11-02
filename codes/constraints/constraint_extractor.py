import argparse
import ast
import json
import os
import re
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Union, Optional

from codes.chatgpt import ChatGPTCall
from codes.coverage.ControlFlow import CFGNode
from codes.logger.logger import logger
from codes.prompting.Message import Message
from utils import utils
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


class ConstraintAnalyzer(object):
    """
    Given the function entry of an api, get the path condition (i.e., constraints) on api arguments
    """

    def __init__(self, api_entry, args_type: dict[str, str], cons_trace_dir: Optional[str] = None, symbol_path: Optional[str] = None,
                 extraction_mode: str = "rules", lib_name: str = "tensorflow", llm=None):
        # Step 1: get the control flow graph of the function node and its children node.
        self.api_entry_func = api_entry
        self.args_type = args_type
        self.input_list = list(args_type.keys())
        # to speed up constraint validation analysis, we do valid constraint caching
        self.valid_cons_set: set[str] = set()
        self.invalid_cons_set: set[str] = set()  # we log all invalid constraints
        self.cons_trace_dir = cons_trace_dir
        self.symbol_path = symbol_path
        self.extraction_mode = extraction_mode
        self.s_time = 0
        self.e_time = 0
        self.cons_trace_list = []
        self.lib_name = lib_name
        self.llm = llm

    def extract_constraints_from_node(self, p: str, assignment_dict: dict[str, Symbol], func_name: str) -> Optional[str]:
        """This function should always output valid constraints that can be added by ConstraintSolver.add_constraints(cons, validate=True) to the solver.
        If the constraint is invalid, return None.

        Args:
            p (str): _description_
            assignment_dict (dict[str, Symbol]): _description_
            func_name (str): _description_

        Returns:
            Optional[str]: _description_
        """
        # we first apply some rules to transfer some symbols into our vocabulary
        p = ConstraintParser.rule_based_constraint_parser(p)
        p = ConstraintParser.remove_unnecessary_spaces(p)
        if ConstraintParser.contains_invalid_cons_keyword(p, library_name=self.lib_name):
            # we skip analyzing those unrelated nodes which contains specific keywords
            return None
        try:
            # we dynamically check if the expression can be parsed by python's ast, otherwise, it should be parsed by tree-sitter-c.
            node: CFGNode = static_tools.CFGNode(ast=ast.parse(p).body[0])
        except:
            node: CFGNode = static_tools.CppCFGNode(
                ast=static_tools.parser.parse(bytes(p, "utf8")).root_node.children[0].children[0])
        # print(f"The node is: {p}\n", InputConstraintExtractor.is_if_assert(node), InputConstraintExtractor.is_assert(node))  # [DEBUG]
        if InputConstraintExtractor.is_if_assert(node):
            # if the node is an assertion that we focus
            assert node.language_type == "py"
            assert isinstance(node.ast_node.test, ast.Compare)
            predicate_res: str = ast.unparse(node.ast_node.test.comparators[0])
            predicate: str = ast.unparse(node.ast_node.test.left)
            symbol = Symbol("condition", predicate, assignment_dict,
                            self.args_type, self.llm, self.symbol_path, func_name)
            predicate: str = symbol.parse_condition(mode=self.extraction_mode)
            predicate = extract_constraints(predicate, self.input_list)
        elif InputConstraintExtractor.is_assert(node):
            predicate_res = "True"
            predicate = ast.unparse(node.ast_node.test)
            symbol = Symbol("condition", predicate, assignment_dict,
                            self.args_type, self.llm, self.symbol_path, func_name)
            predicate: str = symbol.parse_condition(mode=self.extraction_mode)
        elif InputConstraintExtractor.is_validation_node(node.source()):
            predicate_res = "True"
            # we need to extract predicate from the validation node.
            predicate: str = InputConstraintExtractor.extract_predicate_from_validation(
                node.source())
            symbol = Symbol("condition", predicate, assignment_dict,
                            self.args_type, self.llm, self.symbol_path, func_name)
            predicate: str = symbol.parse_condition(mode=self.extraction_mode)
        else:
            return None
        try:
            predicate = ConstraintParser.rule_based_constraint_parser(
                predicate)
        except:
            pass
        if predicate.lower().strip() == "invalid constraint":
            # print(f"Invalid constraint: {p}")  # [DEBUG]
            return None
        # Step 2: predicate -> constraint
        constraint = f"assert ({predicate})"
        # print(f"The solved constraint is: {constraint}")  # [DEBUG]
        if len(constraint) >= 1000 or len(ConstraintParser.constraint_targets_input(constraint, self.input_list)) == 0:
            # if the constraint is too large (>=1000), which is crazy, our system may not be reliable to process it
            return None
        if (constraint in self.valid_cons_set) or (
                ConstraintParser.z3_valid_constraint(constraint, self.input_list, args_type_dict=self.args_type)):
            # print(f"We find valid constraint: {constraint} is {predicate_res}")  # [DEBUG]
            solved_constraint = f"{constraint} is {predicate_res}"
            self.valid_cons_set.add(constraint)
        else:
            # print(f"We find invalid constraint: {constraint} is {predicate_res}")  # [DEBUG]
            solved_constraint = None
            self.invalid_cons_set.add(constraint)
        # we also save the trace for future evaluation
        if self.cons_trace_dir is not None:
            trace_list = symbol.get_trace()
            trace_list[-1] = f"condition: {trace_list[-1].split(' = ')[-1].strip()}"
            # trace[-1] = f"condition: {symbol.value}"
            if trace_list not in self.cons_trace_list:
                self.cons_trace_list.append(trace_list)
                trace_file_name = os.path.join(
                    self.cons_trace_dir, f"trace_{len(os.listdir(self.cons_trace_dir))}.txt")
                with open(trace_file_name, "w") as file:
                    file.write("\n".join(trace_list))
                    file.write(f"\nSolved constraint:\n{constraint}\n")
                    if solved_constraint is not None:
                        file.write(f"Parsable Status: True\n")
                    else:
                        file.write(f"Parsable Status: False\n")
        return solved_constraint

    @staticmethod
    def find_callee(node: CFGNode, callee_func_dict):
        cn_dict = static_tools.get_call_from_node(
            node.ast_node, node.language_type)
        callee_func = {}
        for cn in cn_dict.keys():
            if static_tools.fuzzy_func_matching(cn, list(callee_func_dict)) != "":
                callee_func[cn] = callee_func_dict[static_tools.fuzzy_func_matching(
                    cn, list(callee_func_dict.keys()))]
        return callee_func

    @staticmethod
    def _process_if_node(node_str: str) -> str:
        node_str = node_str.replace("_if: ", "assert ")
        node_str = node_str.replace("!=", "<temp>neq<temp>")
        node_str = node_str.replace('!', 'not ')
        node_str = node_str.replace("<temp>neq<temp>", "!=")
        node_str = node_str.replace("->",
                                    ".")  # we need to convert the cpp expression to python expression inside the assertion
        node_str = node_str.replace("assert ", "assert (")
        node_str += ")"
        return node_str

    def update_assignment_dict(self, node, assignments: dict[str, Symbol], func_name: str):
        # This function will only update assignments if the node is the assignment node
        variable, value = static_tools.get_assign_expr(
            node, node.language_type)
        if value == "" or variable == "":
            return assignments
        symbol_instance = Symbol(expr=variable, value=value, asg_dict=assignments.copy(), args_type=self.args_type,
                                 llm=self.llm, symbol_path=self.symbol_path, func_name=func_name)
        assignments[variable] = symbol_instance
        return assignments

    @staticmethod
    def cons_is_valid(cons: list[str]) -> bool:
        """
        Use a simple rule to check if the cons is valid.
        Rule:
        invalid cons: there are two strings have the same prefix while one ends with 'is True' and another one ends with 'is False'.
        Example:
        ["XXX is True", "XXX is False"] -> False
        ["XXX is True", "YYY is False"] -> True
        :param cons:
        :return:
        """
        seen = {}  # Dictionary to store seen prefixes

        for s in cons:
            if s.endswith(" is True"):
                prefix = s[:-8]  # Remove " is True" to get the prefix
                if prefix in seen and seen[prefix] == "False":
                    return False  # Found a matching pair
                seen[prefix] = "True"

            elif s.endswith(" is False"):
                prefix = s[:-9]  # Remove " is False" to get the prefix
                if prefix in seen and seen[prefix] == "True":
                    return False  # Found a matching pair
                seen[prefix] = "False"

        return True  # No matching pairs found

    def _merge_cons(self, cons1_list: list[[str]], cons2_list: list[[str]]) -> list[[str]]:
        # Return the cartesian produce of two constraint lists
        # _merge_cons([[]], [['a', 'b']]) = [['a', 'b']]
        # _merge_cons([['c']], [['a', 'b'], ['e', 'f']]) = [['c', 'a', 'b'], ['c', 'e', 'f']]
        cons1_list = cons1_list.copy()
        cons2_list = cons2_list.copy()
        if len(cons1_list) == 0 and len(cons2_list) == 0:
            return [frozenset([])]
        if len(cons1_list) == 0:
            return cons2_list
        if len(cons2_list) == 0:
            return cons1_list
        merged_cons = list(map(lambda x: frozenset(
            list(x[0]) + list(x[1])), product(cons1_list, cons2_list)))
        # remove invalid cons by
        final_cons = []
        for cons in merged_cons:
            if ConstraintAnalyzer.cons_is_valid(cons):
                final_cons.append(cons)
        # remove duplicated cons
        final_cons = ConstraintAnalyzer._rm_duplicate_cons(final_cons)
        final_cons = ProgramGenerator._get_solvable_constraints(
            final_cons, self.args_type)
        return final_cons

    @staticmethod
    def _rm_duplicate_cons(cons_list: list[[str]]) -> list[[str]]:
        return list(set(cons_list))

    def walk_cfg_node(self, node: CFGNode, asg_dict: dict[str, Symbol], func_node, cons_prefix = None) -> list[[str]]:
        # print(f"Working on node: {node.source()}, the function is: {func_node.func_name}")  # [DEBUG]
        if (self.s_time != 0 and (time.time() - self.s_time) > 600) or node is None:
            if cons_prefix is not None:
                return [frozenset([cons_prefix])]
            else:
                return []
        assignment_dict = asg_dict.copy()
        assignment_dict = self.update_assignment_dict(
            node, assignment_dict, func_name=func_node.func_name)
        current_cons_list: list[[str]] = []
        # to speed up, each element in cons_list is a frozenset element
        final_cons_list: list[[str]] = []
        callee_node = self.find_callee(node, func_node.children)
        cn_dict = static_tools.get_call_from_node(node.ast_node, node.language_type)
        if len(callee_node) != 0:
            for cn, f in callee_node.items():
                # do input-arg mapping
                arg_list = static_tools.get_args(f.get_source(), f.language_type)
                input_list, input_keywords = cn_dict[cn]
                if self.lib_name.lower() == "pytorch":
                    # for corner case such as: window.narrow(xx), we need to add the object name ('window') to the input list
                    # see test: test_constraint_analyzer_cc_1/2/3
                    envir_name = static_tools.fuzzy_func_matching(
                        cn, list(func_node.children))
                    if len(arg_list) > len(input_list) and \
                            len(arg_list) >= 1 and len(arg_list[0]) == 1 and arg_list[0][0] == "self" \
                            and cn[:-len(envir_name)].rstrip(".:") != "":
                        input_list = [
                            cn[:-len(envir_name)].rstrip(".:")] + input_list
                    if cn.endswith("_out") and len(arg_list) > 1 and len(input_list) > 1:
                        # only applicable for *_out, handle: angle_out(result, self) -> angle_out(self, result)
                        # we will remove the result from both the arg_list and input_list
                        new_arg_list = []
                        for i in range(len(arg_list)):
                            if arg_list[i][0] == "result" and "result" in input_list:
                                input_list.remove("result")
                            else:
                                new_arg_list.append(arg_list[i])
                        arg_list = new_arg_list
                child_assignment_dict = asg_dict.copy()
                for arg in arg_list:
                    arg_name = arg[0]
                    if arg_name in input_keywords:
                        child_assignment_dict[arg_name] = Symbol(
                            arg_name, input_keywords[arg_name],
                            child_assignment_dict, args_type=self.args_type, llm=self.llm,
                            symbol_path=self.symbol_path, func_name=f.func_name
                        )
                    elif len(input_list) > 0:
                        child_assignment_dict[arg_name] = Symbol(
                            arg_name, input_list.pop(0), child_assignment_dict, llm=self.llm,
                            args_type=self.args_type, symbol_path=self.symbol_path, func_name=f.func_name
                        )
                    else:
                        if len(arg) == 2:  # the argument has default value
                            child_assignment_dict[arg_name] = Symbol(
                                arg_name, arg[-1], child_assignment_dict,
                                args_type=self.args_type, llm=self.llm,
                                symbol_path=self.symbol_path, func_name=f.func_name
                            )
                        else:
                            # the argument has no default value and on concrete value has been passed into the function.
                            # it is possible there may be some false negatives, but we simply ignore this case.
                            pass
                # print(f"Entering function: {f.func_name}, the arg list is: {arg_list}, the input list is: {input_list}, the assignment is: {child_assignment_dict}")  # [DEBUG]
                callee_entry = static_tools.get_func_cfg(
                    f.func_impl, f.language_type)
                child_cons_list = self.walk_cfg_node(
                    callee_entry, child_assignment_dict, f)
                current_cons_list = self._merge_cons(
                    current_cons_list, child_cons_list)
        if len(node.children) == 2:
            # arc node: first child refers to the true branch, second refers to the else branch
            kind = {0: "True", 1: "False"}
            for i, child in enumerate(node.children):
                node_str = self._process_if_node(node.source())
                node_str += f" is {kind[i]}"
                node_cons = self.extract_constraints_from_node(
                    node_str, assignment_dict, func_name=func_node.func_name)
                child_cons_list = self.walk_cfg_node(
                    child, assignment_dict, func_node, node_cons)
                final_cons_list += self._merge_cons(
                    current_cons_list, child_cons_list)
        elif len(node.children) == 1:
            node_cons = self.extract_constraints_from_node(
                node.source(), assignment_dict, func_name=func_node.func_name)
            child_cons_list = self.walk_cfg_node(
                node.children[0], assignment_dict, func_node, node_cons)
            final_cons_list = self._merge_cons(
                current_cons_list, child_cons_list)
        elif len(node.children) == 0:
            final_cons_list = current_cons_list
        else:
            logger.warning(
                f"Node: {node.source()} has invalid children number: {len(node.children)}")
            final_cons_list = current_cons_list
        if cons_prefix is not None:
            final_cons_list = self._merge_cons(
                [frozenset([cons_prefix])], final_cons_list)
        else:
            final_cons_list = ConstraintAnalyzer._rm_duplicate_cons(
                final_cons_list)
        if len(final_cons_list) >= 500:
            print(f"Too many solvable constraints, we randomly choose 500.")
            final_cons_list = np.random.choice(final_cons_list, SOLVABLE_CONS_LIMIT, replace=False)
            final_cons_list = list(final_cons_list)
        return final_cons_list

    def traverse(self):
        entry_node = static_tools.get_func_cfg(self.api_entry_func.func_impl,
                                               language_type=self.api_entry_func.language_type)
        if entry_node is None:
            return
        self.s_time = time.time()
        cons_list = self.walk_cfg_node(entry_node, {}, self.api_entry_func)
        # a cons_list contains a list of constraint set. Each constraint set should be solved.
        if cons_list == [[]] or cons_list == [frozenset()]:
            return []
        print(f"Total seconds: {time.time() - self.s_time}")
        return cons_list


def extract_constraints(condition: Union[str, ast.expr], param_list: [str]) -> str:
    """
    :param condition:
    :param param_list:
    :return:
    """
    if isinstance(condition, str):
        try:
            condition = ast.parse(condition).body[0]
        except:
            # if the string is not parsable, it is in CPP format, we skip it.
            return condition
    if isinstance(condition, ast.Expr):
        return extract_constraints(condition.value, param_list)
    if isinstance(condition, ast.UnaryOp):
        op = condition.op
        op_str = OP_TO_STRING[op.__class__.__name__.lower()]
        operand = condition.operand
        operand_str = ConstraintParser.rule_based_constraint_parser(
            ast.unparse(operand))
        return f"{op_str} {operand_str}"
    elif isinstance(condition, ast.Compare):
        op = condition.ops[0]
        op_str = OP_TO_STRING[op.__class__.__name__.lower()]
        comparator = condition.comparators[0]
        operand = condition.left
        comp_str = ConstraintParser.rule_based_constraint_parser(
            ast.unparse(comparator))
        operand_str = ConstraintParser.rule_based_constraint_parser(
            ast.unparse(operand))
        return f"{operand_str} {op_str} {comp_str}"
    elif isinstance(condition, ast.BoolOp):
        op = condition.op
        op_str = OP_TO_STRING[op.__class__.__name__.lower()]
        constraint_expr = f"({extract_constraints(condition.values[0], param_list)})"
        for v in condition.values[1:]:
            constraint_expr += f" {op_str} ({extract_constraints(v, param_list)})"
        return constraint_expr
    else:
        return ast.unparse(condition)


class InputConstraintExtractor(object):
    """
    This class will automatically get the implementation of the target API, including its downstream functions,
    then collect the API input constraints.
    Constraint format can be some natural languages or a well-defined language that can be understood by machine.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.valid_count = 0
        self.invalid_count = 0

    @staticmethod
    def is_if_assert(node: CFGNode) -> bool:
        # Require the assertion to be: assert (xxx) is True/False
        ast_node = node.ast_node
        return isinstance(ast_node, ast.Assert) and isinstance(ast_node.test, ast.Compare) \
            and hasattr(ast_node.test, "comparators") and isinstance(ast_node.test.ops[0], ast.Is)

    @staticmethod
    def is_assert(node: CFGNode) -> bool:
        return isinstance(node.ast_node, ast.Assert)

    @staticmethod
    def is_validation_node(node_str: str) -> bool:
        # we add some validation check function into our source
        check_func_list: list[str] = [
            "args_to_matching_eager",
            "_linalg_check_errors",
            "squareCheckInputs",
            "checkFloatingOrComplex",
            "checkIsMatrix",
            "OP_REQUIRES",
            "TORCH_CHECK",
            "TORCH_CHECK_LINALG",
            "TORCH_CHECK_INDEX",
            "TORCH_CHECK_VALUE",
            "TORCH_INTERNAL_ASSERT"
        ]
        if "ctx->forward_input_or_allocate_output" in node_str or "ctx->allocate_output" in node_str:
            return False
        is_valid = False
        for check_func in check_func_list:
            if check_func in node_str:
                is_valid = True
        return is_valid

    @staticmethod
    def extract_predicate_from_validation(node_str: str) -> str:
        """
        node_str = "OP_REQUIRES(context, 0 == num_dims % 2,
                errors::InvalidArgument("The rank of the tensor should be \
                                         even and positive, got shape ",
                                        tensor.shape().DebugString()));"
        InputConstraintExtractor.extract_predicate_from_validation(node_str) = "0 == num_dims % 2"
        tests in tests.TestConstraintAnalyzer.test_val_predicate_extract*
        :param node_str: node contains validation functions such as OP_REQUIRES
        :return:
        """
        if "OP_REQUIRES" not in node_str and "TORCH_CHECK" not in node_str:
            # Currently we only implement the predicate extraction for OP_REQUIRES
            return node_str
        if not node_str.endswith(";"):
            node_str += ";"
        try:
            node = static_tools.parser.parse(
                bytes(node_str, 'utf8')).root_node.children[0]
            func_name = node.children[0].child_by_field_name(
                'function').text.decode()
        except:
            # if the node's error message is wrapped by ' instead of "", it cannot pass the node
            try:
                node = static_tools.parser.parse(
                    bytes(node_str.replace("\'", "\:"), 'utf8')).root_node.children[0]
                func_name = node.children[0].child_by_field_name(
                    'function').text.decode()
            except:
                return node_str
        if func_name in ["OP_REQUIRES", "OP_REQUIRES_OK"]:
            predicate = node.children[0].child_by_field_name(
                'arguments').children[3].text.decode()
        elif func_name in ["TORCH_CHECK", "TORCH_CHECK_INDEX", "TORCH_CHECK_LINALG", "TORCH_CHECK_VALUE",
                           "TORCH_INTERNAL_ASSERT"]:
            predicate = node.children[0].child_by_field_name(
                'arguments').children[1].text.decode()
            # print(f"predicate extracted from func: {func_name} is: {predicate}")  # [DEBUG]
        else:
            raise NotImplementedError(
                f"Not implemented for validation function: {func_name}")
        return predicate

    @staticmethod
    def tf_api_to_signature(api_name: str) -> str:
        with open('./data/tf_api_list.txt') as file:
            content = file.read().rstrip().splitlines()
        for sig in content:
            if sig.startswith(f"{api_name}("):
                return sig

    @staticmethod
    def torch_api_to_signature(api_name: str) -> str:
        with open('./data/torch_api_list.txt') as file:
            content = file.read().rstrip().splitlines()
        for sig in content:
            if sig.startswith(f"{api_name}("):
                return sig

    def get_api_implementations(self, api_name: str, library_name: str):
        """
        Extract the constraints given the api name.
        :param api_name: any string that can be formatted to from xx to yy. For instance, it can be: tf.math.sin, tensorflow.python.eager.backprop._must_record_gradient, etc.
        :param library_name: the full name of library. Currently, we support 'tensorflow' and 'pytorch'
        :return: the dictionary representing the input constraints
        """
        if library_name == "tensorflow":
            skip_tokens = ['/framework/', '/eager/', '/core/config',
                           '.so', 'python/util', "control_flow_ops"]
        elif library_name == "pytorch":
            skip_tokens = None
        else:
            raise NotImplementedError(f"Library not found: {library_name}")
        source_path = f"./data/api_implementations/{library_name}/{api_name}.json"
        entry_node = static_tools.get_api_implementation(api_name, library_name, source_path=source_path,
                                                         skip_tokens=skip_tokens, depth=5)
        return entry_node

    def extract_constraint(self, impl: str, args_type: dict[str, str], save_dir, symbol_path,
                           library_name: str, extraction_mode: str) -> list[str]:
        """
        Extract the constraint on the function input inside the counterpart
        :param impl: the source code of the function
        :param args_type: list of arguments along with their types
        :param save_dir:
        :param symbol_path:
        :param library_name:
        :param extraction_mode:
        :return: list of constraint, each constraint is for one path
        """
        save_dir = save_dir / library_name
        cons_trace_dir = save_dir / "cons_traces"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(cons_trace_dir, exist_ok=True)
        impl_node = FuncNode(file_path="temp.py", func_impl=impl)
        api_list = static_tools.find_apis(impl, library_name=library_name)
        for api_name in api_list:
            entry_node = self.get_api_implementations(api_name, library_name)
            if entry_node is None:
                continue
            if entry_node.func_impl.strip() == "":
                print(
                    f"[WARNING] We fail to find the function implementation for {api_name} in {library_name}!")
                continue
            impl = impl.replace(api_name, entry_node.func_name)
            impl_node.children[entry_node.func_name] = entry_node
        impl_node.func_impl = impl
        # from tests.TestCallGraph import show_func_node  # [DEBUG]
        # show_func_node(impl_node)  # [DEBUG]
        # exit(0)
        analyzer = ConstraintAnalyzer(impl_node, args_type, cons_trace_dir, symbol_path, extraction_mode,
                                      lib_name=library_name, llm=self.llm)
        cons_list = analyzer.traverse()
        for i, cons in enumerate(cons_list):
            with open(os.path.join(save_dir, f'cons{i}.txt'), 'w') as file:
                cons_str = "Valid Constraint:\n"
                for c in cons:
                    cons_str += c + "\n"
                file.write(cons_str)
        return cons_list, list(analyzer.invalid_cons_set)

    def run(self,
            counterpart_dir: str = "./data/working_dir/counterpart/tensorflow_torch_gpt35/counterparts",
            save_dir: str = "./data/working_dir/constraints_temp",
            mode: str = "rules",
            i_start: int = 0, i_end: int = 100
            ):
        save_dir = Path(save_dir)
        start_time = time.time()
        counter_list = os.listdir(counterpart_dir)
        if i_start != -1 and i_end != -1:
            counter_list = counter_list[i_start:i_end]
        for counter in counter_list:
            counter = json.load(
                open(os.path.join(counterpart_dir, counter), 'r'))
            api_name = utils.get_api_name(counter['function_name'])
            # if api_name == "torch.diagonal" or api_name == "torch.nn.functional.avg_pool1d":
            #     continue
            func_constraint_dir = save_dir / api_name
            if os.path.exists(func_constraint_dir):
                print(
                    f"The constraint for API: {api_name} already exists, we skip it.")
                continue
            print(f"Working on api: {api_name}")
            input_list: list[str] = counter['inputs']
            if '' in input_list:
                input_list.remove('')
            library_name = "tensorflow" if api_name.startswith("tf.") else "pytorch" if api_name.startswith(
                "torch.") else None
            if library_name is None:
                raise ValueError(
                    "Invalid func name! The api_name should start with either tf. or torch.")
            try:
                args_type, obj_dict = DynamicTypeCatcher.catch(counter)
            except Exception as e:
                print(
                    f"Failed to catch the argument type for {api_name}, the error is: {e}, we skip it.")
                continue
            logger.info(f"[LOG] The argument type is: {args_type}")
            impl_dict = counter['counterparts']
            cons_dict = {}
            invalid_cons_dict = {}
            os.makedirs(func_constraint_dir, exist_ok=True)
            for lib_name, impl in impl_dict.items():
                predicate_constraint_path = func_constraint_dir / \
                    lib_name / "predicate_constraint.json"
                learned_rules = {}
                if os.path.exists(predicate_constraint_path):
                    print(
                        "We find existing knowledge about how to map predicate to a constraint.")
                    learned_rules.update(
                        json.load(open(predicate_constraint_path, 'r')))
                cons_dict[lib_name], invalid_cons_dict[lib_name] = self.extract_constraint(
                    impl, args_type=args_type, library_name=lib_name,
                    save_dir=func_constraint_dir,
                    symbol_path=predicate_constraint_path,
                    extraction_mode=mode
                )
            print(f'======= Summary For Function: {api_name} =========')
            for lib_name, cons in cons_dict.items():
                print(f'{lib_name}: {len(cons)} path constraints')
            for lib_name, invalid_cons in invalid_cons_dict.items():
                print(f'{lib_name}: {len(invalid_cons)} invalid conditions')
            cost = self.llm.current_cost()
            print(f"The current cost for GPT usage is {cost} USD")
            if cost >= 4:
                print(
                    f"The current cost {cost} exceed the gpt usage limit: 4 USD. We need to end it to save money!!!")
        end_time = time.time()
        print(f"Total time costs: {end_time - start_time} seconds.")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--counterpart_dir", type=str,
                       default="./data/working_dir/counterpart/tensorflow_pytorch_gpt35_3_seeds/counterparts",
                       help="The directory of saved constraints")
    parse.add_argument("--save_dir", type=str, default="./data/working_dir/constraints/rules_pytorch",
                       help="The directory of saved constraints")
    parse.add_argument("--i_start", type=int, default=0,
                       help="The start index of the counterpart list")
    parse.add_argument("--i_end", type=int, default=100,
                       help="The end index of the counterpart list")
    parse.add_argument("--mode", type=str, default="rules",
                       help="The constraint extraction mode: rules or llm")
    parse.add_argument('--model_name', type=str, default="gpt-3.5-turbo")
    parse.add_argument("--library_name", type=str,
                       default="tensorflow", help="The name of library")
    parse.add_argument('--api_key_path', type=str,
                       default="./data/api_openai.key")
    flags, _ = parse.parse_known_args(sys.argv[1:])
    print(flags)
    llm = ChatGPTCall(api_key_file=flags.api_key_path,
                      model_name=flags.model_name)
    extractor = InputConstraintExtractor(llm=llm)
    extractor.run(save_dir=flags.save_dir, mode=flags.mode, counterpart_dir=flags.counterpart_dir,
                  i_start=flags.i_start, i_end=flags.i_end)
    print(f"Valid input constraint: {extractor.valid_count}")
    print(f"Invalid input constraint: {extractor.invalid_count}")

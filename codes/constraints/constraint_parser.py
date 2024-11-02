import ast
import re
from typing import Union

import lark
from lark import Lark

import utils.static_analysis_utils as static_tools
from codes.chatgpt import ChatGPTCall
from codes.prompting.Message import Message

OP_TO_STRING = {
    "eq": "==",
    "noteq": "!=",
    "is": "==",
    "isnot": "!=",
    "and": "and",
    "or": "or",
    "gt": ">",
    "lt": "<",
    "not": "not",
    "gte": ">=",
    "lte": "<=",
    "in": "in",
    "notin": "not in",
    'usub': '-'
}

TORCH_INVALID_CONS_KEYWORDS = [
    ".is_vulkan()",
    ".is_quantized()",
    ".is_sparse()",
    ".is_contiguous()",
    ".is_mkldnn()",
    ".has_value()",
    ".is_xla()",
    ".is_coalesced()",
    ".isWildcard()",
    "at.has_internal_overlap(self)",
]

TF_INVALID_CONS_KEYWORDS = [
    "tensor_util.is_tf_type",
    "isinstance",  # this may be buggy
    "context.executing_eagerly()",
    "indexed_slices.IndexedSlices",
    "ctx.forward_input_to_output_with_shape",
    "_resource_variable_type",
    "in.BitcastFrom",
    "resource_variable_ops.BaseResourceVariable",
    "sparse_tensor.SparseTensor",
    "helper.Simplify",
    "ctx->allocate_temp",
    "CopyFrom(",
    "functor.Copy",
    "DT_Variant",
    "allocate_output(0",
    "forward_input_or_allocate_output",
    "None",
]


class ConstraintParser(object):
    """
    This class is designed to parse constraints given a target expression (expected python, may be buggy for cpp expression)
    This class supports two types of constraint parser: 1) rule-based constraint parser; 2) llm-based constraint parser.
    This class also supports several helper such as input constraint validator and some static analysis helpers.
    """

    @staticmethod
    def handle_args_to_matching_eager(raw_constraint: str) -> str:
        """
        Convert _execute.args_to_matching_eager to variable.dtype in [...]
        Example:
        string = '(_attr_T, _inputs_T) = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64])'
        handle_args_to_matching_eater(string) = 'x.dtype in [...] and y.dtype in [...]'
        :param raw_constraint:
        :return:
        """
        if "= _execute.args_to_matching_eager(" not in raw_constraint:
            return raw_constraint
        try:
            node = ast.parse(raw_constraint).body[0].value
        except:
            return raw_constraint
        if not isinstance(node, ast.Call) or len(node.args) not in [3, 4]:
            return raw_constraint
        var_node = node.args[0]
        dtype_node = node.args[2]
        if not isinstance(var_node, ast.List) or not isinstance(dtype_node, ast.List):
            return raw_constraint
        assert isinstance(var_node, ast.List)
        assert isinstance(dtype_node, ast.List)
        var_list = []
        for n in var_node.elts:
            if isinstance(n, ast.Name):
                var_list.append(n.id)
            elif isinstance(n, ast.Call):
                var_list.append(ast.unparse(n))
        dtype_list = [n.attr for n in dtype_node.elts]
        cons_list = []
        for var in var_list:
            cons_list.append(f"({var}.dtype in {dtype_list})")
        if len(cons_list) == 0:
            return raw_constraint
        return "assert (" + " and ".join(cons_list) + ") is True"

    @staticmethod
    def _remove_comments(line: str):
        # Remove single-line comments
        line = re.sub(r"//.*$", "", line)
        # Remove multi-line comments
        line = re.sub(r"/\*.*?\*/", "", line, flags=re.DOTALL)
        return line

    @staticmethod
    def remove_unnecessary_spaces(code_string):
        # Remove pointers (&) and spaces before them, excluding the "&&" operator
        code_string = re.sub(r'(?<=[^\s&])\s*&(?!&)', '', code_string)

        return code_string

    @staticmethod
    def contains_invalid_cons_keyword(node_str: str, library_name: str):
        if library_name == "pytorch":
            for keyword in TORCH_INVALID_CONS_KEYWORDS:
                if keyword in node_str:
                    return True
        elif library_name == "tensorflow":
            for keyword in TF_INVALID_CONS_KEYWORDS:
                if keyword in node_str:
                    return True
        return False

    @staticmethod
    def rule_based_constraint_parser(raw_constraint: str):
        """
        :param raw_constraint:
        :return:
        """
        rules = {
            ".get_shape()": ".shape",
            ".shape()": ".shape",
            ".num_element()": ".num_element",
            ".shape.ndims": ".ndims",
            ".shape.rank": '.ndims',
            ".dim()": ".ndims",
            ".dims()": ".ndims",
            ".numel()": ".num_element",
            ".NumElements()": ".num_element",
            " && ": " and ",
            " || ": " or ",
            "::": ".",
            # "\n": ""
        }
        raw_constraint = ConstraintParser.handle_args_to_matching_eager(raw_constraint)  # This is to handle sanity check statement 'args_to_matching_eager'
        # remove CPP comments
        raw_constraint = ConstraintParser._remove_comments(raw_constraint)
        # need to replace ndim with ndims
        raw_constraint = re.sub(r'\bndim\b', 'ndims', raw_constraint)

        def _handle_non_str_dtype(cons: str):
            import re
            from codes.constraints.constraint_solver import Tensor
            # constraints_bool = x.dtype == bool -> constraints_bool = x.dtype == 'bool'
            if "dtype" in cons:
                for dtype in Tensor.dtype_list:
                    cons = re.sub(rf"(?<![\"'])\b{dtype}\b(?![\"'])", f"'{dtype}'", cons)
            return cons

        raw_constraint = _handle_non_str_dtype(raw_constraint)

        def is_mutable(string):
            mutable = False
            for record in rules.keys():
                if f"{record}" in string:
                    mutable = True
            return mutable

        if not is_mutable(raw_constraint):
            return raw_constraint
        """
        TODO [Not urgent]: Maybe we can use re.sub like below to replace these string
        pattern = rf'\b{variable}\b'
        origin_expr = re.sub(pattern, value, origin_expr)
        """
        # Apply the mutation operators recursively
        constraint: str = raw_constraint
        for key, value in rules.items():
            constraint = constraint.replace(key, value)
        # Recursively apply mutations
        return ConstraintParser.rule_based_constraint_parser(constraint)

    @staticmethod
    def z3_valid_constraint(constraint: str, input_list: list[str], args_type_dict: dict[str, str] = None):
        """
        Some test cases can be found at ./tests/TestConstraintParser.test_valid_constraint_*
        """
        if not constraint.startswith("assert"):
            constraint = f"assert {constraint}"
        if isinstance(constraint, str):
            try:
                constraint_node = ast.parse(constraint).body[0]
            except:
                return False
        if static_tools.is_py_literal(constraint_node):
            return True
        # We first check if the constraint has variables other than variables in the input list
        var_list = static_tools.find_py_expr_vars(constraint_node, [])
        for var in var_list:
            if var not in input_list and f"[{var}]" not in ast.unparse(constraint_node):
                # if the var is in subscript, such as the 'i' in image.shape[i] we do not consider the constraint invalid because we consider the FIELD-Insensitive Analysis.
                return False
        from codes.constraints.constraint_solver import ConstraintSolver
        try:
            # For some constraint, its validity depends on the condition res:
            # x.shape == x.shape is False
            # assert (input_ndims == input_ndims) is False
            # assert (n_shape[0] == n_shape[0]) is False
            # assert (n_num_element * m_num_element < 1000) is False
            # We consider a constraint valid regardless of its condition res, thus we evaluate on both
            true_valid = ConstraintSolver(args_type_dict=args_type_dict).validate_constraint(f"{constraint} is True")
            false_valid = ConstraintSolver(args_type_dict=args_type_dict).validate_constraint(f"{constraint} is False")
            return true_valid and false_valid
        except Exception:
            print(f"Error in z3_valid_constraint: {constraint}")
            return False

    @staticmethod
    def constraint_targets_input(constraint: str, input_list: [str]) -> [str]:
        """
        Check if the constraint contains variable inside the input_list, if it contains, it returns the inputs targeted
        :param constraint:
        :param input_list:
        :return:
        """
        try:
            # we dynamically check if the expression can be parsed by python's ast, otherwise, it should be parsed by tree-sitter.
            node = ast.parse(constraint).body[0]
            language_type = "py"
        except:
            try:
                node = static_tools.parser.parse(bytes(constraint, "utf8")).root_node.children[0].children[0]
                language_type = ""
            except:
                return []
        var_list = static_tools.find_expr_vars(node, variable_list=[], language_type=language_type)
        filter_var_list = []
        for var in var_list:
            if var in input_list and var not in filter_var_list:
                filter_var_list.append(var)
        return filter_var_list

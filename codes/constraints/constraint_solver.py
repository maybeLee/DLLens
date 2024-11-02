import ast
import random

import numpy as np
import z3

from utils.utils import is_py_variable
from codes.logger.logger import logger
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ConstraintSolver:
    def __init__(self, args_type_dict=None):
        """
        A constraint solver
        """

        self.args_type_dict = {} if args_type_dict is None else args_type_dict
        self.solver = z3.Solver()
        self.z3_constraint_expr: list[str] = []
        self.var_list = {}  # variable (e.g., image_ndims) constrained

    @staticmethod
    def preprocess_constraint(constraint: str) -> str:
        """
        image.ndims -> image_ndims
        :param constraint:
        :return:
        """
        attr_list = [
            'shape', 'dtype', 'ndims',
            'num_element', 'value', 'tensor_type',
            "intmin", "intmax"  # min and max are only used during error catch, will not be used in the path constraint extraction
            ]
        for attr in attr_list:
            constraint = constraint.replace(f".{attr}", f"_{attr}")

        def _handle_shape_i(cons: str):
            # we also need to handle the shape[i] case to field insensitive, temporarily, we simply replace shape[i] with shape[0]
            if "_shape[i]" in cons:
                logger.warning(
                    f"_shape[i] is found in constraint, we simply replace it with shape[0]. It may has some error.")
            cons = cons.replace(f"_shape[i]", "_shape[0]")
            return cons

        constraint = _handle_shape_i(constraint)
        if not constraint.startswith("assert"):
            constraint = f"assert {constraint}"
        # Remove the token using slicing
        c = constraint[len("assert"):].strip()
        condition = " ".join(c.split(" ")[:-2])
        compare_res = constraint.split(" ")[-1]
        new_constraint = f"assert ({condition}) is {compare_res}"
        new_constraint = ConstraintSolver.constraint_unification(new_constraint)
        try:
            return ast.unparse(ast.parse(new_constraint))
        except:
            return new_constraint

    @staticmethod
    def find_vars(node):
        var_list = []
        for n in ast.walk(node):
            if is_py_variable(n):
                var_list.append(ast.unparse(n))
        return var_list

    @staticmethod
    def constraint_unification(constraint: str) -> str:
        """
        Do some unification on the string constraints
        :param constraint:
        :return:
        """
        constraint = constraint.replace("complexfloat", "complex64")
        constraint = constraint.replace("complexdouble", "complex128")
        constraint = constraint.replace("cfloat", "complex64")
        constraint = constraint.replace("cdouble", "complex128")
        constraint = constraint.replace("\'float\'", "\'float32\'")
        constraint = constraint.replace("\"float\"", "\"float32\"")
        constraint = constraint.replace("\'complex\'", "\'complex64\'")
        constraint = constraint.replace("\"complex\"", "\"complex64\"")
        return constraint

    @staticmethod
    def return_properties(cons_list: list[[str]]) -> list:
        """
        Only For Evaluation
        :param cons_list:
        :return:
        """
        total_var_list = []
        for path in cons_list:
            for cons in path:
                constraint = ConstraintSolver.preprocess_constraint(cons)
                try:
                    node = ast.parse(constraint).body[0]
                except:
                    return total_var_list
                assert isinstance(node, ast.Assert) and isinstance(
                    node.test, ast.Compare)
                compare_res: bool = node.test.comparators[0].value
                condition = node.test.left
                if not isinstance(condition, (ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Name)):
                    return total_var_list
                var_list = ConstraintSolver.find_vars(condition)
                for var in var_list:
                    if var not in total_var_list:
                        total_var_list.append(var)
        return total_var_list

   
if __name__ == "__main__":
    ...

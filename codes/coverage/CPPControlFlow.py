# This class is built based on tree-sitter (https://tree-sitter.github.io/tree-sitter/)'s ast parser
import os.path

import tree_sitter
from tree_sitter import Language, Parser

from codes.coverage.ControlFlow import CFGNode, graph_to_path

if not os.path.exists("build/my-languages.so") and os.path.exists("vendor/tree-sitter-cpp"):
    Language.build_library(
        # Store the library in the `build` directory
        'build/my-languages.so',

        # Include one or more languages
        [
            'vendor/tree-sitter-cpp',
        ]
    )

CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')
parser = Parser()
parser.set_language(CPP_LANGUAGE)

py_parser = tree_sitter.Parser()
if not os.path.exists("./build/python-language.so") and os.path.exists("vendor/tree-sitter-python"):
    tree_sitter.Language.build_library(
        # Store the library in the `build` directory
        'build/python-language.so',

        # Include one or more languages
        [
            'vendor/tree-sitter-python',
        ]
    )

PY_LANGUAGE = tree_sitter.Language("./build/python-language.so", "python")
py_parser.set_language(PY_LANGUAGE)

REGISTRY_IDX = 0

REGISTRY = {}


def get_registry_idx():
    global REGISTRY_IDX
    v = REGISTRY_IDX
    REGISTRY_IDX += 1
    return v


def reset_registry():
    global REGISTRY_IDX
    global REGISTRY
    REGISTRY_IDX = 0
    REGISTRY = {}


def register_node(node):
    node.rid = get_registry_idx()
    REGISTRY[node.rid] = node


def get_registry():
    return dict(REGISTRY)


class CppCFGNode(CFGNode):
    def __init__(self, parents: list = [], ast: tree_sitter.Node = None):
        assert type(parents) is list
        register_node(self)
        self.parents = parents
        self.ast_node = ast
        self.update_children(parents)  # requires self.rid
        self.children = []
        self.calls = []
        self.language_type = "cc"

    def lineno(self):
        # override existing method
        return self.ast_node.start_point[0] if hasattr(self.ast_node, "start_point") else 0

    def source(self):
        return self.ast_node.text.decode()


class CppCFG(object):
    def __init__(self):
        self.founder = CppCFGNode(parents=[], ast=parser.parse(bytes("""start""", "utf8")).root_node)
        # self.founder.ast_node.lineno = 0
        self.functions = {}
        self.functions_node = {}

    def get_defining_function(self, node):
        if node.lineno() in self.functions_node:
            return self.functions_node[node.lineno()]
        if not node.parents:
            self.functions_node[node.lineno()] = ''
            return ''
        val = self.get_defining_function(node.parents[0])
        self.functions_node[node.lineno()] = val
        return val

    def update_functions(self):
        for nid, node in REGISTRY.items():
            _n = self.get_defining_function(node)

    def update_children(self):
        for nid, node in REGISTRY.items():
            for p in node.parents:
                p.add_child(node)

    @staticmethod
    def parse(src: str) -> tree_sitter.Node:
        return parser.parse(bytes(src, "utf8")).root_node

    def walk(self, node: tree_sitter.Node, myparents):
        fname = "on_%s" % node.type.lower()
        if hasattr(self, fname):
            fn = getattr(self, fname)
            v = fn(node, myparents)
            return v
        else:
            return myparents

    def on_translation_unit(self, node: tree_sitter.Node, myparents):
        """
        TopLevelCode(node* children)
        :param node:
        :param myparents:
        :return: node
        Each time a unit inside the code is executed, make a link from the result to next unit
        """
        p = myparents
        for n in node.children:
            p = self.walk(n, p)
        return p

    def on_struct_specifier(self, node: tree_sitter.Node, myparents):
        """
        Struct(node* children)
        :param node:
        :param myparents:
        :return: node
        Each time a unit inside the struct object is executed, make a link from the result to next unit
        """
        p = myparents
        for n in node.children:
            p = self.walk(n, p)
        return p

    def on_compound_statement(self, node: tree_sitter.Node, myparents):
        """
        CompoundStatement(node* children)
        :param node:
        :param myparents:
        :return: node
        Each time a statement inside the compound statement is executed, make a link from the result to next statement
        """
        p = myparents
        for n in node.children:
            p = self.walk(n, p)
        return p

    def on_declaration(self, node: tree_sitter.Node, myparents):
        """
        declarationNode(node* children, **init_declarator), walk on its init_declarator for more expression
        :param node:
        :param myparents:
        :return:
        """
        declarator: tree_sitter.Node = node.child_by_field_name('declarator')
        value_node: tree_sitter.Node = declarator.child_by_field_name('value')
        if not value_node:
            p = [CppCFGNode(parents=myparents, ast=node)]
            return p
        if value_node.type == 'lambda_expression':
            # we don't list all the logics in the lambda expression for an declaration statement.
            p = [
                CppCFGNode(parents=myparents,
                           ast=parser.parse(bytes(
                               f'{value_node.child_by_field_name("captures").text.decode()}'
                               f'{value_node.child_by_field_name("declarator").text.decode()}', 'utf8')
                           ).root_node.children[0]
                           )]
        if value_node.type == "conditional_expression":
            try:
                # print(f"Entering this function, the node is: {node.text.decode()}")  # [DEBUG]
                # Tackle code like this: const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
                condition_node = value_node
                target: str = declarator.child_by_field_name('declarator').text.strip().decode()
                return self.build_conditional_node(target, condition_node, myparents)
            except:
                pass
        else:
            p = [CppCFGNode(parents=myparents, ast=node)]
        if declarator:
            p = self.walk(value_node, p)
        return p

    def on_lambda_expression(self, node: tree_sitter.Node, myparents):
        """
        LambdaNode(*body), walk through its body
        :param node:
        :param myparents:
        :return:
        """
        p = myparents
        for n in node.children:
            p = self.walk(n, p)
        return p

    def on_expression_statement(self, node: tree_sitter.Node, myparents):
        """
        ExpressionNode(node* unit)
        walk on the expression
        :param node:
        :param myparents:
        :return:
        """
        if node.children[0].type == "assignment_expression" and node.children[0].child_by_field_name(
                "right").type == "conditional_expression":
            # Tackle code like this: log_abs_det = log_abs_det > 0 ? -std::log(RealScalar(0)) : std::log(RealScalar(0))
            assign_node = node.children[0]
            condition_node = node.children[0].child_by_field_name("right")
            target: str = assign_node.child_by_field_name("left").text.strip().decode()
            return self.build_conditional_node(target, condition_node, myparents)
        if node.children[0].type == "call_expression":
            # if it is an call expression, we don't blindly add it into our path
            arg_has_lambda = False
            call_node = node.children[0]
            for arg in call_node.child_by_field_name("arguments").children:
                if arg.type == "lambda_expression":  # check if the argument has lambda expressipn
                    arg_has_lambda = True
            if arg_has_lambda is False:
                p = [CppCFGNode(parents=myparents, ast=call_node)]
            else:
                # if the argument is an lambda expression, we don't include this call expression
                p = myparents
            return self.walk(node.children[0], p)
        p = [CppCFGNode(parents=myparents, ast=node)]
        return self.walk(node.children[0], p)

    @staticmethod
    def build_conditional_node(target: str, condition_node: tree_sitter.Node, myparents: [CppCFGNode]) -> [CppCFGNode]:
        condition: str = condition_node.child_by_field_name("condition").text.strip().decode()
        _test_node = CppCFGNode(
            parents=myparents,
            ast=parser.parse(bytes(
                '_if: %s' % condition,
                'utf8')
            ).root_node.children[0]
        )
        g1 = CppCFGNode(
            parents=[_test_node],
            ast=parser.parse(bytes(
                f"{target} = {condition_node.child_by_field_name('consequence').text.decode()}",
                'utf8')
            ).root_node.children[0]
        )
        g2 = CppCFGNode(
            parents=[_test_node],
            ast=parser.parse(bytes(
                f"{target} = {condition_node.child_by_field_name('alternative').text.decode()}",
                'utf8')
            ).root_node.children[0]
        )
        return [g1, g2]

    def on_conditional_expression(self, node: tree_sitter.Node, myparents):
        return self.build_conditional_node(target="_", condition_node=node, myparents=myparents)

    def on_return_statement(self, node: tree_sitter.Node, myparents):
        """
        ReturnNode(node*, unit)
        :param node:
        :param myparents:
        :return:
        """
        p = [CppCFGNode(parents=myparents, ast=node)]
        # return self.walk(node.children[0], p)
        return []

    def on_call_expression(self, node: tree_sitter.Node, myparents):
        """
        If we find the call expression in the statement, we iterate each argument for case where logics are stored inside the argument
        :param node:
        :param myparents:
        :return:
        """
        p = myparents
        for n in node.children:
            p = self.walk(n, p)
        return p

    def on_argument_list(self, node: tree_sitter.Node, myparents):
        """
        If we find the argument_list in the statement, we iterate each argument for case where logics are stored inside the argument
        :param node:
        :param myparents:
        :return:
        """
        p = myparents
        for n in node.children:
            p = self.walk(n, p)
        return p

    def on_if_statement(self, node: tree_sitter.Node, myparents):
        """
        IfStmt(*condition_clause, *statements, )

        :param node:
        :param myparents:
        :return:
        """
        condition = node.child_by_field_name('condition').text.strip().decode()
        _test_node = CppCFGNode(
            parents=myparents,
            ast=parser.parse(bytes(
                '_if: %s' % condition,
                'utf8')
            ).root_node.children[0]
        )
        test_node = self.walk(node.child_by_field_name('condition'), [_test_node])
        g1 = test_node
        g1 = self.walk(node.child_by_field_name('consequence'), g1)
        g2 = test_node
        orelse: tree_sitter.Node = node.child_by_field_name('alternative')
        if orelse:
            g2 = self.walk(orelse.children[-1], g2)
        return g1 + g2

    def on_for_statement(self, node: tree_sitter.Node, myparents):
        # for loop in cpp file, we simply unfold it.
        p = myparents
        for n in node.children:
            p = self.walk(n, p)
        return p

    def on_for_range_loop(self, node: tree_sitter.Node, myparents):
        # for loop in cpp file, we simply unfold it.
        p = myparents
        for n in node.children:
            p = self.walk(n, p)
        return p

    def on_while_statement(self, node: tree_sitter.Node, myparents):
        # for loop in cpp file, we simply unfold it.
        p = myparents
        for n in node.children:
            p = self.walk(n, p)
        return p

    def on_for(self, node: tree_sitter.Node, myparents):
        return myparents

    def on_while(self, node: tree_sitter.Node, myparents):
        return myparents

    def gen_cfg(self, src):
        """
        >>> i = PyCFG()
        >>> i.walk("100")
        5
        """
        node = self.parse(src)
        nodes = self.walk(node, [self.founder])
        self.last_node = CppCFGNode(parents=nodes, ast=parser.parse(bytes("""stop""", "utf8")).root_node)
        self.update_children()
        self.update_functions()
        # self.link_functions()


def gen_cfg(fnsrc, remove_start_stop=True):
    reset_registry()
    cfg = CppCFG()
    cfg.gen_cfg(fnsrc)
    cache = dict(REGISTRY)
    if remove_start_stop:
        return {
            k: cache[k]
            for k in cache if cache[k].source() not in {'start', 'stop'}
        }
    else:
        return cache


def gen_cc_path(src: str) -> list[list[CFGNode]]:
    """
    Get the cc execution path based on a function's code. Steps to do so:
    1. Get the directed graph.
    2. Find all paths based on the graph
    3. Draw the graph
    :param src:
    :return:
    """
    from collections import defaultdict
    cfg = gen_cfg(src, remove_start_stop=False)


    def get_all_paths(fenter, depth=0):
        if not fenter.children:
            return [[fenter]]

        fnpaths = []
        for idx, child in enumerate(fenter.children):
            child_paths = get_all_paths(child, depth + 1)
            for path in child_paths:
                # In a conditional branch, idx is 0 for IF, and 1 for Else
                fnpaths.append([fenter] + path)
        return fnpaths

    return get_all_paths(list(cfg.values())[0])
    # # get the graph (customized for python and cc file)
    # graph: dict[int, list[int]] = defaultdict(list)  # {cnode1_id: [cnode2_id, cnode3_id, ...]}
    # s_id: list[int] = []
    # e_id: list[int] = []
    # for nid, cnode in cfg.items():
    #     if cnode.source() == 'start':
    #         s_id.append(nid)
    #     if cnode.source() == 'stop':
    #         e_id.append(nid)
    #     for pn in cnode.parents:
    #         graph[int(pn.i())].append(nid)
    # return graph_to_path(graph, s_id, e_id, cfg)


def gen_cc_input(root_node: tree_sitter.Node) -> [tuple]:
    """
    Get the cpp input based on a function's code.
    :param root_node: the source code of the function
    :return: [(arg), (arg, value), ...]
    """
    input_list: [tuple] = []

    def gen_input(node: tree_sitter.Node):
        if node.type == "function_declarator":
            if len(input_list) != 0:
                return
            parameters_list: tree_sitter.Node = node.child_by_field_name('parameters')
            for child in parameters_list.children:
                if child.type == "parameter_declaration" or child.type == "optional_parameter_declaration":
                    declarator: tree_sitter.Node = child.child_by_field_name('declarator')
                    if declarator is None:
                        input_list.append(child.text.decode())
                        continue
                    if declarator.type == "pointer_declarator":
                        declarator = declarator.child_by_field_name('declarator')
                    elif declarator.type == "reference_declarator":
                        declarator = declarator.children[-1]
                    if child.type == "optional_parameter_declaration":
                        default_value: tree_sitter.Node = child.child_by_field_name('default_value')
                        input_list.append((declarator.text.decode(), default_value.text.decode()))
                    else:
                        input_list.append((declarator.text.decode(),))
        # Recursively print all the node's children
        for child in node.children:
            gen_input(child)

    gen_input(root_node)
    return input_list


def get_cc_func_name(root_node: tree_sitter.Node) -> str:
    """
    Get the cpp function name based on a function's code.
    :param root_node: the source code of the function
    :return: the function name
    """
    func_name = []

    def gen_func_name(node: tree_sitter.Node):
        if len(func_name) >= 1:
            return
        if node.type == "function_declarator":
            # we also need to handle: the TORCH_IMPL_FUNC(linalg_cross_out) case,
            # where the actual function name should be `linalg_cross_out`
            declarator = node.child_by_field_name("declarator")
            if declarator.type == "function_declarator" and len(declarator.child_by_field_name("parameters").children) == 3:
                # the parameter list would be thing like: (linalg_cross_out)
                param = declarator.child_by_field_name("parameters").children[1]
                func_name.append(param.text)
            elif declarator.type == "qualified_identifier" and declarator.child_by_field_name("name") is not None:
                func_name.append(declarator.child_by_field_name("name").text)
            else:
                func_name.append(declarator.text)
        # Recursively print all the node's children
        for child in node.children:
            gen_func_name(child)

    gen_func_name(root_node)
    if len(func_name) > 0:
        return func_name[0].decode()
    else:
        return ""

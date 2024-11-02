system_msg = """
You are an expert in TensorFlow and PyTorch and you are familiar with their APIs/functions. Your task is to analyze the given execution path and extract constraints from the provided condition.
"""

condition_solver_template = """
Analyze the following execution path, summarize the NECESSARY constraint on the `attribute` of input arguments {} to satisfy the condition at the end of the path.

[Execution Path]
{}

[argument type]
{}

{}

The constraint should only consider symbols {}, do not include any other operations except logical expressions.
Do not output any constraint if it is unrelated to the attribute.

Output the constraint in one python boolean expression. 
Return nothing if you find no constraints related to input arguments.
[Output Format]
```
CONSTRAINT
```
"""

condition_solver_template_v2 = """
Here is one execution path
[Execution Path]
[EXECUTION_PATH]

After executing the this path, related variables need to satisfy the following condition:

[Condition]
[CONDITION]

You need to conduct the following steps:

1. Understand what constraint is required for related variables to pass this condition. 
2. Analyze how input argument(s) [INPUT_ARGUMENTS] influence related variables through the execution path.
3. Summarize the NECESSARY constraint on [INPUT_ARGUMENTS] so this condition can be satisfied. The constraint should only consider symbols [INPUT_ARGUMENTS] and related attributes, do not include any other operations except logical expressions.
To help you summarize the constraint, here are types and attributes of these arguments.

[argument type]
[ARGUMENT_TYPE]

[ARGUMENT_ATTRIBUTE]

- If you find some parts of the given condition is not related to argument attributes, just ignore these parts.
- Carefully think about what attributes and arguments are explicitly specified from the given condition.
- Output the constraint in one python boolean expression. Do not output other text except the boolean expression.
- Return `Invalid Constraint` if you find no constraints related to input arguments or you find the condition is not solvable.

[Output Format]
```
FILL_IN_YOUR_ANSWER
```
"""

condition_reflect_template = """
Review the assistant analysis above carefully; consider the following:
- The generated constraint should be a boolean expression that only includes the symbols [INPUT_ARGUMENTS] and their attributes.
- If the constraint includes attributes or arguments that are not explicitly required by the given condition, you should not include them in the final answer.
- If the argument and argument name is correct.
Thinking step by step, conclude a correct and comprehensive answer
[Output Format]
```
FILL_IN_YOUR_ANSWER
```
"""

invalid_reflect_template = """
Review the assistant analysis above carefully, check if the analyzed condition is indeed invalid.
Thinking step by step, return "Invalid Constraint" if the condition is indeed invalid, otherwise, return the valid constraint.
[Output Format]
```
FILL_IN_YOUR_ANSWER
```
"""

cons_error_feedback_query = """
We find the following exception when evaluating your generated constraint: [CONDITION].
[Error Message(s)]
```
[ERROR]
```
Repair your previous answer.
[Output Format]
```
FILL_IN_YOUR_ANSWER
```
"""

tensor_attribute = """
.ndims: int, number of dimensions of tensor
.shape: [int], shape of tensor
.shape[i]: int, specific index of the shape
.dtype: str, data type of tensor, e.g., float32
.num_element: int, total number of elements in tensor.
"""

def fill_condition_solver_query(trace, controllable_input_list, args_type):
    controllable_input_list = [f"`{arg}`" for arg in controllable_input_list]
    argument = ', '.join(controllable_input_list)
    t_args_type = [f"{arg}: {args_type[arg.strip('`')]}" for arg in controllable_input_list]

    attribute_list = []
    for arg in controllable_input_list:
        if args_type[arg.strip('`')].lower() in ["tensor", "float", "int"]:
            attribute_list.append(f"[`attribute` for {arg}]{tensor_attribute}")
    attribute_str = "\n".join(attribute_list)
    if len(controllable_input_list) == 1:
        symbol_expr = f"{controllable_input_list[0]} and its attributes"
    else:
        symbol_expr = f"{', '.join(controllable_input_list)} and their attributes"
    return condition_solver_template.format(argument, trace, t_args_type, attribute_str, symbol_expr)


def fill_condition_solver_query_v2(trace, controllable_input_list, args_type):
    controllable_input_list_ = [f"`{arg}`" for arg in controllable_input_list]
    condition = trace.strip().split("\n")[-1].split("condition: ")[-1]
    execution_path = "\n".join(trace.split("\n")[:-1])
    input_arguments = ", ".join(controllable_input_list_)
    t_args_type = [f"{arg}: {args_type[arg.strip('`')]}" for arg in controllable_input_list_]
    tensor_inputs = [arg for arg in controllable_input_list_ 
                     if args_type[arg.strip('`')].lower() in ["tensor", "float", "int"]]
    tensor_str = ', '.join(tensor_inputs)
    if len(tensor_inputs) != 0:
        argument_attr = f"[`attribute` for {tensor_str}]\n{tensor_attribute}"
    else:
        argument_attr = ""
    return condition_solver_template_v2\
        .replace("[EXECUTION_PATH]", execution_path)\
        .replace("[CONDITION]", condition)\
        .replace("[INPUT_ARGUMENTS]", input_arguments)\
        .replace("[ARGUMENT_ATTRIBUTE]", argument_attr)\
        .replace("[ARGUMENT_TYPE]", str(t_args_type))


def fill_condition_reflect_query(controllable_input_list, previous_ans):
    controllable_input_list_ = [f"`{arg}`" for arg in controllable_input_list]
    input_arguments = ", ".join(controllable_input_list_)
    if previous_ans.lower().strip() == "invalid constraint":
        return invalid_reflect_template
    return condition_reflect_template.replace("[INPUT_ARGUMENTS]", input_arguments)

def fill_constraint_error_feedback_query(error, condition):
    return cons_error_feedback_query.replace("[ERROR]", error).replace("[CONDITION]", condition)

if __name__ == "__main__":
    trace = """
y_true = y_true
y_pred = y_pred
y_pred = tf.convert_to_tensor(y_pred)
y_true = tf.cast(y_true, y_pred.dtype)
y_true = backend.clip(y_true, backend.epsilon(), 1)
y_pred = backend.clip(y_pred, backend.epsilon(), 1)
input_tensor = y_true * tf.math.log(y_true / y_pred)
x = input_tensor
x_rank = x.shape.rank
condition: x_rank
"""
    controllable_input_list = ["y_true", "y_pred"]
    args_type = {"y_true": "Tensor", "y_pred": "Tensor"}
    print(fill_condition_solver_query_v2(trace, 
                                        controllable_input_list,
                                        args_type))

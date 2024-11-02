import numpy as np

system_msg = """
You are an expert in TensorFlow and PyTorch and you are familiar with their APIs and their functionalities. 
"""

counterpart_query = """
Given the following sample inputs, your task is to generate a function named `[DST_LIB]_call` that uses the [DST_LIB]'s API.
The generated function SHOULD have the same functionality as the function `[SRC_LIB]_call` using the [SRC_LIB]'s API. 
In `[DST_LIB]_call`, you SHOULD NOT include APIs from external packages beyond [DST_LIB].

[Sample Inputs]
```
[SAMPLE_INPUTS]
```
[Function using [SRC_LIB] API]
```
[SRC_LIB_FUNC]
```
[Function using [DST_LIB] API]
```
CODE
```
"""

input_error_feedback_query = """
Your previous input is invalid. The following error occurs. 
Repair your previous input, avoid violating the API's input constraint.
[Error Message]
```
{}
```
[Output Format]
```
CODE
```
"""

counterpart_feedback_query = """
{}

You should carefully analyze above information.
{}
Based on your analysis, refine your generated function.

[Output Format]
```
CODE
```
"""

def construct_feedback_query(crash_res, incon_res):
    crash_info, incon_info = "", ""
    crash_action, incon_action = "", ""
    if len(crash_res) > 0:
        crash = np.random.choice(crash_res)
        crash_info = f"The generated function crashes on the following input:\n"\
                     f"{crash.feedback_query().strip()}"
        crash_action = f"For crashes, check the error message then refine the function."
    if len(incon_res) > 0:
        incon = np.random.choice(incon_res)
        incon_info = f"The generated function produces incorrect output on the following input:\n"\
                     f"{incon.feedback_query().strip()}"
        incon_action = f"For inconsistent output, analyze the difference between the [Expected Output] and the [Actual Output].\n"\
                       f"If the output shape is mismatch, you should first adjust the function to make the output matches.\n"\
                       f"If the output value is mismatch, you should carefully analyze the difference and refine the function."
    return counterpart_feedback_query.format(crash_info + "\n" + incon_info, crash_action + "\n" + incon_action)


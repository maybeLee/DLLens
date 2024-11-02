sample_input_query_1 = """
Task 1: Import {}
Task 2: Generate valid parameter. DO NOT use APIs such as {} to generate random number. The name of parameter variables should be: {}
Task 3: call the function:
```
{}
```
In your code, avoid using TensorFlow and PyTorch APIs except for `tf.constant` and `torch.tensor`. 
If the parameter is a tensor, the size should smaller than 10.
Wrap the program with ``` symbol
"""
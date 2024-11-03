"""
Library version: torch (2.1.0)

Crash happens when calling addcdiv_/addcmul_:
ERROR:!needs_dynamic_casting<func_t>::check(iter) INTERNAL ASSERT FAILED at "../aten/src/ATen/native/cpu/Loops.h":349, please report a bug to PyTorch.
"""

import torch
results = dict()
_input_tensor_tensor = torch.rand([5, 5], dtype=torch.bfloat16)
_input_tensor = _input_tensor_tensor.clone()
tensor1_tensor = torch.rand([5, 5], dtype=torch.float64)
tensor1 = tensor1_tensor.clone()
tensor2_tensor = torch.rand([5, 5], dtype=torch.float64)
tensor2 = tensor2_tensor.clone()
try:
  results["res_1"] = _input_tensor.addcmul_(tensor1, tensor2, )
except Exception as e:
  results["err_1"] = "ERROR:"+str(e)
try:
  results["res_2"] = _input_tensor_tensor.clone().addcdiv_(tensor1_tensor.clone(),tensor2_tensor.clone(),)
except Exception as e:
  results["err_2"] = "ERROR:"+str(e)

print(results)

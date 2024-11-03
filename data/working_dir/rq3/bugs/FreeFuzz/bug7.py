"""
Library version: torch (2.1.0)

When divide int64 by bool 'False', cpu gives ZeroDivisionError, gpu gives a value

Comparison result:
{'err_cpu': 'ERROR:ZeroDivisionError', 'res_gpu': tensor([[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2]], device='cuda:0')}
"""

results = dict()
import torch
arg_1_tensor = torch.randint(-2048,1,[1, 10], dtype=torch.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = False
try:
  results["res_cpu"] = torch.floor_divide(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.floor_divide(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)

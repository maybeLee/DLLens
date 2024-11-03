"""
Library version: torch (2.1.0)

Inconsistent result between cpu & cuda when dividing a complex128 tensor by 0. Here is the output

Comparison result:
{'res_cpu': tensor(inf+infj, dtype=torch.complex128), 'res_gpu': tensor(nan+nanj, device='cuda:0', dtype=torch.complex128)}
"""

results = dict()
import torch
arg_1_tensor = torch.rand([], dtype=torch.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
try:
  results["res_cpu"] = torch.div(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.div(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)

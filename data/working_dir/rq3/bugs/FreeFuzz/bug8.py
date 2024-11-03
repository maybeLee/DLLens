"""
Library version: torch (2.1.0)

torch.logit should return nan according to doc (https://pytorch.org/docs/stable/special.html#torch.special.logit), but cpu returns inf

Comparison result:
{'res_cpu': tensor([inf, inf, inf, inf, inf], dtype=torch.bfloat16), 'res_gpu': tensor([nan, nan, nan, nan, nan], device='cuda:0', dtype=torch.bfloat16)}
"""

results = dict()
import torch
arg_1_tensor = torch.rand([5], dtype=torch.bfloat16)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.000001
try:
  results["res_cpu"] = torch.logit(arg_1,eps=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.logit(arg_1,eps=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)


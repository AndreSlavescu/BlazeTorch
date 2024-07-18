import torch
import torch.testing as tt

# BlazeTorch backend
import blazetorch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

blazetorch_result = blazetorch.add(a, b)
torch_result = torch.add(a, b)

tt.assert_close(blazetorch_result, torch_result, rtol=1e-5, atol=1e-5)
print("Results Match!")
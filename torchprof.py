import torch
import torchvision.models as models
from torch.profiler import profile
from torch.profiler import record_function
from torch.profiler import ProfilerActivity

# model = models.resnet18()
# input = torch.randn(5, 3, 224, 224)

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(input)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

import timm
from torch import nn
import torch
from torch.cuda.amp import autocast
import time
from tqdm.auto import tqdm
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="fp8")

p = timm.create_model("resnet50", pretrained=False)
random_tensor = torch.rand(32, 3, 384, 384)
p, random_tensor = accelerator.prepare(p, random_tensor)
# warmup run
for i in range(10):
        p(random_tensor)
# main run
start = time.time()
for i in tqdm(range(1000), disable=not accelerator.is_local_main_process, ):
        p(random_tensor)
print(f"Time taken in seconds: {time.time() - start}")

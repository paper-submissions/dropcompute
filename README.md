# DropCompute

This code was used to implement: DropCompute: simple and more robust distributed synchronous training via compute variance reduction (2023)

This repository is based on the Habana Model-References repository https://github.com/HabanaAI/Model-References/tree/1.9.0

## General usage

To apply drop-compute, use the compute_timer.py module:

1) Import the compute_timer module in your code:
```
from compute_timer import compute_time
```
2) Wrap the iteration loop in try-catch block
3) Add compute timeout check using the compute_timer module
```
compute_timer = compute_time.DeviceTimer()

for input, target in batch_dataloader:
  try:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    compute_timer.check_drop_compute_throw()

  except ComputeTimeout:
    print('compute dropped')
    break
    
take_optimization_step()
```
Please refer to [BERT experiments](../master/deepspeed-bert) for a detailed implementation.

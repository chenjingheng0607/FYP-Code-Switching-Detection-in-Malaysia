import torch

# Check if a GPU is available
if torch.cuda.is_available():
    print("PyTorch is using a GPU.")
    # Print the name of the GPU
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    # Get the number of GPUs
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print (torch.cuda.memory_allocated())
else:
    print("PyTorch is using the CPU.")

# Check the device of a specific tensor (assuming you have one)
# my_tensor = torch.rand(10).to("cuda") # Example of moving a tensor to GPU
# print(f"Tensor device: {my_tensor.device}")
import torch
import blazetorch  # Ensure the custom backend library is loaded

# Use the custom add function from the custom backend
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Call the custom add function
result = blazetorch.add(a, b)

# Print the result
print(f"Result of custom_add: {result}")

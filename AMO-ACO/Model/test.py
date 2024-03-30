import torch
from torch.nn import functional as F
import math
import numpy as np
import matplotlib.pyplot as plt 

# a = torch.tensor([[1,2,3], [0.5,5,6], [7,8,9]], dtype = torch.float32)
# b = torch.tensor([[0,0,0], [0,0,0], [0,1,0]], dtype = torch.float32)
# c = torch.tensor([1,3,5,7,9])

import torch
import numpy as np

# Tạo tensor mẫu
du_lieu_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Chuyển đổi tensor thành mảng NumPy
# du_lieu_numpy = du_lieu_tensor.numpy()

# Ghi mảng NumPy vào tệp văn bản
ten_tep = "du_lieu.txt"
np.savetxt(ten_tep, du_lieu_tensor)

print("Dữ liệu đã được ghi vào tệp văn bản.")




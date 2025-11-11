import torch
print(torch.cuda.is_available())      # True nếu có GPU
print(torch.cuda.device_count())      # số GPU
print(torch.cuda.get_device_name(0))  # tên GPU số 0 (nếu có)
import os

path = r"C:\Users\ploc\Downloads\TabRecSet (CurveTabSet)\TabRecSet (CurveTabSet)\image\english_no-line\_1z6t7pa.jpg"
print(os.path.exists(path))

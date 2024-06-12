import torch
import torch.nn as nn

class Image_adapter(nn.Module):
    def __init__(self, input_size=1024, output_size=768):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size)
        )
        self.down_project = nn.Linear(input_size, output_size)  # 添加降维层
        self.mask = nn.Parameter(torch.zeros(input_size))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, feature):
        mask = self.sigmoid(self.mask)
        masked_feature = mask * feature
        adapted_feature = self.adapter(masked_feature)
        adapted_feature = self.down_project(adapted_feature)  # 降维到输出大小
        out_feature = adapted_feature + self.down_project(masked_feature)  # 确保匹配输出维度
        
        return out_feature   

# class Image_adapter(nn.Module):
#     def __init__(self, input_size=768, output_size=1024):
#         super().__init__()
#         self.adapter = nn.Sequential(
#             nn.Linear(input_size, output_size),
#             nn.ReLU(),
#             nn.Linear(output_size, output_size)
#         )
#         self.mask = nn.Parameter(torch.zeros(output_size))
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, feature):
#         # 确保 mask 和 feature 的形状匹配
#         mask = self.sigmoid(self.mask)
#         if feature.size(-1) != mask.size(0):
#             raise ValueError(f"Expected feature size (-1) to be {mask.size(0)}, but got {feature.size(-1)}")
        
#         masked_feature = mask * feature
#         adapted_feature = self.adapter(masked_feature)
#         out_feature = adapted_feature + masked_feature

#         return out_feature

def cal_cos(text, img, cos):
    a = text.mean(dim=1)
    b = img.squeeze(0)
    sim = cos(a, b).mean()
    return sim
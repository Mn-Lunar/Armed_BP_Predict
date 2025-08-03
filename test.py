import sys
sys.path.insert(0, "D:\\Projects\\BP-Mix-Effect-Solution\\Armed_BP_Predict")
import numpy as np
from mat73 import loadmat
import pprint

#data = loadmat('./dataset/PulseDB_MIMIC/p000160.mat')
data = loadmat('./dataset/PulseDB_Vital/p000001.mat')

print("Top-level keys:", data.keys())

# 探索 Subj_Wins 的结构
subj = data['Subj_Wins']
print("\nType of Subj_Wins:", type(subj))

# 如果是列表或数组，查看第一个样本的内容
if isinstance(subj, (list, tuple)):
    print("\nSubj_Wins[0] keys:")
    pprint.pprint(subj[0].keys())
elif isinstance(subj, dict):
    print("\nSubj_Wins keys:")
    pprint.pprint(subj.keys())
else:
    print("\nUnknown structure — printing full Subj_Wins object:")
    pprint.pprint(subj)

for key in subj:
    value = subj[key]
    shape = getattr(value, 'shape', 'N/A')
    dtype = type(value)
    # 打印字段信息
    print(f"{key:<15} | type: {dtype.__name__:<20} | shape: {shape}")

current_list = subj['ABP_F']     # 选择需要查看的字段
array_in_current_list = current_list[0]     # 取第一个元素（是个数组）

print("Length:", len(current_list))
print("values:", current_list)



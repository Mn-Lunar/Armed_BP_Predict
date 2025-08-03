import mat73
import pandas as pd
import os
import sys
sys.path.insert(0, "D:\\Projects\\BP-Mix-Effect-Solution\\Armed_BP_Predict")

# 加载 .mat 文件（v7.3）
mat_data = mat73.loadmat('./dataset/PulseDB_MIMIC/p000160.mat')

# 提取主数据字典
subj = mat_data['Subj_Wins']

# 要提取的字段（请按需增减）
fields = ['ABP_F', 'ABP_Lag', 'ABP_Raw', 'ABP_SPeaks', 'ABP_Turns',
          'Age', 'CaseID', 'ECG_F', 'ECG_RPeaks', 'ECG_Raw',
          'ECG_Record', 'ECG_Record_F', 'Gender', 'IncludeFlag',
          'PPG_ABP_Corr', 'PPG_F', 'PPG_Raw', 'PPG_Record',
          'PPG_Record_F', 'PPG_SPeaks', 'SegDBP', 'SegSBP',
          'SegmentID', 'SubjectID', 'T', 'WinID', 'WinSeqID']

# 构造记录
records = []
n = len(subj['PPG_Raw'])  # 用某一字段长度判断样本数

for i in range(n):
    if subj['IncludeFlag'][i][0] != 1:
        continue  # 跳过无效样本

    row = {}
    for field in fields:
        try:
            val = subj[field][i][0]
            # 如果是数组，压缩为字符串
            if hasattr(val, 'shape'):
                row[field] = ','.join(map(str, val))
            else:
                row[field] = val
        except Exception as e:
            row[field] = f"[ERR: {str(e)}]"

    records.append(row)

# 导出 Excel
df = pd.DataFrame(records)
os.makedirs('./output', exist_ok=True)
df.to_excel('./output/p000160.xlsx', index=False)
print("✅ 转换成功，样本数:", len(df))

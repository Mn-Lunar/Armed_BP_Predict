import sys
sys.path.insert(0, "D:\\Projects\\BP-Mix-Effect-Solution\\Armed_BP_Predict")

import os;
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
           
import tensorflow as tf
from buildDataSet import SignalDatasetProcessing
def test_loading(mat_path, use_demographics=False, signal_channels=['PPG']):
    

    # 初始化数据处理类
    dataset_builder = SignalDatasetProcessing(
        mat_path=mat_path,
        use_demographics=use_demographics,
        signal_channels=signal_channels,
        batch_size=4,  # 小批量，方便调试查看
        shuffle=False
    )

    # 获取 tf.data.Dataset
    tf_dataset = dataset_builder.get_dataset()

    # 取第一个 batch 进行展示
    for batch in tf_dataset.take(1):
        if use_demographics:
            (signals, demographics), labels = batch
            print("Signals shape:", signals.shape)          # e.g., (4, 1250, 1)
            print("Demographics shape:", demographics.shape)  # e.g., (4, 4)
        else:
            signals, labels = batch
            print("Signals shape:", signals.shape)          # e.g., (4, 1250, 1)

        print("Labels shape:", labels.shape)
        print("First label values:", labels.numpy())
        break

if __name__ == "__main__":
    test_loading(
        mat_path='H:\\Project\\PulseDB\\Supplementary_Subset_Files\\VitalDB_Train_Subset.mat',  
        
        use_demographics=True,
        signal_channels=["PPG"]  
    )
#You need to install the package mat73 because PulseDB uses MAT file version 7.3 to store large volume data
from mat73 import loadmat 
import numpy as np
import tensorflow as tf


class SignalDatasetProcessing:

    SIGNAL_DICT = {'ECG': 0, 'PPG': 1, 'ABP': 2}

    def __init__(self, mat_path, label_name='SBP', field_name='Subset',
                    use_demographics=False, signal_channels=['PPG'],
                    batch_size=32, shuffle=True, buffer_size=1000):
            self.mat_path = mat_path
            self.label_name = label_name
            self.field_name = field_name
            self.use_demographics = use_demographics # Random Effects
            self.signal_channels = signal_channels
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.buffer_size = buffer_size
            

            # Load and preprocess data
            self._load_mat()
            self.dataset = self._to_tf_dataset()
            print("Finished data database building.")

    def _load_mat(self):
        print("Loading .mat file...")
        Data = loadmat(self.mat_path)
        print(" .mat file loaded.")
        
        Subset = Data[self.field_name] # Data[FieldName]
    
        # Select specific signal channels (by name)
        full_signals = Subset['Signals']
        print(f"Raw signals shape: {full_signals.shape}")
        # PulseDB shape: (N, C=3, T)
        # N: Segment数目
        # C: 通道数，signal包括ECG PPG ABP
        # T: 时间戳，即记录signals时的时间点
        # segment_0:
        #    ECG: [1250 values]
        #    PPG: [1250 values]
        #    ABP: [1250 values]
        
        # 选择需要的信号通道
        channel_selected = [self.SIGNAL_DICT[c] for c in self.signal_channels]
        self.Signals = full_signals[:, channel_selected, :]  # (N, selected_C, T)
        print(f" Selected signals shape: {self.Signals.shape}")

        # 转换为适用于Keras的数据结构
        # transpose to (N, T, selected_C) for Keras
        self.Signals = np.transpose(self.Signals, (0, 2, 1)).astype(np.float32)

        # Extract labels, defalt to SBPs
        Labels = Subset[self.label_name]
        self.Labels = np.array(Labels).squeeze().astype(np.float32)
        print(f"Raw Labels shape: {self.Labels.shape}")

        # 限定前 30000 条样本
        num_training_samples = 50000 #self.Signals.shape[0] #
        self.Signals = self.Signals[:num_training_samples]
        self.Labels = self.Labels[:num_training_samples]
        print(f"Truncated to first {num_training_samples} samples.")

    
        # TODO： 写一个方法去添加指定的信息
        # 选择性添加 random effect 的信息
        # Optionally add demographic features
        if self.use_demographics:
            print("Loading demographics...")
            Age = Subset['Age'][:num_training_samples]
            Gender = np.array(Subset['Gender']).squeeze()[:num_training_samples]
            Gender = (Gender == 'M').astype(float)
            Height = Subset['Height'][:num_training_samples]
            Weight = Subset['Weight'][:num_training_samples]
            Demographics = np.stack((Age, Gender, Height, Weight), axis=1).astype(np.float32)
            self.Inputs = (self.Signals, Demographics)
            print(f"Demographics shape: {Demographics.shape}")
        else:
            self.Inputs = self.Signals
            print("Finished data preprocessing.")


        
    
    def _to_tf_dataset(self):
        print("Change data to tf format...")
        self.Labels = tf.reshape(self.Labels, [-1, 1])
        print(f"Processed Labels shape: {self.Labels.shape}")

        if self.use_demographics:
            ds = tf.data.Dataset.from_tensor_slices(((self.Inputs[0], self.Inputs[1]), self.Labels))
        else:
            ds = tf.data.Dataset.from_tensor_slices((self.Inputs, self.Labels))

        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.buffer_size)

        return ds.batch(self.batch_size).prefetch(1)

    def get_dataset(self):
        return self.dataset





""" def Build_Dataset(Path,FieldName='Subset'):
        Data=loadmat(Path)
        # Access 10-s segments of ECG, PPG and ABP signals
        Signals=Data[FieldName]['Signals']
        # Access SBP labels of each 10-s segment
        SBPLabels=Data[FieldName]['SBP']
        # Access Age of the subject corresponding to each of the 10-s segment
        Age=Data[FieldName]['Age']
        # Access Gender of the subject corresponding to each of the 10-s segment
        Gender=np.array(Data[FieldName]['Gender']).squeeze()
        # Convert Gender to numerical 0-1 labels
        Gender=(Gender=='M').astype(float)
        # Access Height and Weight of the subject corresponding to each of the 10-s segment
        # If the subject is from the MIMIC-III matched subset, height and weight will be NaN 
        # since they were only recorded in VitalDB
        Height=Data[FieldName]['Height']
        Weight=Data[FieldName]['Weight']
        # Concatenate the demographic information as one matrix
        Demographics=np.stack((Age,Gender,Height,Weight),axis=1)
        return Signals,SBPLabels,Demographics
        
Build_Dataset('PulseDB\\Subset_Files\\Train_Subset.mat') """
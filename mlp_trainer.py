import os
import tensorflow as tf
from buildDataSet import SignalDatasetProcessing
from Armed_Predictor.mlp_classifiers import BaseMLP, DomainAdversarialMLP, Adversary
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

class ModelRunner:
    def __init__(self,
                 mat_path,
                 test_mat_path=None,
                 use_demographics=True,
                 signal_channels=["PPG"],
                 batch_size=32,
                 learning_rate=0.001,
                 epochs=5):
        
        self.mat_path = mat_path
        self.test_mat_path = test_mat_path
        self.use_demographics = use_demographics
        self.signal_channels = signal_channels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self._build_dataset()
        self._build_model()

    def _build_dataset(self):
        # 构造数据集
        self.dataset_builder = SignalDatasetProcessing(
            mat_path=self.mat_path,
            use_demographics=self.use_demographics,
            signal_channels=self.signal_channels,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.dataset = self.dataset_builder.get_dataset()

        if self.test_mat_path:
            self.test_dataset_builder = SignalDatasetProcessing(
                mat_path=self.test_mat_path,
                use_demographics=self.use_demographics,
                signal_channels=self.signal_channels,
                batch_size=self.batch_size,
                shuffle=False
            )
            self.test_dataset = self.test_dataset_builder.get_dataset()
        else:
            self.test_dataset = None
    
    
    def _build_model(self, modelname = "BaseMLP"):
        if modelname == "BaseMLP":
            self.model = BaseMLP()
            self.loss_fn = tf.keras.losses.MeanSquaredError()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        elif modelname == "AdvMLP":
            self.model = DomainAdversarialMLP()
            self.model.compile()
            
 


    
    def _process_input(self, batch):
        """统一处理输入，包括是否使用 demographics 的情况"""
        if self.use_demographics:
            (signals, demographics), labels = batch
            signals_flat = tf.reshape(signals, [signals.shape[0], -1])
            input_vector = tf.concat([signals_flat, demographics], axis=1)
            
            # ([120, 132, 118, 125]) -->  ([[120], [130], [119], [124]])
            labels = tf.reshape(labels, [-1, 1])

        else:
            signals, labels = batch
            signals_flat = tf.reshape(signals, [signals.shape[0], -1])
            input_vector = signals_flat
            labels = tf.reshape(labels, [-1, 1])

        return input_vector, labels
    
    def _process_input_with_demographics(self, batch):
        """统一处理输入，包括是否使用 demographics 的情况"""
        if self.use_demographics:
            (signals, demographics), labels = batch
            signals_flat = tf.reshape(signals, [signals.shape[0], -1])
            input_vector = tf.concat([signals_flat, demographics], axis=1)
            labels = tf.reshape(labels, [-1, 1])

        else:
            signals, labels = batch
            signals_flat = tf.reshape(signals, [signals.shape[0], -1])
            input_vector = signals_flat
            labels = tf.reshape(labels, [-1, 1])

        return input_vector, demographics, labels

    def extract_all_demographics(self):
        all_demo = []

        for batch in self.dataset:
            (_, demographics), _ = batch
            all_demo.append(demographics.numpy())
        
        return np.concatenate(all_demo, axis=0)


    def kmean_demographics(self, n_clusters=20):
        """对 demographics 进行 KMeans 聚类，并将簇标签转换为 One-Hot 编码"""

        # 1. 提取所有样本的 demographics 特征矩阵
        all_demo = self.extract_all_demographics()  # shape: (N, D)，例如 (4000, 4)

        # 2. 用 KMeans 进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(all_demo)  # shape: (N,)，每个样本的簇编号

        # 3. 将簇标签转为 One-Hot 编码
        encoder = OneHotEncoder(sparse=False)
        cluster_onehot = encoder.fit_transform(cluster_labels.reshape(-1, 1))  # shape: (N, n_clusters)

        # 4. 可选：打印信息
        print(f"KMeans 聚类完成，聚类中心 shape: {kmeans.cluster_centers_.shape}")
        print(f"One-Hot 编码后的 shape: {cluster_onehot.shape}")

        # 5. 返回 one-hot 和原始 label
        return cluster_onehot, cluster_labels, kmeans
    
    def build_cluster_augmented_dataset(self, cluster_onehot):
        """
        将 one-hot 簇标签拼接回 signal + demographics 输入中，构建新 dataset。
        参数:
            cluster_onehot: shape (N, n_clusters)，由 kmean_demographics 得到
        返回:
            tf.data.Dataset，其中每个元素为 (input_vector, label)
        """

        all_signals = []
        all_demo = []
        all_labels = []

        for batch in self.dataset:
            (signals, demographics), labels = batch
            all_signals.append(signals.numpy())       # shape: (B, T, C)
            all_demo.append(demographics.numpy())     # shape: (B, D)
            all_labels.append(labels.numpy())         # shape: (B, 1)


        # 合并所有 batch 数据
        all_signals = np.concatenate(all_signals, axis=0)  # shape: (N, T, C)
        all_demo = np.concatenate(all_demo, axis=0)        # shape: (N, D)
        all_labels = np.concatenate(all_labels, axis=0)    # shape: (N, 1)

        # Flatten signals
        signals_flat = all_signals.reshape(all_signals.shape[0], -1)  # shape: (N, T*C)

        # 拼接： [signals_flat | demographics | cluster_onehot]
        input_vector = np.concatenate([signals_flat, all_demo, cluster_onehot], axis=1)  # shape: (N, D_total)

        # 构建新的 tf.data.Dataset
        new_dataset = tf.data.Dataset.from_tensor_slices((input_vector.astype(np.float32), all_labels.astype(np.float32)))
        new_dataset = new_dataset.batch(self.batch_size).shuffle(buffer_size=1000)

        print("拼接完成，新数据集构建成功。")
        print(f"→ input_vector shape: {input_vector.shape}")
        print(f"→ labels shape: {all_labels.shape}")

        return new_dataset
        


    # train base mlp
    def train(self):
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            for batch in tqdm(self.dataset, desc=f"Training Epoch {epoch+1}"):
                
                input_vector, labels = self._process_input(batch)
                
                with tf.GradientTape() as tape:
                    preds = self.model(input_vector, training=True)
                    preds = tf.reshape(preds, [-1, 1])
                    #计算损失函数
                    loss = self.loss_fn(labels, preds)

                # 更新损失函数的梯度
                grads = tape.gradient(loss, self.model.trainable_variables)# 对损失函数求可训练参数的梯度
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))# 优化参数

                # 进行下一组计算
                epoch_loss += loss.numpy()
                batches += 1
            
            if (epoch + 1) % 5 == 0 and self.test_dataset is not None:
                print(f"→ Evaluating on test set after Epoch {epoch+1}")
                self.evaluate_testset()

            avg_loss = epoch_loss / batches
            print(f"→ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")


    

    def trainADVMLP(self, epochs=10, verbose=1):
        """
        使用标准 Keras 的 model.fit() 接口进行训练
        """
        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            verbose=verbose
        )

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset)

    def predict(self, test_dataset):
        return self.model.predict(test_dataset)




    def evaluate_one_batch(self):
        for batch in self.dataset.take(1):
            input_vector, labels = self._process_input(batch)
            preds = self.model(input_vector, training=False)
            print("Predictions:", preds.numpy().flatten())
            print("True Labels:", labels.numpy().flatten())
            break
    
    def evaluate_testset(self):
        if self.test_dataset is None:
            print("No test dataset provided.")
            return

        all_preds, all_labels = [], []
        for batch in self.test_dataset:
            input_vector, labels = self._process_input(batch)
            preds = self.model(input_vector, training=False)
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())

        all_preds = tf.concat(all_preds, axis=0).numpy().flatten()
        all_labels = tf.concat(all_labels, axis=0).numpy().flatten()

        mse = tf.keras.losses.MeanSquaredError()(all_labels, all_preds).numpy()
        mae = tf.keras.losses.MeanAbsoluteError()(all_labels, all_preds).numpy()
        print(f"→ Test MSE: {mse:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    runner = ModelRunner(
        mat_path='D:\\Projects\\BP-Mix-Effect-Solution\\Armed_BP_Predict\\dataset\\PulseDB_Vital\\VitalDB_Train_Subset.mat',
        test_mat_path='D:\\Projects\\BP-Mix-Effect-Solution\\Armed_BP_Predict\\dataset\\PulseDB_Vital\\VitalDB_CalFree_Test_Subset.mat',
        
        use_demographics=False,
        signal_channels=["PPG"],
        batch_size=8,
        learning_rate=0.001,
        epochs=20
    )
    runner.train()
    runner.evaluate_testset()

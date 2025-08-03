import os
import tensorflow as tf
from buildDataSet import SignalDatasetProcessing
from Armed_Predictor.mlp_classifiers import BaseMLP  # 确保你的 BaseMLP 类保存在 model_mlp.py 中
from tqdm import tqdm

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
        

    def _build_model(self):
        self.model = BaseMLP()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _process_input(self, signals, demographics):
        signals_flat = tf.reshape(signals, [signals.shape[0], -1])
        return tf.concat([signals_flat, demographics], axis=1)

    def train(self):
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            for batch in tqdm(self.dataset, desc=f"Training Epoch {epoch+1}"):
                if self.use_demographics:
                    (signals, demographics), labels = batch
                    input_vector = self._process_input(signals, demographics)
                else:
                    signals, labels = batch
                    input_vector = tf.reshape(signals, [signals.shape[0], -1])

                with tf.GradientTape() as tape:
                    preds = self.model(input_vector, training=True)
                    labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.float32)
                    preds = tf.reshape(preds, [-1, 1])
                    loss = self.loss_fn(labels, preds)

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                epoch_loss += loss.numpy()
                batches += 1

            avg_loss = epoch_loss / batches
            print(f"→ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")


    def evaluate_one_batch(self):
        for batch in self.dataset.take(1):
            if self.use_demographics:
                (signals, demographics), labels = batch
                input_vector = self._process_input(signals, demographics)
            else:
                signals, labels = batch
                input_vector = tf.reshape(signals, [signals.shape[0], -1])

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
            if self.use_demographics:
                (signals, demographics), labels = batch
                input_vector = self._process_input(signals, demographics)
            else:
                signals, labels = batch
                input_vector = tf.reshape(signals, [signals.shape[0], -1])

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
        epochs=100
    )
    runner.train()
    runner.evaluate_testset()

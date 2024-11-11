import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

# 設置路徑
root_path = os.path.abspath(os.path.dirname(__file__))

# 超參數設置
learning_rate = 0.001
epochs = 50
batch_size = 32

def load_training_data():
    # 加載訓練數據
    train_dataset = np.load(os.path.join(root_path, 'dataset', 'train.npz'))
    train_data = train_dataset['data']
    train_label = to_categorical(train_dataset['label'])  # 使用 one-hot 編碼
    return train_data, train_label

def load_validation_data():
    # 加載驗證數據
    valid_dataset = np.load(os.path.join(root_path, 'dataset', 'validation.npz'))  # 修改文件名
    valid_data = valid_dataset['data']
    valid_label = to_categorical(valid_dataset['label'])  # 使用 one-hot 編碼
    return valid_data, valid_label

def train_model():
    # TensorBoard 日誌設置
    log_dir = "Logs/YOURMODEL"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    # 定義模型
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')  # 假設有 5 個分類類別
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # 加載訓練數據
    train_data, train_label = load_training_data()

    # 訓練模型
    model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])

    # 保存模型
    model.save('YOURMODEL.h5')

def evaluate_model():
    # 加載訓練好的模型
    model = tf.keras.models.load_model('YOURMODEL.h5')

    # 加載驗證數據
    valid_data, valid_label = load_validation_data()

    # 進行預測
    predictions = model.predict(valid_data, batch_size=batch_size)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(valid_label, axis=1)

    # 計算準確率
    accuracy = np.mean(true_labels == predicted_labels)
    print(f'Predicted labels: {predicted_labels}')
    print(f'True labels: {true_labels}')
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    # 訓練模型
    train_model()

    # 評估模型
    evaluate_model()




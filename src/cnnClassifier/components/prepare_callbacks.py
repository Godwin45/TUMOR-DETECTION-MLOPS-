import os
import tensorflow as tf
import urllib.request as request
from pathlib import Path
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config):
        self.config = config


    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        monitor = str(Path('val_loss'))
        checkpoint_filepath = str(self.config.checkpoint_model_filepath)  # Convert WindowsPath object to string

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='min', verbose=1, patience=20)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True)

        return early_stopping, checkpointer

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,  # Invoke the _create_tb_callbacks method
            self._create_ckpt_callbacks  # Invoke the _create_ckpt_callbacks method
        ]

import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    layers = None


    
    def get_base_model(self):
        self.model = tf.keras.applications.resnet50.ResNet50(
            input_tensor= tf.keras.layers.Input(shape=(256, 256, 3)),
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, learning_rate):
        for layer in model.layers:
            layer.trainable = False

        # Add classification head to the base model
        headmodel = model.output
        headmodel = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(headmodel)
        headmodel = tf.keras.layers.Flatten(name='flatten')(headmodel)
        headmodel = tf.keras.layers.Dense(256, activation="relu")(headmodel)
        headmodel = tf.keras.layers.Dropout(0.3)(headmodel)
        headmodel = tf.keras.layers.Dense(256, activation="relu")(headmodel)
        headmodel = tf.keras.layers.Dropout(0.3)(headmodel)
        #headmodel = tf.keras.layers.Dense(256, activation="relu")(headmodel)
        #headmodel = tf.keras.layers.Dropout(0.3)(headmodel)
        headmodel = tf.keras.layers.Dense(units = classes, activation='softmax')(headmodel)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=headmodel)


        full_model.compile(
            loss = 'categorical_crossentropy',
              optimizer='adam', 
              metrics= ["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    

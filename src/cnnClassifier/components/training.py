from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path
import os

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
    
    def train_valid_generator(self):
        self.config.training_data['mask'] = self.config.training_data['mask'].apply(lambda x: str(x))
        
        # Specify the full path to the 'Brain_MRI' directory
        directory = os.path.join('artifacts/data_ingestion/Brain_MRI')
        
        datagenerator_kwargs = dict(
            rescale=1./255.,
            validation_split=0.15
        )

        dataflow_kwargs = dict(
            x_col='image_path',
            y_col='mask',
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            target_size=(256, 256)
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_dataframe(
            dataframe=self.config.training_data,
            directory=directory,
            subset="validation",
            shuffle=True,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_dataframe(
            dataframe=self.config.training_data,
            directory=directory,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.n // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.n // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

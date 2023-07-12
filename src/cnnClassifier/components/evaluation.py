import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
import os

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.score = None

    
    def _valid_generator(self):

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

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.feature_engineering import FeatureEngineering
from cnnClassifier import logger


STAGE_NAME = "Feature Engineering stage"

class FeatureEngineeringPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_engineering_config = config.get_feature_engineering_config()
        feature_engineering = FeatureEngineering(config=feature_engineering_config)
        feature_engineering.preprocess_data()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureEngineeringPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


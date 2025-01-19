from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        # Data Ingestion
        ingestion = DataIngestion()
        train_data, test_data = ingestion.initiate_data_ingestion()

        # Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data, test_data)

        # Model Training
        trainer = ModelTrainer()
        r2_score = trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"Model Training Completed. R2 Score: {r2_score:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

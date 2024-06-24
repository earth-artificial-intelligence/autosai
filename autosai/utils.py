from tensorflow.keras.models import save_model as keras_save_model

class ModelSaver:
    @staticmethod
    def save_model(model, model_type, filename):
        if model_type == 'tabnet':
            model.save_model(filename)
        else:
            keras_save_model(model, filename)

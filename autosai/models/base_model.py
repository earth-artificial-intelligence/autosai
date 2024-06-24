class BaseModel:
    def __init__(self, trial, input_shape):
        self.trial = trial
        self.input_shape = input_shape

    def build_model(self):
        raise NotImplementedError("Subclasses should implement this method.")


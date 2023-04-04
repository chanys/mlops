import os

from fastapi import FastAPI
from onnx_inference import ONNXPredictor
from hydra import initialize, compose
from omegaconf import OmegaConf


app = FastAPI(title="MLOps Basics App")

# @app.get("/")
# async def home():
#     return "<h2>This is a sample NLP Project</h2>"




# load the model
# configuration
initialize("../expts/configs")  # this must be a relative directory
cfg = compose(config_name="config.yaml")
config = OmegaConf.to_object(cfg)
config['processing']['mode'] = 'inference'
config['data']['labels'] = ['unacceptable', 'acceptable']
config['processing']['model_path'] = './expts/model_output'
predictor = ONNXPredictor(config)

@app.get("/predict")
async def get_prediction(text: str):
    result = predictor.predict(text)
    return result



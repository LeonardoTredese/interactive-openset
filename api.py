from flask import Flask, request
import torch as to 
import numpy as np
from model.MLP import MLP

app = Flask(__name__)
device = to.device("cpu")
models = []


@app.before_first_request
def load_saves():
    app.logger.info("Loading saved models...")
    saves = to.load('./saves')
    for i in range(1,10):
        model_name = "model_%d_" % i 
        args = saves[model_name + "args"]
        kwargs = saves[model_name + "kwargs"]
        model = MLP(*args,**kwargs).to(device)
        model.load_state_dict(saves[model_name + "state_dict"])
        models.append(model)
    app.logger.info("Finished loading models")   

@app.route("/evaluate/<int:model_id>", methods=['post'])
def evaluate_input(model_id):
    digit = request.json['digit']
    if not ( 0 < model_id < len(models)):
        return "specified model not found", 404
    elif not (len(digit) == models[model_id].in_dim):
        print(len(digit))
        return "digit has to be of length %d" % models[model_id].in_dim, 400
    digit = to.FloatTensor(digit).to(device)
    logits = models[model_id](digit)
    _, prediction = to.max(logits, -1)
    prediction = prediction.cpu().numpy().tolist()
    logits = logits.detach().numpy().tolist()
    return { "digit": prediction , "scores": logits  }, 200

    

from flask import Flask, request
from flask_cors import CORS
import torch as to 
import torch.nn.functional as F
import numpy as np
from model.MLP import MLP

app = Flask(__name__)
CORS(app)
device = to.device("cpu")
models = {}


@app.before_first_request
def load_saves():
    app.logger.info("Loading saved models...")
    saves = to.load('./saves')
    for i in range(2,11):
        model_name = "model_%d_" % i 
        args = saves[model_name + "args"]
        kwargs = saves[model_name + "kwargs"]
        model = MLP(*args,**kwargs).to(device)
        model.load_state_dict(saves[model_name + "state_dict"])
        model.eval()
        models[model_name] = model
    app.logger.info("Finished loading models")   

@app.route("/evaluate/<int:model_id>/<float:unknown_margin>", methods=['post'])
def evaluate_input(model_id, unknown_margin = 0):
    digit = request.json['digit']
    model_name = "model_%d_" % model_id
    if not (model_name in models):
        return "specified model not found", 404
    elif not (len(digit) == models[model_name].in_dim):
        return "digit has to be of length %d" % models[model_name].in_dim, 400
    digit = to.FloatTensor(digit).to(device)
    logits = F.softmax(models[model_name](digit), -1)
    values, prediction = to.max(logits, -1)
    if values >= unknown_margin:
       prediction = prediction.cpu().numpy().tolist()
    else:
       prediction = "unknown"
    logits = logits.detach().numpy().tolist()
    return { "digit": prediction , "scores": logits  }, 200

    

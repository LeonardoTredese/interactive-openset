from flask import Flask, request
from flask_cors import CORS
import torch as to 
import torch.nn.functional as F
import numpy as np
from model.NN import NN

app = Flask(__name__)
CORS(app)
device = to.device("cuda:0" if to.cuda.is_available() else "cpu")
models = {}


@app.before_first_request
def load_saves():
    app.logger.info("Loading saved models...")
    saves = to.load('./saves', map_location=device)
    for i in range(2,11):
        model_name = "model_%d_" % i 
        args = saves[model_name + "args"]
        kwargs = saves[model_name + "kwargs"]
        model = NN(*args,**kwargs).to(device)
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
    elif not (len(digit) == 784):
        return "digit has to be of length %d" % models[model_name].in_dim, 400
    digit = to.FloatTensor(digit).to(device)
    logits = F.softmax(models[model_name](digit.view(-1,1,28, 28)), -1)
    value, prediction = to.max(logits, -1)
    if value >= unknown_margin:
       prediction = prediction.cpu().numpy().tolist()
    else:
       prediction = "unknown"
    logits = logits.detach().numpy().tolist()[0]
    return { "digit": prediction , "scores": logits  }, 200

    

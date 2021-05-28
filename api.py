from flask import Flask, request
from flask_cors import CORS
import torch as to 
import torchvision as tv
import torch.nn.functional as F
import numpy as np
from model.NN import NN
from pytorch_grad_cam import GradCAM
from itertools import islice

app = Flask(__name__)
app.debug = True
CORS(app)
device = to.device("cuda:0" if to.cuda.is_available() else "cpu")
models = {}
cams = {}



@app.before_first_request
def load_saves():
    app.logger.info("Loading saved models...")
    saves = to.load('./saves', map_location=device)
    load_models(saves)
    app.logger.info("Finished loading models")

@app.route("/evaluate/<int:model_id>/<int:prediction>", methods=['post'])
@app.route("/evaluate/<int:model_id>", methods=['post'])
def evaluate_input(model_id, prediction = -1 ):
    digit = request.json['digit']
    model_name = "model_%d_" % model_id
    if not (model_name in models):
        return "specified model not found", 404
    elif not (len(digit) == 784):
        return "digit has to be of length %d" % models[model_name].in_dim, 400
    digit = to.FloatTensor(digit).to(device).view(-1,1,28, 28)
    logits = F.softmax(models[model_name](digit), -1)
    prediction = to.max(logits, -1)[1] if prediction < 0 else to.LongTensor([prediction])
    gray_scale = cams[model_name](input_tensor=digit, target_category=prediction)
    return {
        "prob": logits[0].detach().numpy().tolist(),
        "grayScale": gray_scale[0].tolist(),
        "pred": prediction[0].detach().numpy().tolist()
        }, 200

@app.route("/examples")
def extract_samples():
    return { "examples": load_examples() }, 200

def load_models(saves):
    for i in range(2,11):
        model_name = "model_%d_" % i 
        args = saves[model_name + "args"]
        kwargs = saves[model_name + "kwargs"]
        model = NN(*args,**kwargs).to(device)
        model.load_state_dict(saves[model_name + "state_dict"])
        model.eval()
        models[model_name] = model
        cams[model_name] = GradCAM(model=model, target_layer=model.conv1, use_cuda=to.cuda.is_available())

def load_examples():
    limit=25
    ds = tv.datasets.MNIST(root='./', train=False, transform=tv.transforms.ToTensor(), download=True)
    dl = to.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    return [{
           "x": x[0].detach().numpy().tolist(),
           "y": y[0].detach().numpy().tolist()
          } for x, y in islice(dl, limit)]
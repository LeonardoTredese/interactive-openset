let mouse = {x: -1, y: 0};
let last_mouse = {x: -1, y: 0};

 
window.onload = function() {
    var canvas = document.getElementById("input");
    var context = canvas.getContext("2d");
    context.lineWidth = 29;
    context.lineJoin = 'round';
    context.lineCap = 'round';
    context.strokeStyle = 'black'
    context.rect(0, 0, 280, 280);
    context.fillStyle = 'white';
    context.fill();
    
    canvas.addEventListener('mousedown', function() {
        canvas.addEventListener('mousemove', onPaint, false);
    }, false);
    
    canvas.addEventListener('mouseup', function() {
        canvas.removeEventListener('mousemove', onPaint, false);
        uploadEx()
    }, false);

    canvas.addEventListener('mousemove', function(e) {
        last_mouse.x = mouse.x;
        last_mouse.y = mouse.y;
        if (e.offsetX) {
            mouse.x = e.offsetX;
            mouse.y = e.offsetY;
        }
        else if (e.layerX) {
            mouse.x = e.layerX;
            mouse.y = e.layerY;
        }
    }, false);
    
    document.getElementById('grad-cam-attention').onchange = uploadEx
    getExamples()
}

var onPaint = function() {
    var canvas = document.getElementById("input");
    var context = canvas.getContext("2d");
    context.beginPath();
    context.moveTo(last_mouse.x, last_mouse.y);
    context.lineTo(mouse.x, mouse.y);
    context.closePath();
    context.stroke();
};


 
function createExampleCanvas(examples){
    for(let i = 0; i < examples.length; i++){
         let canv = document.createElement('canvas')
         let div = createNumberDiv(examples[i].y, `example-${i}-val`)
         canv.id = `example-${i}`
         canv.width = 28;
         canv.height = 28;
         canv.className = "small-canvas"
         canv.onclick = () => {
             clearCanvas('input')
             drawMatrix('input', examples[i].x[0],280,10,[0,0,0])
             uploadEx()
         }
         div.appendChild(canv)
         document.getElementById('examples').appendChild(div)
         drawMatrix(canv.id, examples[i].x[0],28,1,[0,0,0])
    }
 }
function updateExampleCanvas(examples){
    for(let i = 0; i < examples.length; i++){
        let id = `example-${i}`
        document.getElementById(id + '-val').innerHTML = examples[i].y
        let canv = document.getElementById(id)
        canv.onclick = () => {
            clearCanvas('input')
            drawMatrix('input', examples[i].x[0],280,10,[0,0,0])
            uploadEx()
        }
        clearCanvas(id)
        drawMatrix(id, examples[i].x[0],28,1,[0,0,0])
    }
 }

 function getExamples(){
     var xhr = new XMLHttpRequest();
     var url = "http://127.0.0.1:5000/examples"
     xhr.open("GET", url, true);
     xhr.setRequestHeader("Access-Control-Allow-Origin", "*");
     xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var json = JSON.parse(xhr.responseText);
            if(document.getElementById('example-0'))
                updateExampleCanvas(json.examples)
            else
                createExampleCanvas(json.examples)
        }
    };
    xhr.send();
 }

function clearCanvas(id) {
     var canvas = document.getElementById(id);
     var context = canvas.getContext("2d");
     context.clearRect(0, 0, canvas.width, canvas.height);
     context.rect(0, 0, 280, 280);
     context.fillStyle = 'white';
     context.fill();
 }

function createNumberDiv(number, id){
    let div = document.createElement('div')
    let strong = document.createElement('strong')
    strong.innerHTML = number
    strong.id = id
    div.className = "vertical-container"
    div.appendChild(strong)
    return div
}

function drawPredictionGraphs(prob){
    var digits =  [...Array(prob.length).keys()];
    var probabilities = [{
        x: digits,
        y: prob,
        type: 'bar',
        name: 'digits'
    },];
    var entropyData = [{
        y: [informationEntropy(prob)],
        type: 'bar',
        marker: {
            color: "orange"
        },
        name: 'Entropy of results'
    }, {
        x:[-0.5, 0.5],
        y:Array(2).fill(document.getElementById('margin').value),
        type: 'line',
        marker: {
            color: "red"
        },
        name: 'Entropy margin'
    }]
    Plotly.newPlot('probabilities', probabilities);
    Plotly.newPlot('entropy', entropyData);
}

function upload(digit){
    var xhr = new XMLHttpRequest();
    var model = document.getElementById('model').value
    var gradClass = document.getElementById('grad-cam-attention').value  
    var url = "http://127.0.0.1:5000/evaluate/" + model + (gradClass >= 0 ? '/' + gradClass : '');
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var json = JSON.parse(xhr.responseText);
            const prediction = json.prob.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0)
            const entropyMargin = document.getElementById('margin').value
            document.getElementById("digit").innerText = informationEntropy(json.prob) < entropyMargin ? prediction : "unknown";
            drawPredictionGraphs(json.prob)
            drawMatrix('grad-cam', json.grayScale, 28, 10, [255, 0, 0])
        }
    };
    var data = JSON.stringify({"digit": digit});
    xhr.send(data);
}

function uploadEx() {
    var canvas = document.getElementById("input");
    var context = canvas.getContext("2d");
    var originalMatrix = context.getImageData(0,0,canvas.width, canvas.height).data;
    var reducedChannels = reduceImageChannels(4, originalMatrix);
    originalMatrix = listToMatrix(reducedChannels, 28*10)
    var output = [];
    for(i = 0; i <  28; i++){
        for(j = 0; j < 28 ; j++){
            output.push(tileAverage(i*10, j*10, 10, originalMatrix))
        }
    }
    clearCanvas('grad-cam')
    drawMatrix('grad-cam', listToMatrix(output,28), 28, 10, [0, 0, 0])
    upload(output)
};

function reduceImageChannels(nChannels, data){
    var output = [];
    var acc = 0;
    for(var i = 0; i < data.length; i++){
        if ((i % nChannels) == (nChannels -1)){
           output.push((acc + data[i]) / nChannels);
           acc = 0;
        } else {
           acc += data[i]
        }
    }
    return output
}

function listToMatrix(list, elementsPerSubArray) {
    var matrix = [], i, k;
    for (i = 0, k = -1; i < list.length; i++) {
        if (i % elementsPerSubArray === 0) {
            k++;
            matrix[k] = [];
        }
        matrix[k].push(list[i]);
    }
    return matrix;
}

function tileAverage(offsetx, offsety, dimension, data){
    var avg = 0, i, j;
    for(i = offsetx; i < dimension + offsetx; i++){
        for(j = offsety; j < dimension + offsety; j++){
             avg += data[i][j]
        }
    }
    avg = avg / (dimension * dimension);
    return (255 - avg)/255
}

function drawMatrix(id, matrix, dimension, grain, color){
    var canv = document.getElementById(id)
    var cont = canv.getContext("2d")
    for(j = 0; j < matrix.length ; j++){
        for(i = 0; i < matrix.length; i++){
            cont.fillStyle = 'rgba('+color[0]+', '+color[1]+', '+color[2]+', '+ matrix[j][i] +')';
            cont.fillRect(i * grain, j * grain, grain, grain);
        }
    }
}

function informationEntropy(probabilities) {
    return -probabilities.reduce((acc, prob) => acc + prob * Math.log2(prob));
}
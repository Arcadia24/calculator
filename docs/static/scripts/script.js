// Canvas drawing code adapted from Hugo Daumain 
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let isDrawing = false;
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
let predictionTimeout;

// Labels and storage for the calculated values
const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'mul', 'sub'];
let resultofpred = [];

// Add event listeners to the clear and extract buttons
var bouton = document.getElementById("ExtractButton");
bouton.addEventListener("click", MakeOperation);
var bouton2 = document.getElementById("ClearButton");
bouton2.addEventListener("click", () => {resultofpred.pop(); displayArray(resultofpred); clear();});

// Load the model
window.onload = async function() {
  await loadModel();
}

async function loadModel() {
  let loadingIndicator = document.getElementById('loadingIndicator');
  loadingIndicator.innerText = "Loading model...";
  model = new onnx.InferenceSession();
  await model.loadModel('./static/model/cnn_custom.onnx');
  loadingIndicator.innerText = "Model loaded. Start drawing!";  
}

/*
  * The prediction is delayed by 2 seconds after the user stops drawing.
  * This is to avoid making too many predictions while the user is drawing and allow to write the symbole division
  * If the user starts drawing again before the 2 seconds are up, the timer is reset.
*/


function startPredictionTimer() {
  // Clear any existing timers to ensure only one timer runs at a time
  clearTimeout(predictionTimeout);

  // Set the new timer
  predictionTimeout = setTimeout(async function() {
      await eval();clear();
  }, 2000);
}

// for the mouse
canvas.addEventListener('mousedown', (e) => { 
  startPredictionTimer()
  isDrawing = true;
  ctx.beginPath();
  ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
});


canvas.addEventListener('mouseup', () => { 
  startPredictionTimer()
  isDrawing = false; 

});

canvas.addEventListener('mousemove', draw);
// for phone users 

canvas.addEventListener('touchstart', function(e) {
  startPredictionTimer()
  e.preventDefault();
  isDrawing = true;
  let touch = e.touches[0];
  ctx.beginPath();
  ctx.moveTo(touch.clientX - canvas.offsetLeft, touch.clientY - canvas.offsetTop);
});

canvas.addEventListener('touchend', function(e) {
  startPredictionTimer()
  e.preventDefault();
  isDrawing = false;
});

canvas.addEventListener('touchmove', function(e) {
  e.preventDefault();
  if (!isDrawing) return;
  let touch = e.touches[0];
  ctx.lineWidth = 10;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'white';
  ctx.lineTo(touch.clientX - canvas.offsetLeft, touch.clientY - canvas.offsetTop);
  ctx.stroke();
});

function draw(event) {
  if (!isDrawing) return;

  ctx.lineWidth = 10;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'white';

  ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
  ctx.stroke();
}

function clear(){
    let ctx = canvas.getContext('2d');
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
// This funtion is lauch by the timer, it will make the prediction, push in the list of prediction and display it.
function eval () {
  predict().then((result) => {
    console.log(result);
    resultofpred.push(labels[result]);
    displayArray(resultofpred);
  });
}
// Display the array of prediction
// If the value is an operation, it will be displayed as a symbol
async function displayArray(array) {
  const arrayDisplayDiv = document.getElementById("arrayDisplay");
  const valueMap = {
    'add': '+',
    'mul': '*',
    'sub': '-',
    'div': '/'
  };
  const transformedArray = array.map(item => valueMap[item] || item);
  arrayDisplayDiv.textContent = transformedArray.join("");
}

// Predict the number drawn on the canvas
// The preprocess code adapted from Hugo Daumain
// Binary image conversion 
async function predict() {
  // Redimensionnez l'image du canvas à 28x28
  let tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = 28;
  tmpCanvas.height = 28;
  let tmpCtx = tmpCanvas.getContext('2d');

  tmpCtx.drawImage(canvas, 0, 0, 28, 28);


  // Extraire les pixels et les convertir en niveaux de gris
  let NOT_ZERO_NUMB = 0
  let pix_num = 0
  let imgData = tmpCtx.getImageData(0, 0, 28, 28).data;


  for (let i = 0; i < imgData.length; i+=1){
      if (imgData[i]!=0){
          NOT_ZERO_NUMB += 1; }
      pix_num += 1
  }

  let input = new Float32Array(28 * 28);
  //let array2D = [];
  for (let i = 0; i < imgData.length; i += 4) {
      let grayscale = 0
      if (imgData[i]!=0)  {grayscale = 255}
      if (imgData[i+1]!=0)  {grayscale = 255}
      if (imgData[i+2]!=0)  {grayscale = 255}
      grayscale = (((grayscale/ 255)) - 0.1736) / 0.3317;
      // Normaliser entre -1 et 1
      input[i/4] = grayscale
  }

  // Binariser l'image
  for (let i = 0; i < input.length; i += 1) {
    if (input[i] < 1) {
      input[i] = 0;
    }

  }

  let tensorInput = new onnx.Tensor(input, 'float32', [1, 1, 28, 28]);
  let outputMap = await model.run([tensorInput]);
  
  let outputData = outputMap.values().next().value.data;

  // Retourner la classe avec la plus haute probabilité
  console.log(outputData.indexOf(Math.max(...outputData)))

  return outputData.indexOf(Math.max(...outputData));
}

/*
  * This function will make the operation in the list of prediction
  * It will first make the multiplication and division, then the addition and substraction
  * It will then update the list of prediction with the result
*/
function MakeOperation() {
  const numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  const operations = ['add', 'div', 'mul', 'sub'];
  let array = resultofpred;
  let temp = [];
  let temp2 = [];
  let operationsArray = [];
  let number;
  console.log(array)
  for (let i = 0; i < array.length; i++) {
    if (numbers.includes(array[i])) {
      temp.push(array[i]);
    }
    else if (typeof array[i] === 'number'){
      temp2.push(array[i]);
    }
    else{
      if (temp.length > 0) {
        number = temp.join('');
        temp2.push(parseInt(number));
        temp = [];
      }
      operationsArray.push(array[i]);
    }
  }
  number = temp.join('');
  temp2.push(parseInt(number));
  console.log(operationsArray)
  console.log(temp2);
  while (operationsArray.includes('div') || operationsArray.includes('mul')) {
    if (operationsArray.includes('mul')){
      let index = operationsArray.indexOf('mul');
      let result = temp2[index] * temp2[index + 1];
      temp2.splice(index, 2, result);
      operationsArray.splice(index, 1);
    }
    if (operationsArray.includes('div')){
      let index = operationsArray.indexOf('div');
      let result = temp2[index] / temp2[index + 1];
      temp2.splice(index, 2, result);
      operationsArray.splice(index, 1);
    }
  }
  while (operationsArray.includes('add') || operationsArray.includes('sub')) {
    if (operationsArray.includes('add')){
      let index = operationsArray.indexOf('add');
      let result = temp2[index] + temp2[index + 1];
      console.log(result)
      console.log(index)
      temp2.splice(index, 2, result);
      operationsArray.splice(index, 1);
    }
    if (operationsArray.includes('sub')){
      let index = operationsArray.indexOf('sub');
      let result = temp2[index] - temp2[index + 1];
      temp2.splice(index, 2, result);
      operationsArray.splice(index, 1);
    }
  }
  resultofpred = temp2;
  displayArray(resultofpred)
}

const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;

// Initialize canvas
ctx.fillStyle = "black";
ctx.lineWidth = 10;
ctx.lineCap = "round";
ctx.strokeStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Drawing event handlers
canvas.addEventListener("mousedown", () => {
  isDrawing = true;
});
canvas.addEventListener("mouseup", () => {
  isDrawing = false;
  ctx.beginPath();
});
canvas.addEventListener("mousemove", draw);

// Draw function
function draw(event) {
  if (!isDrawing) return;
  ctx.lineTo(
    event.clientX - canvas.offsetLeft,
    event.clientY - canvas.offsetTop
  );
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(
    event.clientX - canvas.offsetLeft,
    event.clientY - canvas.offsetTop
  );
}

// Clear button
document.getElementById("clearButton").addEventListener("click", () => {
  ctx.fillRect(0, 0, canvas.width, canvas.height);
});

// Predict button
document.getElementById("predictButton").addEventListener("click", async () => {
  // Resize the canvas drawing to 28x28 and normalize
  const resizedImage = getResizedImage(canvas, 28, 28);
  const normalizedImage = normalizeImage(resizedImage);

  // Load the ONNX model and make a prediction
  try {
    const modelUrl = "./model/model.onnx"; // Path to your ONNX model
    const session = await ort.InferenceSession.create(modelUrl);

    console.log(normalizedImage);
    const inputTensor = new ort.Tensor(
      "float32",
      normalizedImage,
      [1, 1, 28, 28]
    );
    console.log(inputTensor);
    const feeds = { input: inputTensor }; // Replace "input" with the actual input name
    const results = await session.run(feeds);

    const output = results[Object.keys(results)[0]].data;
    const predictedDigit = output.indexOf(Math.max(...output));
    document.getElementById(
      "output"
    ).textContent = `Prediction: ${predictedDigit}`;
  } catch (error) {
    console.error("Error during inference:", error);
    document.getElementById("output").textContent =
      "Error during prediction. Check the console for details.";
  }
});

// Helper functions
function getResizedImage(canvas, width, height) {
  const offScreenCanvas = document.createElement("canvas");
  offScreenCanvas.width = width;
  offScreenCanvas.height = height;
  const offScreenCtx = offScreenCanvas.getContext("2d");

  offScreenCtx.drawImage(canvas, 0, 0, width, height);
  return offScreenCtx.getImageData(0, 0, width, height);
}

function normalizeImage(imageData) {
  const { data, width, height } = imageData;
  const normalized = new Float32Array(width * height);

  for (let i = 0; i < data.length; i += 4) {
    var grayscale = data[i] / 255.0; // Normalize pixel to [0, 1]
    grayscale = grayscale - 0.1309;
    grayscale = grayscale / 0.3018;
    normalized[i / 4] = grayscale;
  }
  console.log(normalized);
  return normalized;
}

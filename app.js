const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const toggleClassButton = document.getElementById('toggleClass');
const trainButton = document.getElementById('trainButton');
const restartButton = document.getElementById('restartButton');
const statusDisplay = document.getElementById('statusDisplay');
const epochInput = document.getElementById('epochInput');

let isBlueClass = true; // Toggle between blue and red classes
let data = []; // Store {x, y, class} points

// Canvas click listener to add data points
canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Add point to data
    data.push({ x, y, class: isBlueClass ? 0 : 1 });

    // Draw point on canvas
    drawStylishDot(x, y, isBlueClass ? 'blue' : 'red');
});

// Toggle class button
toggleClassButton.addEventListener('click', () => {
    isBlueClass = !isBlueClass;
    toggleClassButton.textContent = `Current Class: ${isBlueClass ? 'Blue' : 'Red'}`;
    toggleClassButton.classList.toggle('blue', isBlueClass);
    toggleClassButton.classList.toggle('red', !isBlueClass);
});

// Restart button
restartButton.addEventListener('click', () => {
    data = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    statusDisplay.textContent = 'Canvas cleared. Add new data points!';
});

// Function to draw stylish dots with a solid fill and border
const drawStylishDot = (x, y, color) => {
    const dotSize = 12; // Dot radius
    const borderSize = 3; // Border thickness

    // Set up the shadow for the dot
    ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
    ctx.shadowBlur = 5;
    ctx.shadowOffsetX = 2;
    ctx.shadowOffsetY = 2;

    // Draw border first (white or light color) for better contrast
    ctx.lineWidth = borderSize;
    ctx.strokeStyle = 'white'; // White border
    ctx.beginPath();
    ctx.arc(x, y, dotSize, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw the dot with the specified color
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, dotSize - borderSize, 0, 2 * Math.PI); // Smaller circle for fill
    ctx.fill();

    // Reset shadow after drawing
    ctx.shadowColor = 'transparent'; 
};

// Normalize data to [-1, 1]
const normalize = (x, y) => {
    return [(x / canvas.width) * 2 - 1, (y / canvas.height) * 2 - 1];
};

// Draw classification boundary
const drawDecisionBoundary = async (model) => {
    const resolution = 200;
    const scaledWidth = canvas.width / resolution;
    const scaledHeight = canvas.height / resolution;

    const inputs = [];
    for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
            const nx = (i / resolution) * 2 - 1;
            const ny = (j / resolution) * 2 - 1;
            inputs.push([nx, ny]);
        }
    }

    const predictions = tf.tidy(() => {
        const inputTensor = tf.tensor2d(inputs);
        return model.predict(inputTensor).dataSync();
    });

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;

    for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
            const prediction = predictions[i * resolution + j];
            const alpha = 50;
            const color = prediction > 0.5 ? [255, 0, 0] : [0, 0, 255];

            const xStart = Math.floor(i * scaledWidth);
            const yStart = Math.floor(j * scaledHeight);
            const xEnd = Math.min(xStart + scaledWidth, canvas.width);
            const yEnd = Math.min(yStart + scaledHeight, canvas.height);

            for (let x = xStart; x < xEnd; x++) {
                for (let y = yStart; y < yEnd; y++) {
                    const index = (y * canvas.width + x) * 4;
                    pixels[index] = color[0];
                    pixels[index + 1] = color[1];
                    pixels[index + 2] = color[2];
                    pixels[index + 3] = alpha;
                }
            }
        }
    }

    ctx.putImageData(imageData, 0, 0);
    drawDataPoints();
};

const drawDataPoints = () => {
    for (const point of data) {
        drawStylishDot(point.x, point.y, point.class === 0 ? 'blue' : 'red');
    }
};

// Train model
trainButton.addEventListener('click', async () => {
    if (data.length < 2) {
        statusDisplay.textContent = 'Not enough data to train!';
        return;
    }

    const totalEpochs = parseInt(epochInput.value) || 1000;
    const updateInterval = 10;

    statusDisplay.textContent = `Training for ${totalEpochs} epochs...`;

    const xs = tf.tensor2d(data.map(p => normalize(p.x, p.y)));
    const ys = tf.tensor2d(data.map(p => [p.class]));

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [2], units: 4, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });

    for (let epoch = 0; epoch < totalEpochs; epoch++) {
        await model.fit(xs, ys, { epochs: 1 });
        if ((epoch + 1) % updateInterval === 0 || epoch === totalEpochs - 1) {
            statusDisplay.textContent = `Epoch ${epoch + 1}/${totalEpochs}. Updating boundary...`;
            await drawDecisionBoundary(model);
        }
    }

    statusDisplay.textContent = 'Training Complete!';
    xs.dispose();
    ys.dispose();
    model.dispose();
});

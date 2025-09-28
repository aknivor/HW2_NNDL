// Titanic Binary Classifier using TensorFlow.js
// Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
// Target: Survived (0/1)
// Identifier: PassengerId (excluded from features)

// Global variables
let trainData = null;
let testData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let testPredictions = null;
let rocData = null;
let auc = 0;

// Data Load & Inspection
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both train and test CSV files');
        return;
    }

    try {
        // Load CSV files
        const trainText = await readFile(trainFile);
        const testText = await readFile(testFile);
        
        // Parse CSV data
        trainData = parseCSV(trainText);
        testData = parseCSV(testText);
        
        // Display data info
        displayDataInfo();
        
        // Visualize data
        visualizeData();
        
    } catch (error) {
        alert('Error loading data: ' + error.message);
    }
}

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('File reading failed'));
        reader.readAsText(file);
    });
}

function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = parseCSVLine(lines[0]);
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        const row = {};
        headers.forEach((header, index) => {
            row[header.trim()] = values[index] ? values[index].trim() : '';
        });
        data.push(row);
    }
    
    return data;
}

function parseCSVLine(line) {
    const values = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (char === '"') {
            // Toggle quote state
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            // Comma outside quotes - end of field
            values.push(current);
            current = '';
        } else {
            // Regular character
            current += char;
        }
        
        // Handle double quotes (escaped quotes in CSV)
        if (char === '"' && nextChar === '"' && inQuotes) {
            current += '"';
            i++; // Skip next quote
        }
    }
    
    // Push the last field
    values.push(current);
    
    return values;
}

function displayDataInfo() {
    const infoDiv = document.getElementById('data-info');
    
    // Calculate missing values percentage
    const missingPercent = calculateMissingPercent(trainData);
    
    infoDiv.innerHTML = `
        <h3>Data Overview</h3>
        <p><strong>Train Data Shape:</strong> ${trainData.length} rows × ${Object.keys(trainData[0]).length} columns</p>
        <p><strong>Test Data Shape:</strong> ${testData.length} rows × ${Object.keys(testData[0]).length} columns</p>
        <p><strong>Missing Values in Train Data:</strong></p>
        <ul>
            ${Object.entries(missingPercent).map(([key, value]) => 
                `<li>${key}: ${(value * 100).toFixed(1)}%</li>`
            ).join('')}
        </ul>
        
        <h3>Data Preview (First 5 rows)</h3>
        ${generateTablePreview(trainData.slice(0, 5))}
    `;
}

function calculateMissingPercent(data) {
    const missing = {};
    const total = data.length;
    
    Object.keys(data[0]).forEach(key => {
        const missingCount = data.filter(row => !row[key] || row[key] === '').length;
        missing[key] = missingCount / total;
    });
    
    return missing;
}

function generateTablePreview(data) {
    if (data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    let html = '<table><tr>';
    
    // Headers
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr>';
    
    // Rows
    data.forEach(row => {
        html += '<tr>';
        headers.forEach(header => {
            html += `<td>${row[header]}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</table>';
    return html;
}

function visualizeData() {
    // Prepare data for visualization
    const survivalBySex = {};
    const survivalByPclass = {};
    
    trainData.forEach(row => {
        // Survival by Sex
        const sex = row.Sex;
        if (!survivalBySex[sex]) survivalBySex[sex] = { survived: 0, total: 0 };
        survivalBySex[sex].total++;
        if (row.Survived === '1') survivalBySex[sex].survived++;
        
        // Survival by Pclass
        const pclass = `Class ${row.Pclass}`;
        if (!survivalByPclass[pclass]) survivalByPclass[pclass] = { survived: 0, total: 0 };
        survivalByPclass[pclass].total++;
        if (row.Survived === '1') survivalByPclass[pclass].survived++;
    });
    
    // Convert to tfjs-vis format
    const sexData = {
        values: Object.entries(survivalBySex).map(([sex, stats]) => ({
            x: sex,
            y: (stats.survived / stats.total) * 100
        }))
    };
    
    const pclassData = {
        values: Object.entries(survivalByPclass).map(([pclass, stats]) => ({
            x: pclass,
            y: (stats.survived / stats.total) * 100
        }))
    };
    
    // Render charts
    tfvis.render.barchart(
        { name: 'Survival Rate by Sex', tab: 'Data Inspection' },
        sexData,
        { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
    );
    
    tfvis.render.barchart(
        { name: 'Survival Rate by Passenger Class', tab: 'Data Inspection' },
        pclassData,
        { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' }
    );
}

// Preprocessing
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first');
        return;
    }

    try {
        // Preprocess train data
        const processedTrain = preprocessDataset(trainData, true);
        
        // Preprocess test data (using statistics from train data for consistency)
        const processedTest = preprocessDataset(testData, false, {
            ageMedian: processedTrain.stats.ageMedian,
            fareMean: processedTrain.stats.fareMean,
            fareStd: processedTrain.stats.fareStd,
            embarkedMode: processedTrain.stats.embarkedMode
        });
        
        // Store processed data
        trainData.processed = processedTrain;
        testData.processed = processedTest;
        
        displayPreprocessInfo(processedTrain, processedTest);
        
    } catch (error) {
        alert('Error preprocessing data: ' + error.message);
    }
}

function preprocessDataset(data, isTraining, stats = null) {
    // Calculate statistics from training data
    if (isTraining) {
        const ages = data.map(row => parseFloat(row.Age)).filter(age => !isNaN(age));
        const fares = data.map(row => parseFloat(row.Fare)).filter(fare => !isNaN(fare));
        const embarked = data.map(row => row.Embarked).filter(e => e);
        
        // Calculate median manually
        const calculateMedian = (arr) => {
            if (arr.length === 0) return 0;
            const sorted = [...arr].sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        };
        
        // Calculate mode manually
        const calculateMode = (arr) => {
            if (arr.length === 0) return 'S';
            const frequency = {};
            let maxCount = 0;
            let mode = arr[0];
            
            arr.forEach(item => {
                frequency[item] = (frequency[item] || 0) + 1;
                if (frequency[item] > maxCount) {
                    maxCount = frequency[item];
                    mode = item;
                }
            });
            return mode;
        };
        
        stats = {
            ageMedian: calculateMedian(ages),
            fareMean: fares.length > 0 ? tf.mean(tf.tensor1d(fares)).dataSync()[0] : 0,
            fareStd: fares.length > 0 ? tf.tensor1d(fares).std().dataSync()[0] : 1,
            embarkedMode: calculateMode(embarked)
        };
    }
    
    // Extract features and labels
    const features = [];
    const labels = [];
    const identifiers = [];
    
    data.forEach(row => {
        // Handle missing values
        const age = isNaN(parseFloat(row.Age)) ? stats.ageMedian : parseFloat(row.Age);
        const fare = isNaN(parseFloat(row.Fare)) ? 0 : parseFloat(row.Fare);
        const embarked = row.Embarked || stats.embarkedMode;
        
        // Create feature vector
        const featureVector = [
            // Numerical features (standardized)
            age,
            (fare - stats.fareMean) / stats.fareStd,
            parseFloat(row.SibSp) || 0,
            parseFloat(row.Parch) || 0,
            
            // One-hot encoded Sex (male: [1,0], female: [0,1])
            row.Sex === 'male' ? 1 : 0,
            row.Sex === 'female' ? 1 : 0,
            
            // One-hot encoded Pclass
            row.Pclass === '1' ? 1 : 0,
            row.Pclass === '2' ? 1 : 0,
            row.Pclass === '3' ? 1 : 0,
            
            // One-hot encoded Embarked
            embarked === 'C' ? 1 : 0,
            embarked === 'Q' ? 1 : 0,
            embarked === 'S' ? 1 : 0
        ];
        
        features.push(featureVector);
        
        // Labels (only for training data)
        if (isTraining && row.Survived !== undefined) {
            labels.push(parseInt(row.Survived));
        }
        
        identifiers.push(row.PassengerId);
    });
    
    return {
        features: tf.tensor2d(features),
        labels: isTraining && labels.length > 0 ? tf.tensor1d(labels) : null,
        identifiers: identifiers,
        stats: stats
    };
}

function displayPreprocessInfo(trainProcessed, testProcessed) {
    const infoDiv = document.getElementById('preprocess-info');
    
    infoDiv.innerHTML = `
        <h3>Preprocessing Complete</h3>
        <p><strong>Processed Train Features Shape:</strong> ${trainProcessed.features.shape}</p>
        <p><strong>Processed Test Features Shape:</strong> ${testProcessed.features.shape}</p>
        <p><strong>Feature Names:</strong> Age, Fare_std, SibSp, Parch, Sex_male, Sex_female, 
           Pclass_1, Pclass_2, Pclass_3, Embarked_C, Embarked_Q, Embarked_S</p>
        <p><strong>Age Median:</strong> ${trainProcessed.stats.ageMedian.toFixed(2)}</p>
        <p><strong>Fare Mean:</strong> ${trainProcessed.stats.fareMean.toFixed(2)}</p>
        <p><strong>Fare Std:</strong> ${trainProcessed.stats.fareStd.toFixed(2)}</p>
        <p><strong>Embarked Mode:</strong> ${trainProcessed.stats.embarkedMode}</p>
    `;
}

// Model Creation
function createModel() {
    if (!trainData.processed) {
        alert('Please preprocess data first');
        return;
    }

    try {
        // Define model architecture
        model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [trainData.processed.features.shape[1]],
                    units: 16,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid'
                })
            ]
        });
        
        // Compile model
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        displayModelInfo();
        
    } catch (error) {
        alert('Error creating model: ' + error.message);
    }
}

function displayModelInfo() {
    const infoDiv = document.getElementById('model-info');
    
    // Count total parameters
    let totalParams = 0;
    model.summary(null, null, (line) => {
        const match = line.match(/params:\s*([\d,]+)/);
        if (match) {
            totalParams += parseInt(match[1].replace(/,/g, ''));
        }
    });
    
    // Create summary
    let summary = '';
    model.summary(null, null, (line) => {
        summary += line + '<br>';
    });
    
    infoDiv.innerHTML = `
        <h3>Model Created Successfully</h3>
        <div><strong>Model Summary:</strong><br>${summary}</div>
        <p><strong>Total Parameters:</strong> ${totalParams.toLocaleString()}</p>
    `;
}

// Training
async function trainModel() {
    if (!model) {
        alert('Please create model first');
        return;
    }

    try {
        // Prepare training and validation data (80/20 stratified split)
        const {xTrain, yTrain, xVal, yVal} = prepareTrainingData();
        validationData = [xVal, yVal];
        
        // Train model
        trainingHistory = await model.fit(xTrain, yTrain, {
            epochs: 50,
            batchSize: 32,
            validationData: validationData,
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance', tab: 'Training' },
                ['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                { callbacks: ['onEpochEnd'] }
            )
        });
        
        // Calculate ROC/AUC
        calculateROC();
        
        // Update metrics with default threshold
        updateMetrics();
        
        document.getElementById('training-info').innerHTML = 
            '<p><strong>Training completed successfully!</strong></p>';
            
    } catch (error) {
        alert('Error training model: ' + error.message);
    }
}

function prepareTrainingData() {
    const features = trainData.processed.features;
    const labels = trainData.processed.labels;
    
    // Get indices for stratified split
    const positiveIndices = [];
    const negativeIndices = [];
    
    const labelsArray = labels.dataSync();
    for (let i = 0; i < labelsArray.length; i++) {
        if (labelsArray[i] === 1) {
            positiveIndices.push(i);
        } else {
            negativeIndices.push(i);
        }
    }
    
    // Shuffle indices
    tf.util.shuffle(positiveIndices);
    tf.util.shuffle(negativeIndices);
    
    // Calculate split sizes (80/20)
    const posValSize = Math.floor(positiveIndices.length * 0.2);
    const negValSize = Math.floor(negativeIndices.length * 0.2);
    
    const valIndices = [
        ...positiveIndices.slice(0, posValSize),
        ...negativeIndices.slice(0, negValSize)
    ];
    const trainIndices = [
        ...positiveIndices.slice(posValSize),
        ...negativeIndices.slice(negValSize)
    ];
    
    tf.util.shuffle(valIndices);
    tf.util.shuffle(trainIndices);
    
    // Create train/validation datasets
    const xTrain = tf.gather(features, trainIndices);
    const yTrain = tf.gather(labels, trainIndices);
    const xVal = tf.gather(features, valIndices);
    const yVal = tf.gather(labels, valIndices);
    
    return { xTrain, yTrain, xVal, yVal };
}

function calculateROC() {
    if (!validationData) return;
    
    const [xVal, yVal] = validationData;
    const predictions = model.predict(xVal).dataSync();
    const trueLabels = yVal.dataSync();
    
    // Generate ROC data
    const thresholds = Array.from({length: 101}, (_, i) => i / 100);
    rocData = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];
            
            if (pred === 1 && actual === 1) tp++;
            else if (pred === 1 && actual === 0) fp++;
            else if (pred === 0 && actual === 0) tn++;
            else if (pred === 0 && actual === 1) fn++;
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocData.push({ fpr, tpr, threshold, tp, fp, tn, fn });
    });
    
    // Calculate AUC using trapezoidal rule
    auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].fpr - rocData[i-1].fpr) * 
               (rocData[i].tpr + rocData[i-1].tpr) / 2;
    }
    
    // Plot ROC curve
    const rocValues = rocData.map(point => ({ x: point.fpr, y: point.tpr }));
    tfvis.render.scatterplot(
        { name: `ROC Curve (AUC = ${auc.toFixed(3)})`, tab: 'Metrics' },
        { values: rocValues },
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            styles: { line: { color: 'blue' } }
        }
    );
}

// Metrics Update
function updateMetrics() {
    if (!rocData) return;
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    // Find closest ROC point to current threshold
    const rocPoint = rocData.reduce((prev, curr) => 
        Math.abs(curr.threshold - threshold) < Math.abs(prev.threshold - threshold) ? curr : prev
    );
    
    const precision = rocPoint.tp / (rocPoint.tp + rocPoint.fp) || 0;
    const recall = rocPoint.tpr;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (rocPoint.tp + rocPoint.tn) / (rocPoint.tp + rocPoint.fp + rocPoint.tn + rocPoint.fn) || 0;
    
    // Update confusion matrix display
    document.getElementById('confusion-matrix').innerHTML = `
        <h3>Confusion Matrix (Threshold: ${threshold.toFixed(2)})</h3>
        <table>
            <tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>
            <tr><th>Actual Negative</th><td>${rocPoint.tn}</td><td>${rocPoint.fp}</td></tr>
            <tr><th>Actual Positive</th><td>${rocPoint.fn}</td><td>${rocPoint.tp}</td></tr>
        </table>
    `;
    
    // Update metrics display
    document.getElementById('metrics-values').innerHTML = `
        <h3>Performance Metrics</h3>
        <p><strong>Accuracy:</strong> ${accuracy.toFixed(3)}</p>
        <p><strong>Precision:</strong> ${precision.toFixed(3)}</p>
        <p><strong>Recall:</strong> ${recall.toFixed(3)}</p>
        <p><strong>F1-Score:</strong> ${f1.toFixed(3)}</p>
        <p><strong>AUC:</strong> ${auc.toFixed(3)}</p>
    `;
}

// Prediction
async function predictTest() {
    if (!model || !testData.processed) {
        alert('Please train model and preprocess test data first');
        return;
    }

    try {
        const testFeatures = testData.processed.features;
        const predictions = model.predict(testFeatures);
        const probabilities = predictions.dataSync();
        
        testPredictions = {
            identifiers: testData.processed.identifiers,
            probabilities: probabilities
        };
        
        document.getElementById('prediction-info').innerHTML = `
            <p><strong>Predictions generated for ${probabilities.length} test samples</strong></p>
            <p>First 5 predictions:</p>
            <ul>
                ${probabilities.slice(0, 5).map((prob, i) => 
                    `<li>Passenger ${testPredictions.identifiers[i]}: ${prob.toFixed(3)}</li>`
                ).join('')}
            </ul>
        `;
        
    } catch (error) {
        alert('Error making predictions: ' + error.message);
    }
}

// Export Functions
async function exportModel() {
    if (!model) {
        alert('Please create and train model first');
        return;
    }

    try {
        await model.save('downloads://titanic-tfjs-model');
        alert('Model exported successfully!');
    } catch (error) {
        alert('Error exporting model: ' + error.message);
    }
}

function downloadPredictions() {
    if (!testPredictions) {
        alert('Please generate predictions first');
        return;
    }

    try {
        // Create submission CSV (PassengerId, Survived)
        let submissionCSV = 'PassengerId,Survived\n';
        const threshold = parseFloat(document.getElementById('threshold-slider').value);
        
        testPredictions.probabilities.forEach((prob, i) => {
            const survived = prob >= threshold ? 1 : 0;
            submissionCSV += `${testPredictions.identifiers[i]},${survived}\n`;
        });
        
        // Create probabilities CSV
        let probabilitiesCSV = 'PassengerId,Probability\n';
        testPredictions.probabilities.forEach((prob, i) => {
            probabilitiesCSV += `${testPredictions.identifiers[i]},${prob.toFixed(4)}\n`;
        });
        
        // Download files
        downloadFile(submissionCSV, 'submission.csv', 'text/csv');
        downloadFile(probabilitiesCSV, 'probabilities.csv', 'text/csv');
        
    } catch (error) {
        alert('Error downloading predictions: ' + error.message);
    }
}

function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

// Initialize the application
console.log('Titanic Classifier App initialized. Load your CSV files to begin.');

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
        const trainText = await readFile(trainFile);
        const testText = await readFile(testFile);
        
        trainData = parseCSV(trainText);
        testData = parseCSV(testText);
        
        displayDataInfo();
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
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            values.push(current);
            current = '';
        } else {
            current += char;
        }
        
        if (char === '"' && nextChar === '"' && inQuotes) {
            current += '"';
            i++;
        }
    }
    
    values.push(current);
    return values;
}

function displayDataInfo() {
    const infoDiv = document.getElementById('data-info');
    
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
    
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr>';
    
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
    const survivalBySex = {};
    const survivalByPclass = {};
    
    trainData.forEach(row => {
        const sex = row.Sex;
        if (!survivalBySex[sex]) survivalBySex[sex] = { survived: 0, total: 0 };
        survivalBySex[sex].total++;
        if (row.Survived === '1') survivalBySex[sex].survived++;
        
        const pclass = `Class ${row.Pclass}`;
        if (!survivalByPclass[pclass]) survivalByPclass[pclass] = { survived: 0, total: 0 };
        survivalByPclass[pclass].total++;
        if (row.Survived === '1') survivalByPclass[pclass].survived++;
    });
    
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
        console.log('Preprocessing training data...');
        const processedTrain = preprocessDataset(trainData, true);
        
        console.log('Preprocessing test data...');
        const processedTest = preprocessDataset(testData, false, processedTrain.stats);
        
        trainData.processed = processedTrain;
        testData.processed = processedTest;
        
        displayPreprocessInfo(processedTrain, processedTest);
        
    } catch (error) {
        alert('Error preprocessing data: ' + error.message);
        console.error('Preprocessing error:', error);
    }
}

// Helper functions for statistics
function calculateMedian(arr) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function calculateMean(arr) {
    if (arr.length === 0) return 0;
    return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

function calculateStd(arr) {
    if (arr.length === 0) return 1;
    const mean = calculateMean(arr);
    const squaredDiffs = arr.map(val => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / arr.length;
    return Math.sqrt(variance);
}

function calculateMode(arr) {
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
}

function preprocessDataset(data, isTraining, stats = null) {
    console.log(`Preprocessing ${data.length} rows, isTraining: ${isTraining}`);
    
    // Calculate statistics from training data
    if (isTraining) {
        console.log('Calculating training statistics...');
        
        // Extract and clean data
        const ages = [];
        const fares = [];
        const embarked = [];
        
        for (let i = 0; i < data.length; i++) {
            const row = data[i];
            
            // Age
            const ageVal = parseFloat(row.Age);
            if (!isNaN(ageVal) && isFinite(ageVal)) {
                ages.push(ageVal);
            }
            
            // Fare
            const fareVal = parseFloat(row.Fare);
            if (!isNaN(fareVal) && isFinite(fareVal)) {
                fares.push(fareVal);
            }
            
            // Embarked
            if (row.Embarked && row.Embarked.trim() !== '') {
                embarked.push(row.Embarked);
            }
        }
        
        console.log(`Valid data - Ages: ${ages.length}, Fares: ${fares.length}, Embarked: ${embarked.length}`);
        
        // Calculate statistics with fallbacks
        const ageMedian = ages.length > 0 ? calculateMedian(ages) : 28.0;
        const fareMean = fares.length > 0 ? calculateMean(fares) : 32.20;
        const fareStd = fares.length > 0 ? calculateStd(fares) : 49.69;
        const embarkedMode = embarked.length > 0 ? calculateMode(embarked) : 'S';
        
        stats = {
            ageMedian: ageMedian,
            fareMean: fareMean,
            fareStd: fareStd,
            embarkedMode: embarkedMode
        };
        
        console.log('Training statistics:', stats);
    }
    
    const features = [];
    const labels = [];
    const identifiers = [];
    let processedCount = 0;
    
    // Process each row
    for (let i = 0; i < data.length; i++) {
        const row = data[i];
        
        try {
            // Parse Age with fallback
            let age = parseFloat(row.Age);
            if (isNaN(age) || !isFinite(age)) {
                age = stats.ageMedian;
            }
            
            // Parse Fare with fallback
            let fare = parseFloat(row.Fare);
            if (isNaN(fare) || !isFinite(fare)) {
                fare = stats.fareMean;
            }
            
            // Handle Embarked
            let embarked = row.Embarked;
            if (!embarked || embarked.trim() === '') {
                embarked = stats.embarkedMode;
            }
            
            // Parse other numeric fields
            const sibsp = parseInt(row.SibSp) || 0;
            const parch = parseInt(row.Parch) || 0;
            
            // Create feature vector - SIMPLIFIED to avoid standardization issues
            const featureVector = [
                age,                           // Raw age
                fare,                          // Raw fare (no standardization)
                sibsp,
                parch,
                row.Sex === 'male' ? 1 : 0,    // Male
                row.Sex === 'female' ? 1 : 0,  // Female
                row.Pclass === '1' ? 1 : 0,    // Pclass 1
                row.Pclass === '2' ? 1 : 0,    // Pclass 2  
                row.Pclass === '3' ? 1 : 0,    // Pclass 3
                embarked === 'C' ? 1 : 0,      // Embarked C
                embarked === 'Q' ? 1 : 0,      // Embarked Q
                embarked === 'S' ? 1 : 0       // Embarked S
            ];
            
            // Validate all values are finite numbers
            let isValid = true;
            for (let j = 0; j < featureVector.length; j++) {
                const val = featureVector[j];
                if (isNaN(val) || !isFinite(val)) {
                    console.warn(`Invalid value at index ${j} in row ${i}:`, val);
                    isValid = false;
                    break;
                }
            }
            
            if (!isValid) {
                continue;
            }
            
            features.push(featureVector);
            processedCount++;
            
            // Handle labels for training data
            if (isTraining && row.Survived !== undefined && row.Survived !== '') {
                const label = parseInt(row.Survived);
                if (label === 0 || label === 1) {
                    labels.push(label);
                }
            }
            
            identifiers.push(row.PassengerId);
            
        } catch (error) {
            console.error(`Error processing row ${i}:`, error);
        }
    }
    
    console.log(`Successfully processed ${processedCount} of ${data.length} rows`);
    
    if (features.length === 0) {
        throw new Error('No valid features could be processed');
    }
    
    // Create tensors
    const featuresTensor = tf.tensor2d(features);
    const labelsTensor = isTraining && labels.length > 0 ? tf.tensor1d(labels) : null;
    
    // Verify tensors don't contain NaN
    const featuresArray = featuresTensor.arraySync();
    let hasNaN = false;
    for (let i = 0; i < Math.min(5, featuresArray.length); i++) {
        for (let j = 0; j < featuresArray[i].length; j++) {
            if (isNaN(featuresArray[i][j])) {
                console.error(`NaN found in features at [${i}][${j}]`);
                hasNaN = true;
            }
        }
    }
    
    if (hasNaN) {
        throw new Error('Features tensor contains NaN values');
    }
    
    console.log('Features tensor shape:', featuresTensor.shape);
    console.log('Sample features:', featuresArray[0]);
    
    return {
        features: featuresTensor,
        labels: labelsTensor,
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
        const inputDim = trainData.processed.features.shape[1];
        console.log(`Creating model with input dimension: ${inputDim}`);
        
        model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputDim],
                    units: 8,  // Reduced complexity
                    activation: 'relu',
                    kernelInitializer: 'glorotNormal'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid'
                })
            ]
        });
        
        model.compile({
            optimizer: tf.train.adam(0.01),  // Explicit optimizer with learning rate
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
    
    let summaryLines = [];
    let totalParams = 0;
    
    model.summary(null, null, (line) => {
        summaryLines.push(line);
        const match = line.match(/params:\s*([\d,]+)/);
        if (match) {
            totalParams += parseInt(match[1].replace(/,/g, ''));
        }
    });
    
    infoDiv.innerHTML = `
        <h3>Model Created Successfully</h3>
        <div><strong>Model Summary:</strong><br><pre>${summaryLines.join('\n')}</pre></div>
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
        const {xTrain, yTrain, xVal, yVal} = prepareTrainingData();
        validationData = [xVal, yVal];
        
        console.log('Starting training...');
        console.log('Training data shapes - xTrain:', xTrain.shape, 'yTrain:', yTrain.shape);
        console.log('Validation data shapes - xVal:', xVal.shape, 'yVal:', yVal.shape);
        
        // Verify no NaN in training data
        const xTrainHasNaN = tf.isNaN(xTrain).any().dataSync()[0];
        const yTrainHasNaN = tf.isNaN(yTrain).any().dataSync()[0];
        
        if (xTrainHasNaN || yTrainHasNaN) {
            throw new Error('Training data contains NaN values');
        }
        
        trainingHistory = await model.fit(xTrain, yTrain, {
            epochs: 30,  // Reduced epochs
            batchSize: 32,
            validationData: validationData,
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance', tab: 'Training' },
                ['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                { callbacks: ['onEpochEnd'] }
            )
        });
        
        // Get validation predictions for ROC
        const valPredictions = model.predict(xVal);
        const valProbs = valPredictions.dataSync();
        const valLabels = yVal.dataSync();
        
        // Check for NaN in validation predictions
        const valNaNCount = valProbs.filter(p => isNaN(p)).length;
        if (valNaNCount > 0) {
            console.warn(`Found ${valNaNCount} NaN values in validation predictions`);
        }
        
        calculateROC(valProbs, valLabels);
        updateMetrics();
        
        document.getElementById('training-info').innerHTML = 
            '<p><strong>Training completed successfully!</strong></p>';
            
    } catch (error) {
        alert('Error training model: ' + error.message);
        console.error('Training error:', error);
    }
}

function prepareTrainingData() {
    const features = trainData.processed.features;
    const labels = trainData.processed.labels;
    
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
    
    tf.util.shuffle(positiveIndices);
    tf.util.shuffle(negativeIndices);
    
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
    
    const xTrain = tf.gather(features, trainIndices);
    const yTrain = tf.gather(labels, trainIndices);
    const xVal = tf.gather(features, valIndices);
    const yVal = tf.gather(labels, valIndices);
    
    return { xTrain, yTrain, xVal, yVal };
}

function calculateROC(predictions, trueLabels) {
    console.log('Calculating ROC...');
    
    const thresholds = Array.from({length: 101}, (_, i) => i / 100);
    rocData = [];
    
    for (let i = 0; i < thresholds.length; i++) {
        const threshold = thresholds[i];
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let j = 0; j < predictions.length; j++) {
            const pred = predictions[j] >= threshold ? 1 : 0;
            const actual = trueLabels[j];
            
            if (pred === 1 && actual === 1) tp++;
            else if (pred === 1 && actual === 0) fp++;
            else if (pred === 0 && actual === 0) tn++;
            else if (pred === 0 && actual === 1) fn++;
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocData.push({ fpr, tpr, threshold, tp, fp, tn, fn });
    }
    
    auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].fpr - rocData[i-1].fpr) * 
               (rocData[i].tpr + rocData[i-1].tpr) / 2;
    }
    
    console.log(`AUC: ${auc.toFixed(3)}`);
    
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
    if (!rocData || rocData.length === 0) {
        document.getElementById('confusion-matrix').innerHTML = '<p>No metrics available yet. Train the model first.</p>';
        return;
    }
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    const rocPoint = rocData.reduce((prev, curr) => 
        Math.abs(curr.threshold - threshold) < Math.abs(prev.threshold - threshold) ? curr : prev
    );
    
    const precision = rocPoint.tp / (rocPoint.tp + rocPoint.fp) || 0;
    const recall = rocPoint.tpr;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (rocPoint.tp + rocPoint.tn) / (rocPoint.tp + rocPoint.fp + rocPoint.tn + rocPoint.fn) || 0;
    
    document.getElementById('confusion-matrix').innerHTML = `
        <h3>Confusion Matrix (Threshold: ${threshold.toFixed(2)})</h3>
        <table style="width: 100%; text-align: center; border-collapse: collapse;">
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px; background: #f2f2f2;"></th>
                <th style="border: 1px solid #ddd; padding: 8px; background: #f2f2f2;">Predicted Negative</th>
                <th style="border: 1px solid #ddd; padding: 8px; background: #f2f2f2;">Predicted Positive</th>
            </tr>
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px; background: #f2f2f2;">Actual Negative</th>
                <td style="border: 1px solid #ddd; padding: 8px;">${rocPoint.tn}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">${rocPoint.fp}</td>
            </tr>
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px; background: #f2f2f2;">Actual Positive</th>
                <td style="border: 1px solid #ddd; padding: 8px;">${rocPoint.fn}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">${rocPoint.tp}</td>
            </tr>
        </table>
    `;
    
    document.getElementById('metrics-values').innerHTML = `
        <h3>Performance Metrics</h3>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
            <p><strong>Accuracy:</strong> ${accuracy.toFixed(3)}</p>
            <p><strong>Precision:</strong> ${precision.toFixed(3)}</p>
            <p><strong>Recall:</strong> ${recall.toFixed(3)}</p>
            <p><strong>F1-Score:</strong> ${f1.toFixed(3)}</p>
            <p><strong>AUC:</strong> ${auc.toFixed(3)}</p>
        </div>
    `;
}

// Prediction
async function predictTest() {
    if (!model || !testData.processed) {
        alert('Please train model and preprocess test data first');
        return;
    }

    try {
        console.log('Making predictions on test data...');
        const testFeatures = testData.processed.features;
        
        // Check for NaN in test features
        const testFeaturesArray = testFeatures.arraySync();
        let testFeaturesNaN = false;
        for (let i = 0; i < Math.min(3, testFeaturesArray.length); i++) {
            for (let j = 0; j < testFeaturesArray[i].length; j++) {
                if (isNaN(testFeaturesArray[i][j])) {
                    console.error(`NaN in test features at [${i}][${j}]`);
                    testFeaturesNaN = true;
                }
            }
        }
        
        if (testFeaturesNaN) {
            throw new Error('Test features contain NaN values');
        }
        
        const predictions = model.predict(testFeatures);
        const probabilities = predictions.dataSync();
        
        // Check for NaN in predictions
        const nanCount = probabilities.filter(p => isNaN(p)).length;
        console.log(`Found ${nanCount} NaN values in predictions`);
        
        testPredictions = {
            identifiers: testData.processed.identifiers,
            probabilities: probabilities
        };
        
        // Display predictions
        let predictionDisplay = '';
        const displayCount = Math.min(5, probabilities.length);
        
        for (let i = 0; i < displayCount; i++) {
            const prob = probabilities[i];
            const passengerId = testPredictions.identifiers[i];
            
            if (isNaN(prob)) {
                predictionDisplay += `<li>Passenger ${passengerId}: <span style="color: red;">INVALID (NaN)</span></li>`;
            } else {
                predictionDisplay += `<li>Passenger ${passengerId}: ${prob.toFixed(4)}</li>`;
            }
        }
        
        document.getElementById('prediction-info').innerHTML = `
            <p><strong>Predictions generated for ${probabilities.length} test samples</strong></p>
            <p><strong style="color: ${nanCount > 0 ? 'red' : 'green'};">Invalid predictions (NaN): ${nanCount}</strong></p>
            <p>First ${displayCount} predictions:</p>
            <ul>${predictionDisplay}</ul>
        `;
        
    } catch (error) {
        alert('Error making predictions: ' + error.message);
        console.error('Prediction error:', error);
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
        let submissionCSV = 'PassengerId,Survived\n';
        const threshold = parseFloat(document.getElementById('threshold-slider').value);
        
        testPredictions.probabilities.forEach((prob, i) => {
            const survived = (!isNaN(prob) && prob >= threshold) ? 1 : 0;
            submissionCSV += `${testPredictions.identifiers[i]},${survived}\n`;
        });
        
        let probabilitiesCSV = 'PassengerId,Probability\n';
        testPredictions.probabilities.forEach((prob, i) => {
            probabilitiesCSV += `${testPredictions.identifiers[i]},${isNaN(prob) ? '0.0000' : prob.toFixed(4)}\n`;
        });
        
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

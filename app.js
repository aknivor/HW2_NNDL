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
    
    if (isTraining) {
        console.log('Calculating training statistics...');
        
        // Extract numeric values with proper parsing
        const ages = [];
        const fares = [];
        const embarked = [];
        
        data.forEach(row => {
            // Parse Age
            const age = parseFloat(row.Age);
            if (!isNaN(age)) ages.push(age);
            
            // Parse Fare
            const fare = parseFloat(row.Fare);
            if (!isNaN(fare)) fares.push(fare);
            
            // Parse Embarked
            if (row.Embarked && row.Embarked.trim() !== '') {
                embarked.push(row.Embarked);
            }
        });
        
        console.log(`Valid ages: ${ages.length}, fares: ${fares.length}, embarked: ${embarked.length}`);
        
        stats = {
            ageMedian: calculateMedian(ages),
            fareMean: calculateMean(fares),
            fareStd: calculateStd(fares),
            embarkedMode: calculateMode(embarked)
        };
        
        console.log('Training stats:', stats);
    }
    
    const features = [];
    const labels = [];
    const identifiers = [];
    let skippedRows = 0;
    
    data.forEach((row) => {
        try {
            // Parse with validation
            let age = parseFloat(row.Age);
            if (isNaN(age)) age = stats.ageMedian || 30;
            
            let fare = parseFloat(row.Fare);
            if (isNaN(fare)) fare = stats.fareMean || 0;
            
            const embarked = (row.Embarked && row.Embarked.trim() !== '') ? row.Embarked : (stats.embarkedMode || 'S');
            
            // Handle zero standard deviation
            const fareStd = stats.fareStd > 0 ? stats.fareStd : 1;
            const standardizedFare = (fare - stats.fareMean) / fareStd;
            
            // Create feature vector with validation
            const featureVector = [
                age,  // Use raw age instead of standardized
                standardizedFare,
                parseInt(row.SibSp) || 0,
                parseInt(row.Parch) || 0,
                row.Sex === 'male' ? 1 : 0,
                row.Sex === 'female' ? 1 : 0,
                row.Pclass === '1' ? 1 : 0,
                row.Pclass === '2' ? 1 : 0,
                row.Pclass === '3' ? 1 : 0,
                embarked === 'C' ? 1 : 0,
                embarked === 'Q' ? 1 : 0,
                embarked === 'S' ? 1 : 0
            ];
            
            // Validate feature vector
            const isValid = featureVector.every(val => {
                const numVal = Number(val);
                return !isNaN(numVal) && isFinite(numVal);
            });
            
            if (!isValid) {
                console.warn('Invalid feature vector:', featureVector);
                skippedRows++;
                return;
            }
            
            features.push(featureVector);
            
            if (isTraining && row.Survived !== undefined && row.Survived !== '') {
                const label = parseInt(row.Survived);
                if (!isNaN(label)) {
                    labels.push(label);
                }
            }
            
            identifiers.push(row.PassengerId);
            
        } catch (error) {
            console.error('Error processing row:', error);
            skippedRows++;
        }
    });
    
    console.log(`Processed ${features.length} valid rows, skipped ${skippedRows} rows`);
    
    if (features.length === 0) {
        throw new Error('No valid features found after preprocessing');
    }
    
    // Create tensors and validate
    const featuresTensor = tf.tensor2d(features);
    const labelsTensor = isTraining && labels.length > 0 ? tf.tensor1d(labels) : null;
    
    // Check for NaN in tensors
    const featuresHasNaN = tf.isNaN(featuresTensor).any().dataSync()[0];
    if (featuresHasNaN) {
        throw new Error('Features tensor contains NaN values');
    }
    
    console.log('Features tensor shape:', featuresTensor.shape);
    console.log('Features sample:', featuresTensor.arraySync().slice(0, 2));
    
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
                    units: 16,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid'
                })
            ]
        });
        
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
        
        // Calculate ROC
        const valPredictions = model.predict(xVal);
        const valProbs = valPredictions.dataSync();
        const valLabels = yVal.dataSync();
        
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
    
    auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].fpr - rocData[i-1].fpr) * 
               (rocData[i].tpr + rocData[i-1].tpr) / 2;
    }
    
    console.log(`AUC calculated: ${auc}`);
    
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
        console.log('No ROC data available');
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
        
        // Verify test features don't contain NaN
        const featuresHasNaN = tf.isNaN(testFeatures).any().dataSync()[0];
        if (featuresHasNaN) {
            throw new Error('Test features contain NaN values');
        }
        
        const predictions = model.predict(testFeatures);
        const probabilities = predictions.dataSync();
        
        // Check for NaN in predictions
        const validPredictions = probabilities.map(prob => isNaN(prob) ? 0 : prob);
        const nanCount = probabilities.filter(p => isNaN(p)).length;
        
        if (nanCount > 0) {
            console.warn(`Found ${nanCount} NaN predictions, replaced with 0`);
        }
        
        testPredictions = {
            identifiers: testData.processed.identifiers,
            probabilities: validPredictions
        };
        
        // Display first 5 predictions
        const predictionDisplay = validPredictions.slice(0, 5).map((prob, i) => 
            `<li>Passenger ${testPredictions.identifiers[i]}: ${prob.toFixed(3)}</li>`
        ).join('');
        
        document.getElementById('prediction-info').innerHTML = `
            <p><strong>Predictions generated for ${validPredictions.length} test samples</strong></p>
            ${nanCount > 0 ? `<p style="color: red;"><strong>Warning:</strong> ${nanCount} invalid predictions were replaced with 0</p>` : ''}
            <p>First 5 predictions:</p>
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
            const survived = prob >= threshold ? 1 : 0;
            submissionCSV += `${testPredictions.identifiers[i]},${survived}\n`;
        });
        
        let probabilitiesCSV = 'PassengerId,Probability\n';
        testPredictions.probabilities.forEach((prob, i) => {
            probabilitiesCSV += `${testPredictions.identifiers[i]},${prob.toFixed(4)}\n`;
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

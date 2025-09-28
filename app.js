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
    console.log('Starting data visualization...');
    
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
    
    console.log('Survival by Sex:', survivalBySex);
    console.log('Survival by Pclass:', survivalByPclass);
    
    // Create visualization data
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        x: sex,
        y: (stats.survived / stats.total) * 100
    }));
    
    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
        x: pclass,
        y: (stats.survived / stats.total) * 100
    }));
    
    console.log('Sex data for chart:', sexData);
    console.log('Pclass data for chart:', pclassData);
    
    // Create HTML charts as fallback since tfjs-vis might not be working
    createHTMLCharts(sexData, pclassData);
    
    // Try tfjs-vis with better error handling
    try {
        if (typeof tfvis !== 'undefined') {
            console.log('tfvis is available, attempting to render charts...');
            
            // Ensure visor is open
            const visor = tfvis.visor();
            if (!visor.isOpen()) {
                visor.open();
                console.log('Opened tfjs-vis visor');
            }
            
            // Render charts
            tfvis.render.barchart(
                { name: 'Survival Rate by Sex', tab: 'Data Inspection' },
                [{ values: sexData }],
                { 
                    xLabel: 'Sex', 
                    yLabel: 'Survival Rate (%)',
                    yAxisDomain: [0, 100]
                }
            );
            
            tfvis.render.barchart(
                { name: 'Survival Rate by Passenger Class', tab: 'Data Inspection' },
                [{ values: pclassData }],
                { 
                    xLabel: 'Passenger Class', 
                    yLabel: 'Survival Rate (%)',
                    yAxisDomain: [0, 100]
                }
            );
            
            console.log('tfjs-vis charts rendered successfully');
        } else {
            console.warn('tfvis is not available, using HTML charts only');
        }
    } catch (error) {
        console.error('Error with tfjs-vis charts:', error);
    }
}

function createHTMLCharts(sexData, pclassData) {
    const infoDiv = document.getElementById('data-info');
    
    let sexChartHTML = '<h3>Survival Rate by Sex (HTML Chart)</h3><div style="display: flex; align-items: flex-end; height: 200px; gap: 20px; margin: 20px 0;">';
    let pclassChartHTML = '<h3>Survival Rate by Passenger Class (HTML Chart)</h3><div style="display: flex; align-items: flex-end; height: 200px; gap: 20px; margin: 20px 0;">';
    
    // Create sex chart bars
    sexData.forEach(item => {
        const height = item.y;
        sexChartHTML += `
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="width: 50px; height: ${height}%; background: #4CAF50; display: flex; align-items: flex-end; justify-content: center;"></div>
                <div style="margin-top: 10px; font-weight: bold;">${item.x}</div>
                <div style="font-size: 12px;">${item.y.toFixed(1)}%</div>
            </div>
        `;
    });
    
    // Create pclass chart bars
    pclassData.forEach(item => {
        const height = item.y;
        pclassChartHTML += `
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="width: 50px; height: ${height}%; background: #2196F3; display: flex; align-items: flex-end; justify-content: center;"></div>
                <div style="margin-top: 10px; font-weight: bold;">${item.x}</div>
                <div style="font-size: 12px;">${item.y.toFixed(1)}%</div>
            </div>
        `;
    });
    
    sexChartHTML += '</div>';
    pclassChartHTML += '</div>';
    
    infoDiv.innerHTML += sexChartHTML + pclassChartHTML;
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
        
        const ages = [];
        const fares = [];
        const embarked = [];
        
        for (let i = 0; i < data.length; i++) {
            const row = data[i];
            
            const ageVal = parseFloat(row.Age);
            if (!isNaN(ageVal) && isFinite(ageVal)) {
                ages.push(ageVal);
            }
            
            const fareVal = parseFloat(row.Fare);
            if (!isNaN(fareVal) && isFinite(fareVal)) {
                fares.push(fareVal);
            }
            
            if (row.Embarked && row.Embarked.trim() !== '') {
                embarked.push(row.Embarked);
            }
        }
        
        console.log(`Valid data - Ages: ${ages.length}, Fares: ${fares.length}, Embarked: ${embarked.length}`);
        
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
    
    for (let i = 0; i < data.length; i++) {
        const row = data[i];
        
        try {
            let age = parseFloat(row.Age);
            if (isNaN(age) || !isFinite(age)) {
                age = stats.ageMedian;
            }
            
            let fare = parseFloat(row.Fare);
            if (isNaN(fare) || !isFinite(fare)) {
                fare = stats.fareMean;
            }
            
            let embarked = row.Embarked;
            if (!embarked || embarked.trim() === '') {
                embarked = stats.embarkedMode;
            }
            
            const sibsp = parseInt(row.SibSp) || 0;
            const parch = parseInt(row.Parch) || 0;
            
            // Use raw values without standardization to avoid NaN issues
            const featureVector = [
                age,
                fare,
                sibsp,
                parch,
                row.Sex === 'male' ? 1 : 0,
                row.Sex === 'female' ? 1 : 0,
                row.Pclass === '1' ? 1 : 0,
                row.Pclass === '2' ? 1 : 0,
                row.Pclass === '3' ? 1 : 0,
                embarked === 'C' ? 1 : 0,
                embarked === 'Q' ? 1 : 0,
                embarked === 'S' ? 1 : 0
            ];
            
            let isValid = true;
            for (let j = 0; j < featureVector.length; j++) {
                const val = featureVector[j];
                if (isNaN(val) || !isFinite(val)) {
                    isValid = false;
                    break;
                }
            }
            
            if (!isValid) {
                continue;
            }
            
            features.push(featureVector);
            
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
    
    console.log(`Successfully processed ${features.length} of ${data.length} rows`);
    
    if (features.length === 0) {
        throw new Error('No valid features could be processed');
    }
    
    const featuresTensor = tf.tensor2d(features);
    const labelsTensor = isTraining && labels.length > 0 ? tf.tensor1d(labels) : null;
    
    console.log('Features tensor shape:', featuresTensor.shape);
    
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
                    units: 8,
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
            epochs: 30,
            batchSize: 32,
            validationData: validationData,
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance', tab: 'Training' },
                ['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                { callbacks: ['onEpochEnd'] }
            )
        });
        
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
    
    // Create HTML ROC chart
    createHTMLROCChart();
    
    // Try tfjs-vis ROC chart
    try {
        if (typeof tfvis !== 'undefined') {
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
    } catch (error) {
        console.error('Error with tfjs-vis ROC chart:', error);
    }
}

function createHTMLROCChart() {
    const rocDiv = document.getElementById('roc-curve');
    
    if (!rocDiv) {
        console.error('ROC curve div not found');
        return;
    }
    
    let rocHTML = `
        <h3>ROC Curve (AUC = ${auc.toFixed(3)})</h3>
        <div style="border: 1px solid #ddd; padding: 20px; background: white; border-radius: 5px;">
            <p><strong>ROC Points:</strong></p>
            <div style="max-height: 200px; overflow-y: auto;">
                <table style="width: 100%; font-size: 12px;">
                    <tr>
                        <th>Threshold</th>
                        <th>FPR</th>
                        <th>TPR</th>
                    </tr>
    `;
    
    // Show every 10th point to avoid too much data
    for (let i = 0; i < rocData.length; i += 10) {
        const point = rocData[i];
        rocHTML += `
            <tr>
                <td>${point.threshold.toFixed(2)}</td>
                <td>${point.fpr.toFixed(3)}</td>
                <td>${point.tpr.toFixed(3)}</td>
            </tr>
        `;
    }
    
    rocHTML += `
                </table>
            </div>
            <p><em>Full ROC data available in console. AUC = ${auc.toFixed(3)}</em></p>
        </div>
    `;
    
    rocDiv.innerHTML = rocHTML;
}

// Metrics Update - COMPLETELY REWRITTEN
function updateMetrics() {
    console.log('Updating metrics...');
    
    if (!rocData || rocData.length === 0) {
        console.log('No ROC data available for metrics');
        document.getElementById('confusion-matrix').innerHTML = '<p>No metrics available yet. Please train the model first.</p>';
        document.getElementById('metrics-values').innerHTML = '';
        return;
    }
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    // Find the closest ROC point to the current threshold
    let rocPoint = rocData[0];
    for (let i = 1; i < rocData.length; i++) {
        if (Math.abs(rocData[i].threshold - threshold) < Math.abs(rocPoint.threshold - threshold)) {
            rocPoint = rocData[i];
        }
    }
    
    console.log('Selected ROC point:', rocPoint);
    
    const precision = rocPoint.tp / (rocPoint.tp + rocPoint.fp) || 0;
    const recall = rocPoint.tpr;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (rocPoint.tp + rocPoint.tn) / (rocPoint.tp + rocPoint.fp + rocPoint.tn + rocPoint.fn) || 0;
    
    // Update confusion matrix - FIXED DISPLAY
    const confusionMatrixHTML = `
        <h3>Confusion Matrix (Threshold: ${threshold.toFixed(2)})</h3>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 5px;">
            <table style="width: 100%; text-align: center; border-collapse: collapse; margin: 0 auto;">
                <tr>
                    <th style="border: 1px solid #ddd; padding: 12px; background: #e9ecef;"></th>
                    <th style="border: 1px solid #ddd; padding: 12px; background: #e9ecef;">Predicted Negative</th>
                    <th style="border: 1px solid #ddd; padding: 12px; background: #e9ecef;">Predicted Positive</th>
                </tr>
                <tr>
                    <th style="border: 1px solid #ddd; padding: 12px; background: #e9ecef;">Actual Negative</th>
                    <td style="border: 1px solid #ddd; padding: 12px; background: #d4edda;">${rocPoint.tn}</td>
                    <td style="border: 1px solid #ddd; padding: 12px; background: #f8d7da;">${rocPoint.fp}</td>
                </tr>
                <tr>
                    <th style="border: 1px solid #ddd; padding: 12px; background: #e9ecef;">Actual Positive</th>
                    <td style="border: 1px solid #ddd; padding: 12px; background: #f8d7da;">${rocPoint.fn}</td>
                    <td style="border: 1px solid #ddd; padding: 12px; background: #d4edda;">${rocPoint.tp}</td>
                </tr>
            </table>
        </div>
    `;
    
    // Update metrics values - FIXED DISPLAY
    const metricsHTML = `
        <h3>Performance Metrics</h3>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 20px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff;">
                    <strong>Accuracy</strong><br>
                    <span style="font-size: 24px; font-weight: bold; color: #007bff;">${accuracy.toFixed(3)}</span>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745;">
                    <strong>Precision</strong><br>
                    <span style="font-size: 24px; font-weight: bold; color: #28a745;">${precision.toFixed(3)}</span>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;">
                    <strong>Recall</strong><br>
                    <span style="font-size: 24px; font-weight: bold; color: #ffc107;">${recall.toFixed(3)}</span>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545;">
                    <strong>F1-Score</strong><br>
                    <span style="font-size: 24px; font-weight: bold; color: #dc3545;">${f1.toFixed(3)}</span>
                </div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 5px; margin-top: 15px; border-left: 4px solid #6f42c1;">
                <strong>AUC (Area Under ROC Curve)</strong><br>
                <span style="font-size: 24px; font-weight: bold; color: #6f42c1;">${auc.toFixed(3)}</span>
            </div>
        </div>
    `;
    
    // Update the DOM
    const confusionMatrixElement = document.getElementById('confusion-matrix');
    const metricsValuesElement = document.getElementById('metrics-values');
    
    if (confusionMatrixElement) {
        confusionMatrixElement.innerHTML = confusionMatrixHTML;
    } else {
        console.error('Confusion matrix element not found');
    }
    
    if (metricsValuesElement) {
        metricsValuesElement.innerHTML = metricsHTML;
    } else {
        console.error('Metrics values element not found');
    }
    
    console.log('Metrics updated successfully');
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
        
        const predictions = model.predict(testFeatures);
        const probabilities = predictions.dataSync();
        
        const nanCount = probabilities.filter(p => isNaN(p)).length;
        
        testPredictions = {
            identifiers: testData.processed.identifiers,
            probabilities: probabilities
        };
        
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

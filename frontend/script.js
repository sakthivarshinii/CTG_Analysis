// Theme Toggle
const themeBtn = document.getElementById('themeToggle');
themeBtn.addEventListener('click', () => {
    document.body.classList.toggle('light-theme');
    const isLight = document.body.classList.contains('light-theme');
    themeBtn.innerHTML = isLight ? '<i class="fa-solid fa-sun"></i>' : '<i class="fa-solid fa-moon"></i>';
    updateChartsTheme(isLight);
});

// UI Elements
const form = document.getElementById('ctgForm');
const statusDiv = document.getElementById('validationStatus');
const predStatus = document.getElementById('predStatus');
const predConf = document.getElementById('predConf');
const predAgree = document.getElementById('predAgree');
const riskAlert = document.getElementById('riskAlert');
const riskText = document.getElementById('riskText');
const humanExplanation = document.getElementById('humanExplanation');
const historyTable = document.querySelector('#historyTable tbody');

// Chat UI
const chatWindow = document.getElementById('chatWindow');
const chatInput = document.getElementById('chatInput');
const sendChat = document.getElementById('sendChat');

let currentPredictionLabel = null;
let currentFeatures = null;

// Charts
let shapChartInstance = null;
let ctgChartInstance = null;

const chartFontColor = () => document.body.classList.contains('light-theme') ? '#656d76' : '#7d8590';

function initCharts() {
    const shapCtx = document.getElementById('shapCanvas').getContext('2d');
    shapChartInstance = new Chart(shapCtx, {
        type: 'bar',
        data: {
            labels: ['LB', 'ASTV', 'AC', 'DL', 'UC'],
            datasets: [{
                label: 'SHAP Feature Impact',
                data: [0, 0, 0, 0, 0],
                backgroundColor: 'rgba(88, 166, 255, 0.5)',
                borderColor: '#58a6ff',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { ticks: { color: chartFontColor() }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: chartFontColor() }, grid: { display: false } }
            }
        }
    });

    const ctgCtx = document.getElementById('ctgCanvas').getContext('2d');
    ctgChartInstance = new Chart(ctgCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 20}, (_, i) => i),
            datasets: [
                {
                    label: 'Simulated FHR',
                    data: Array(20).fill(130),
                    borderColor: '#2ea043',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Contractions',
                    data: Array(20).fill(0),
                    borderColor: '#d29922',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 500,
                easing: 'linear'
            },
            scales: {
                x: { display: false },
                y: { type: 'linear', display: true, position: 'left', min: 100, max: 180, ticks: { color: chartFontColor() } },
                y1: { type: 'linear', display: true, position: 'right', min: 0, max: 0.02, ticks: { color: chartFontColor() }, grid: { drawOnChartArea: false } }
            }
        }
    });
}

function updateChartsTheme(light) {
    const col = light ? '#656d76' : '#7d8590';
    shapChartInstance.options.scales.x.ticks.color = col;
    shapChartInstance.options.scales.y.ticks.color = col;
    ctgChartInstance.options.scales.y.ticks.color = col;
    ctgChartInstance.options.scales.y1.ticks.color = col;
    shapChartInstance.update();
    ctgChartInstance.update();
}

// Data Validation
function validateInputs(data) {
    if (data.LB < 50 || data.LB > 250) return "FHR out of biological bounds (50-250).";
    if (data.ASTV < 0 || data.ASTV > 100) return "ASTV must be a percentage (0-100).";
    if (data.AC < 0 || data.DL < 0 || data.UC < 0) return "Rates cannot be negative.";
    return null;
}

// Fetch History
async function fetchHistory() {
    try {
        const res = await fetch('/api/history');
        if (!res.ok) return;
        const records = await res.json();
        
        historyTable.innerHTML = '';
        records.forEach(r => {
            let tr = document.createElement('tr');
            let badgeClass = 'badge-normal';
            if(r.prediction === 'Suspect') badgeClass = 'badge-suspect';
            if(r.prediction === 'Pathological') badgeClass = 'badge-pathological';
            
            tr.innerHTML = `
                <td>${r.patient_id}</td>
                <td><span class="badge ${badgeClass}">${r.prediction}</span></td>
                <td>${r.risk_level.split(' - ')[0]}</td>
            `;
            historyTable.appendChild(tr);
        });
    } catch(e) { console.error('History fetch failed', e); }
}

function updateCTGChart(features) {
    // Simulate FHR based on LB and Variability
    let simFHR = [];
    let simUC = [];
    let baseFHR = features.LB;
    let varMag = (features.ASTV / 100) * 15; // Rough estimate for simulation scale
    for(let i=0; i<20; i++) {
        simFHR.push(baseFHR + (Math.random() * varMag * 2 - varMag));
        simUC.push(features.UC * (Math.random() * 0.5 + 0.5)); // Add noise
    }
    ctgChartInstance.data.datasets[0].data = simFHR;
    ctgChartInstance.data.datasets[1].data = simUC;
    ctgChartInstance.update();
}

// Predict Form Submit
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const d = {
        patient_id: document.getElementById('patient_id').value,
        LB: parseFloat(document.getElementById('LB').value),
        ASTV: parseFloat(document.getElementById('ASTV').value),
        AC: parseFloat(document.getElementById('AC').value),
        DL: parseFloat(document.getElementById('DL').value),
        UC: parseFloat(document.getElementById('UC').value)
    };

    const valError = validateInputs(d);
    if(valError) {
        statusDiv.className = 'status-indicator status-warn';
        statusDiv.innerText = valError;
        return;
    }
    
    statusDiv.className = 'status-indicator status-good';
    statusDiv.innerText = "Data looks valid. Analyzing...";

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(d)
        });
        
        const data = await res.json();
        if(res.ok) {
            statusDiv.style.display = 'none';
            currentPredictionLabel = data.prediction;
            currentFeatures = d;
            
            // Update UI 
            predStatus.innerText = data.prediction;
            predConf.innerText = data.confidence;
            predAgree.innerText = data.agreement;
            riskText.innerText = data.risk_level;
            humanExplanation.innerText = data.explanation;

            // Colors
            predStatus.className = `text-${data.prediction.toLowerCase()}`;
            riskAlert.className = `risk-alert bg-${data.prediction.toLowerCase()}`;

            // SHAP XAI Chart Update
            const feats = ['LB', 'ASTV', 'AC', 'DL', 'UC'];
            const shapData = feats.map(f => data.top_features[f]);
            shapChartInstance.data.datasets[0].data = shapData;
            
            // Make pathlogical bars red
            shapChartInstance.data.datasets[0].backgroundColor = data.prediction === 'Pathological' ? 'rgba(248,81,73,0.5)' : 
                                                                 data.prediction === 'Suspect' ? 'rgba(210,153,34,0.5)' : 
                                                                 'rgba(46,160,67,0.5)';
            shapChartInstance.data.datasets[0].borderColor = data.prediction === 'Pathological' ? '#f85149' : 
                                                             data.prediction === 'Suspect' ? '#d29922' : '#2ea043';
            shapChartInstance.update();

            updateCTGChart(d);
            fetchHistory();
        } else {
            statusDiv.className = 'status-indicator status-warn';
            statusDiv.innerText = "API Error: " + data.detail;
        }
    } catch(err) {
        statusDiv.className = 'status-indicator status-warn';
        statusDiv.innerText = "Network error. Is the server running?";
    }
});

// Chat 
async function handleChat() {
    const msg = chatInput.value.trim();
    if(!msg) return;

    chatWindow.innerHTML += `<div class="chat-msg usr">${msg}</div>`;
    chatInput.value = '';
    chatWindow.scrollTop = chatWindow.scrollHeight;

    const reqData = {
        message: msg,
        prediction: currentPredictionLabel || "Unknown",
        features: currentFeatures || {}
    };

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(reqData)
        });
        const data = await res.json();
        chatWindow.innerHTML += `<div class="chat-msg ai">${data.response}</div>`;
        chatWindow.scrollTop = chatWindow.scrollHeight;
    } catch(e) {
        chatWindow.innerHTML += `<div class="chat-msg ai" style="color:#f85149">Failed to connect to assistant.</div>`;
    }
}

sendChat.addEventListener('click', handleChat);
chatInput.addEventListener('keypress', (e) => { if(e.key === 'Enter') handleChat(); });

// Init
window.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchHistory();
});

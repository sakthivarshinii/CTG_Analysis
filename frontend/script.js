document.addEventListener('DOMContentLoaded', () => {
    // Fetch metrics on load
    fetch('/metrics')
        .then(res => res.json())
        .then(data => {
            if (data.accuracy) {
                document.getElementById('accuracy-display').innerText = `Model Accuracy: ${(data.accuracy * 100).toFixed(2)}%`;
            }
            if (data.feature_importance) {
                renderChart(data.feature_importance);
            }
            if (data.confusion_matrix) {
                renderMatrix(data.confusion_matrix);
            }
        })
        .catch(err => console.error("Error fetching metrics", err));

    const form = document.getElementById('prediction-form');
    
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const predictBtn = document.getElementById('predict-btn');
        const loading = document.getElementById('loading');
        
        predictBtn.classList.add('hidden');
        loading.classList.remove('hidden');
        
        const payload = {
            "LB": document.getElementById('LB').value,
            "ASTV": document.getElementById('ASTV').value,
            "AC": document.getElementById('AC').value,
            "DL": document.getElementById('DL').value,
            "UC": document.getElementById('UC').value
        };
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            predictBtn.classList.remove('hidden');
            loading.classList.add('hidden');
            
            if(data.error) {
                alert("Error: " + data.error);
                return;
            }
            
            showResult(data);
        })
        .catch(err => {
            console.error(err);
            alert("Error connecting to server.");
            predictBtn.classList.remove('hidden');
            loading.classList.add('hidden');
        });
    });
});

function showResult(data) {
    document.getElementById('result-placeholder').classList.add('hidden');
    document.getElementById('result-content').classList.remove('hidden');
    
    const predValue = document.getElementById('pred-value');
    predValue.innerText = data.prediction;
    
    predValue.className = 'badge';
    
    if (data.prediction === 'Normal') predValue.classList.add('status-normal');
    else if (data.prediction === 'Suspect') predValue.classList.add('status-suspect');
    else if (data.prediction === 'Pathological') predValue.classList.add('status-pathological');
    
    document.getElementById('conf-value').innerText = data.confidence;
    document.getElementById('msg-value').innerText = data.message;
}

let featureChartInst = null;
function renderChart(featureImportance) {
    const ctx = document.getElementById('featureChart').getContext('2d');
    
    const labels = Object.keys(featureImportance);
    const data = Object.values(featureImportance);
    
    if(featureChartInst) {
        featureChartInst.destroy();
    }
    
    featureChartInst = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance',
                data: data,
                backgroundColor: 'rgba(88, 166, 255, 0.7)',
                borderColor: 'rgba(88, 166, 255, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#8b949e' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#8b949e' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function renderMatrix(matrix) {
    const grid = document.getElementById('matrix-grid');
    grid.innerHTML = '<div class="matrix-label">Columns: Predicted (Normal | Suspect | Pathological)</div>';
    
    // matrix is a 3x3 array (Actual rows vs Predicted columns)
    matrix.forEach((row, i) => {
        row.forEach((val, j) => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            
            // Highlight diagonal correctly vs incorrectly
            if (i === j) {
                // Correctly classified
                cell.style.backgroundColor = `rgba(46, 160, 67, ${Math.min(1, Math.max(0.2, val / 50))})`;
                cell.style.color = '#fff';
            } else if (val > 0) {
                // Misclassified
                cell.style.backgroundColor = `rgba(248, 81, 73, ${Math.min(1, Math.max(0.2, val / 10))})`;
                cell.style.color = '#fff';
            }
            cell.innerText = val;
            grid.appendChild(cell);
        });
    });
}

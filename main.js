'use strict';

const ageSlider   = document.getElementById('age');
const ageDisplay  = document.getElementById('ageDisplay');
const domainInput = document.getElementById('domain');
const domainPills = document.querySelectorAll('.domain-pill');

ageSlider.addEventListener('input', () => { ageDisplay.textContent = ageSlider.value; });

domainPills.forEach(pill => {
    pill.addEventListener('click', () => {
        domainPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        domainInput.value = pill.dataset.value;
    });
});

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab === 'rf' ? 'rfResults' : 'xgbResults').classList.add('active');
    });
});

function getInputs() {
    return {
        city  : document.getElementById('city').value,
        age   : parseInt(document.getElementById('age').value),
        gender: document.getElementById('gender').value,
        weapon: document.getElementById('weapon').value,
        domain: document.getElementById('domain').value,
    };
}

const RF_ACCURACY  = 91.25;
const XGB_ACCURACY = 94.18;

let yearlyBarChart   = null;
let monthlyLineChart = null;

const MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec'];
const ALL_YEARS = ['2020','2021','2022','2023','2024'];

function showLoading(title, msg) {
    document.getElementById('loadingTitle').textContent = title;
    document.getElementById('loadingMsg').textContent   = msg;
    document.getElementById('loadingOverlay').classList.remove('hidden');
}
function hideLoading() { document.getElementById('loadingOverlay').classList.add('hidden'); }
function showResults() { document.getElementById('resultsSection').classList.remove('hidden'); }

function activateTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.getElementById('rfResults').classList.toggle('active',  tab === 'rf');
    document.getElementById('xgbResults').classList.toggle('active', tab === 'xgb');
}

async function runAnalysis() {
    const inputs = getInputs();
    showLoading('Running Random Forest Analysis', 'Classifying crime patterns...');
    document.getElementById('btnAnalyse').disabled = true;
    try {
        const res  = await fetch('/api/analyse', {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify(inputs)
        });
        let text = await res.text();
        text = text.replace(/\bNaN\b/g, 'null');
        const json = JSON.parse(text);
        if (!json.success) throw new Error(json.error);
        hideLoading(); showResults(); activateTab('rf');
        renderRFResults(json.data, inputs);
    } catch (err) {
        hideLoading();
        alert('Error: ' + err.message);
    } finally {
        document.getElementById('btnAnalyse').disabled = false;
    }
}

async function runPrediction() {
    const inputs = getInputs();
    showLoading('Running XGBoost Prediction', 'Predicting crime domain probabilities...');
    document.getElementById('btnPredict').disabled = true;
    try {
        const res  = await fetch('/api/predict', {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify(inputs)
        });
        let text = await res.text();
        text = text.replace(/\bNaN\b/g, 'null');
        const json = JSON.parse(text);
        if (!json.success) throw new Error(json.error);
        hideLoading(); showResults(); activateTab('xgb');
        renderXGBResults(json.data, inputs);
    } catch (err) {
        hideLoading();
        alert('Error: ' + err.message);
    } finally {
        document.getElementById('btnPredict').disabled = false;
    }
}

function renderRFResults(data, inputs) {
    
    document.getElementById('rfAccuracyNum').textContent   = RF_ACCURACY + '%';
    document.getElementById('rfAccuracyLabel').textContent = 'Accuracy';

    const stats = data.stats || {};
    document.getElementById('rfStats').innerHTML = `
        <div class="stat-card">
            <div class="stat-card-label"><i class="fas fa-building-columns"></i> Total City Crimes</div>
            <div class="stat-card-value">${(stats.total_city_crimes||0).toLocaleString()}</div>
            <div class="stat-card-sub">${inputs.city}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label"><i class="fas fa-folder-closed"></i> Case Closed Rate</div>
            <div class="stat-card-value">${stats.case_closed_rate||0}%</div>
            <div class="stat-card-sub">In ${inputs.city}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label"><i class="fas fa-user-shield"></i> Avg Police Deployed</div>
            <div class="stat-card-value">${stats.avg_police_deployed||0}</div>
            <div class="stat-card-sub">Officers per case</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label"><i class="fas fa-layer-group"></i> Domain Crimes</div>
            <div class="stat-card-value">${(stats.total_domain_crimes||0).toLocaleString()}</div>
            <div class="stat-card-sub">${inputs.domain}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label"><i class="fas fa-chart-bar"></i> Matched Records</div>
            <div class="stat-card-value">${data.total_matched||0}</div>
            <div class="stat-card-sub">Shown in table</div>
        </div>`;

    document.getElementById('recordsCount').textContent = `${(data.records||[]).length} records shown`;
    const tbody = document.getElementById('crimeTableBody');
    if (!data.records || data.records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" class="no-data"><i class="fas fa-search"></i> No records found.</td></tr>`;
        return;
    }
    tbody.innerHTML = data.records.map(r => `
        <tr>
            <td>${r.report_number}</td>
            <td>${r.date_reported||'—'}</td>
            <td>${r.time_of_occurrence||'—'}</td>
            <td>${r.date_of_occurrence||r.date_reported||'—'}</td>
            <td>${r.crime_code}</td>
            <td><strong>${r.crime_description}</strong></td>
            <td>${r.police_deployed}</td>
            <td><span class="badge-closed ${r.case_closed==='Yes'?'badge-yes':'badge-no'}">${r.case_closed}</span></td>
            <td>${r.date_case_closed||'—'}</td>
        </tr>`).join('');
}


function renderXGBResults(data, inputs) {
    
    document.getElementById('xgbAccuracyNum').textContent   = XGB_ACCURACY + '%';
    document.getElementById('xgbAccuracyLabel').textContent = 'Accuracy';

    const cs = data.crime_summary || {};
    const domainColors = {
        'Violent Crime'   : '#ff4060',
        'Other Crime'     : '#3d8bff',
        'Fire Accident'   : '#ff6b35',
        'Traffic Fatality': '#b060ff',
    };
    const dColor = domainColors[cs.predicted_domain] || '#ff6b35';
    const genderLabel = inputs.gender==='M'?'Male':inputs.gender==='F'?'Female':'Other';

    const topCrimesHTML = (cs.top_crimes||[]).map((c, i) => `
        <div style="display:flex;align-items:center;gap:12px;padding:10px 0;
                    border-bottom:1px solid rgba(255,255,255,0.05);">
            <div style="width:28px;height:28px;border-radius:8px;
                        background:rgba(255,107,53,0.15);border:1px solid rgba(255,107,53,0.3);
                        display:flex;align-items:center;justify-content:center;
                        font-size:12px;font-weight:700;color:#ff6b35;flex-shrink:0;">${i+1}</div>
            <div style="font-size:13px;font-weight:600;color:var(--text-primary);">${c.crime}</div>
        </div>`).join('');

    document.getElementById('crimePredictionCard').innerHTML = `
        <div style="background:rgba(255,107,53,0.08);border-radius:14px;padding:20px;
                    margin-bottom:20px;border:1px solid ${dColor}44;text-align:center;">
            <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;
                        letter-spacing:1.5px;margin-bottom:8px;">Predicted Crime Domain</div>
            <div style="font-size:28px;font-weight:800;color:${dColor};margin-bottom:6px;">
                ${cs.predicted_domain||'—'}
            </div>
            <div style="font-size:13px;color:var(--text-secondary);">
                Based on: <strong style="color:var(--text-primary);">
                ${inputs.city} · ${inputs.domain} · Age ${inputs.age} · ${genderLabel} · ${inputs.weapon}
                </strong>
            </div>
        </div>
        <div style="font-size:12px;font-weight:700;color:var(--text-secondary);
                    margin-bottom:6px;text-transform:uppercase;letter-spacing:0.8px;">
            <i class="fas fa-list" style="color:#ff6b35;margin-right:6px;"></i>
            Predicted Crimes for Selected Filters
        </div>
        ${topCrimesHTML||'<div class="no-data">No crime data for selected filters</div>'}`;

    const yearlyData = data.yearly_counts || {};
    
    const yearValues = ALL_YEARS.map(y => yearlyData[y] !== undefined ? yearlyData[y] : 0);

    if (yearlyBarChart) yearlyBarChart.destroy();
    const ctx1 = document.getElementById('yearlyBarCanvas').getContext('2d');

    
    const maxVal = Math.max(...yearValues, 1);

    yearlyBarChart = new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: ALL_YEARS,
            datasets: [{
                label: 'Crimes',
                
                data: yearValues.map(v => v === 0 ? 0.3 : v),
                backgroundColor: 'rgba(255,107,53,0.75)',
                borderColor    : '#ff6b35',
                borderWidth    : 1.5,
                borderRadius   : 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        
                        label: ctx => {
                            const real = yearValues[ctx.dataIndex];
                            return ` ${real} crimes in ${ctx.label}`;
                        }
                    }
                },
                
                datalabels: false,
            },
            scales: {
                x: {
                    ticks: { color: '#8a9bbf', font: { size: 12, weight: '600' } },
                    grid:  { color: 'rgba(255,255,255,0.04)' },
                },
                y: {
                    ticks: {
                        color: '#8a9bbf', font: { size: 11 },
                        
                        callback: v => v < 1 ? '0' : Math.round(v)
                    },
                    grid:  { color: 'rgba(255,255,255,0.06)' },
                    beginAtZero: true,
                    min: 0,
                }
            },
            
            animation: {
                onComplete: function() {
                    const chart = this;
                    const ctx   = chart.ctx;
                    ctx.font    = 'bold 11px Inter';
                    ctx.fillStyle = '#e8edf5';
                    ctx.textAlign = 'center';
                    chart.data.datasets.forEach((dataset, i) => {
                        const meta = chart.getDatasetMeta(i);
                        meta.data.forEach((bar, index) => {
                            const real = yearValues[index];
                            ctx.fillText(real, bar.x, bar.y - 6);
                        });
                    });
                }
            }
        }
    });

    const monthlyData   = data.monthly_counts || {};
    const monthlyValues = MONTH_LABELS.map((_, i) => {
        const v = monthlyData[String(i+1)];
        return v !== undefined ? v : 0;
    });

    if (monthlyLineChart) monthlyLineChart.destroy();
    const ctx2 = document.getElementById('monthlyLineCanvas').getContext('2d');
    monthlyLineChart = new Chart(ctx2, {
        type: 'line',
        data: {
            labels: MONTH_LABELS,
            datasets: [{
                label: 'Crimes',
                data: monthlyValues,
                borderColor         : '#ff6b35',
                backgroundColor     : 'rgba(255,107,53,0.10)',
                borderWidth         : 2.5,
                pointBackgroundColor: '#ff6b35',
                pointBorderColor    : '#fff',
                pointBorderWidth    : 2,
                pointRadius         : 5,
                pointHoverRadius    : 7,
                fill                : true,
                tension             : 0.4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${ctx.parsed.y} crimes in ${ctx.label}`
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#8a9bbf', font: { size: 10 } },
                    grid:  { color: 'rgba(255,255,255,0.04)' },
                },
                y: {
                    ticks: { color: '#8a9bbf', font: { size: 11 } },
                    grid:  { color: 'rgba(255,255,255,0.06)' },
                    beginAtZero: true,
                    min: 0,
                }
            }
        }
    });
}

function checkModelStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(d => {
            const bar = document.getElementById('modelStatusBar');
            if (d.models_loaded) {
                bar.classList.add('ready');
                bar.innerHTML = '<div class="status-spinner"></div><span> Models ready — click Analyse or Predict!</span>';
                setTimeout(() => bar.classList.add('hidden'), 3000);
            } else {
                setTimeout(checkModelStatus, 1000);
            }
        })
        .catch(() => setTimeout(checkModelStatus, 2000));
}

checkModelStatus();
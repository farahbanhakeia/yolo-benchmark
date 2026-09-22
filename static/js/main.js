/**
 * YOLO Benchmark Comparator — Frontend JavaScript
 */

// ════════════════════════════════════════════════════════════
//   VARIABLES GLOBALES
// ════════════════════════════════════════════════════════════

let selectedFile = null;
let benchmarkResults = null;
let charts = {};

// ════════════════════════════════════════════════════════════
//   INITIALISATION
// ════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initUploadZone();
    initSlider();
    initBenchmarkButton();
});

// ════════════════════════════════════════════════════════════
//   UPLOAD ZONE
// ════════════════════════════════════════════════════════════

function initUploadZone() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('imageInput');
    const content = document.getElementById('uploadContent');
    const preview = document.getElementById('previewContainer');
    const removeBtn = document.getElementById('removeImage');

    // Clic sur la zone
    zone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            input.click();
        }
    });

    // Sélection fichier
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Supprimer image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        input.value = '';
        content.style.display = 'flex';
        preview.style.display = 'none';
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Veuillez sélectionner une image valide.');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('imagePreview').src = e.target.result;
        document.getElementById('uploadContent').style.display = 'none';
        document.getElementById('previewContainer').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// ════════════════════════════════════════════════════════════
//   SLIDER
// ════════════════════════════════════════════════════════════

function initSlider() {
    const slider = document.getElementById('confSlider');
    const display = document.getElementById('confValue');
    slider.addEventListener('input', () => {
        display.textContent = parseFloat(slider.value).toFixed(2);
    });
}

// ════════════════════════════════════════════════════════════
//   SÉLECTION MODÈLES
// ════════════════════════════════════════════════════════════

function selectAll() {
    document.querySelectorAll('#modelsGrid input[type="checkbox"]')
        .forEach(cb => cb.checked = true);
}

function deselectAll() {
    document.querySelectorAll('#modelsGrid input[type="checkbox"]')
        .forEach(cb => cb.checked = false);
}

function selectQuick() {
    deselectAll();
    const quick = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m'];
    quick.forEach(name => {
        const cb = document.querySelector(`input[value="${name}"]`);
        if (cb) cb.checked = true;
    });
}

function getSelectedModels() {
    const checked = document.querySelectorAll('#modelsGrid input[type="checkbox"]:checked');
    return Array.from(checked).map(cb => cb.value);
}

// ════════════════════════════════════════════════════════════
//   BENCHMARK
// ════════════════════════════════════════════════════════════

function initBenchmarkButton() {
    document.getElementById('startBenchmark').addEventListener('click', startBenchmark);
}

async function startBenchmark() {
    // Validation
    if (!selectedFile) {
        alert('⚠️ Veuillez sélectionner une image.');
        return;
    }

    const models = getSelectedModels();
    if (models.length === 0) {
        alert('⚠️ Veuillez sélectionner au moins un modèle.');
        return;
    }

    const confidence = document.getElementById('confSlider').value;

    // Préparer le formulaire
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('models_json', JSON.stringify(models));
    formData.append('confidence', confidence);
    // Ajouter aussi chaque modèle individuellement
    models.forEach(m => formData.append('models', m));

    // Afficher le loading
    showLoading(models);

    // Désactiver le bouton
    const btn = document.getElementById('startBenchmark');
    btn.disabled = true;

    try {
        const response = await fetch('/api/benchmark', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Erreur serveur');
        }

        if (data.success) {
            benchmarkResults = data;
            hideLoading();
            displayResults(data);
        } else {
            throw new Error(data.error || 'Erreur inconnue');
        }

    } catch (error) {
        hideLoading();
        alert(`❌ Erreur: ${error.message}`);
        console.error(error);
    } finally {
        btn.disabled = false;
    }
}

// ════════════════════════════════════════════════════════════
//   LOADING
// ════════════════════════════════════════════════════════════

function showLoading(models) {
    document.getElementById('resultsSection').style.display = 'none';
    const section = document.getElementById('loadingSection');
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth' });

    const totalModels = models.length;
    let current = 0;
    const text = document.getElementById('loadingText');
    const detail = document.getElementById('loadingDetail');
    const bar = document.getElementById('progressBar');

    function updateProgress() {
        if (current < totalModels) {
            text.textContent = `Benchmark en cours... (${current + 1}/${totalModels})`;
            detail.textContent = `Test du modèle: ${models[current]}`;
            bar.style.width = `${((current + 1) / totalModels) * 90}%`;
            current++;
            setTimeout(updateProgress, 3000 + Math.random() * 2000);
        }
    }
    updateProgress();
}

function hideLoading() {
    document.getElementById('loadingSection').style.display = 'none';
}

// ════════════════════════════════════════════════════════════
//   AFFICHAGE RÉSULTATS
// ════════════════════════════════════════════════════════════

function displayResults(data) {
    const section = document.getElementById('resultsSection');
    section.style.display = 'block';

    const results = data.results.filter(r => r.success);

    if (results.length === 0) {
        section.innerHTML = '<div class="card"><div class="card-body text-center"><h3>❌ Aucun modèle n\'a réussi le benchmark</h3></div></div>';
        section.scrollIntoView({ behavior: 'smooth' });
        return;
    }

    // 1. Summary cards
    buildSummary(results);

    // 2. Charts
    buildCharts(results);

    // 3. Table
    buildTable(results);

    // 4. Annotated images
    buildAnnotatedImages(data);

    section.scrollIntoView({ behavior: 'smooth' });
}

// ─── Summary ─────────────────────────────────────────────

function buildSummary(results) {
    const container = document.getElementById('summaryCards');

    // Fastest
    const fastest = results.reduce((a, b) => a.inference_time_ms < b.inference_time_ms ? a : b);
    // Most accurate (highest avg confidence)
    const accurate = results.reduce((a, b) => a.avg_confidence > b.avg_confidence ? a : b);
    // Most detections
    const most = results.reduce((a, b) => a.num_detections > b.num_detections ? a : b);
    // Best FPS
    const bestFps = results.reduce((a, b) => a.fps > b.fps ? a : b);

    container.innerHTML = `
        <div class="summary-card">
            <div class="icon color-fastest"><i class="fas fa-bolt"></i></div>
            <div class="label">Le plus rapide</div>
            <div class="value color-fastest">${fastest.model_name}</div>
            <div class="detail">${fastest.inference_time_ms} ms</div>
        </div>
        <div class="summary-card">
            <div class="icon color-accurate"><i class="fas fa-bullseye"></i></div>
            <div class="label">Plus haute confiance</div>
            <div class="value color-accurate">${accurate.model_name}</div>
            <div class="detail">${(accurate.avg_confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="summary-card">
            <div class="icon color-most"><i class="fas fa-search"></i></div>
            <div class="label">Plus de détections</div>
            <div class="value color-most">${most.model_name}</div>
            <div class="detail">${most.num_detections} objets</div>
        </div>
        <div class="summary-card">
            <div class="icon color-best"><i class="fas fa-tachometer-alt"></i></div>
            <div class="label">Meilleur FPS</div>
            <div class="value color-best">${bestFps.model_name}</div>
            <div class="detail">${bestFps.fps} FPS</div>
        </div>
    `;
}

// ─── Charts ──────────────────────────────────────────────

function buildCharts(results) {
    // Détruire les anciens graphiques
    Object.values(charts).forEach(c => { if (c) c.destroy(); });
    charts = {};

    const labels = results.map(r => r.model_name);
    const colors = generateColors(results.length);

    // Inference Time Chart
    charts.inference = new Chart(document.getElementById('inferenceChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Temps d\'inférence (ms)',
                data: results.map(r => r.inference_time_ms),
                backgroundColor: colors.map(c => c + '88'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: chartOptions('Temps (ms)')
    });

    // Confidence Chart
    charts.confidence = new Chart(document.getElementById('confidenceChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Confiance moyenne',
                data: results.map(r => (r.avg_confidence * 100).toFixed(1)),
                backgroundColor: colors.map(c => c + '88'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: chartOptions('Confiance (%)')
    });

    // FPS Chart
    charts.fps = new Chart(document.getElementById('fpsChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'FPS',
                data: results.map(r => r.fps),
                backgroundColor: colors.map(c => c + '88'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: chartOptions('Images/sec')
    });

    // Detections Chart
    charts.detections = new Chart(document.getElementById('detectionsChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Nombre de détections',
                data: results.map(r => r.num_detections),
                backgroundColor: colors.map(c => c + '88'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: chartOptions('Détections')
    });

    // Scatter: Speed vs Accuracy
    const scatterData = results.map((r, i) => ({
        x: r.inference_time_ms,
        y: r.avg_confidence * 100,
        label: r.model_name
    }));

    charts.scatter = new Chart(document.getElementById('scatterChart'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Modèles',
                data: scatterData,
                backgroundColor: colors.map(c => c + 'CC'),
                borderColor: colors,
                borderWidth: 2,
                pointRadius: 10,
                pointHoverRadius: 14
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const d = scatterData[ctx.dataIndex];
                            return `${d.label}: ${d.x}ms, ${d.y.toFixed(1)}%`;
                        }
                    },
                    backgroundColor: '#16213e',
                    borderColor: '#00d4ff',
                    borderWidth: 1,
                    titleColor: '#00d4ff',
                    bodyColor: '#e8e8e8'
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Temps d\'inférence (ms)', color: '#a0a0b0' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#a0a0b0' }
                },
                y: {
                    title: { display: true, text: 'Confiance moyenne (%)', color: '#a0a0b0' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#a0a0b0' }
                }
            }
        }
    });
}

function chartOptions(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#16213e',
                borderColor: '#00d4ff',
                borderWidth: 1,
                titleColor: '#00d4ff',
                bodyColor: '#e8e8e8'
            }
        },
        scales: {
            x: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#a0a0b0', maxRotation: 45 }
            },
            y: {
                title: { display: true, text: yLabel, color: '#a0a0b0' },
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#a0a0b0' },
                beginAtZero: true
            }
        }
    };
}

function generateColors(count) {
    const base = [
        '#00d4ff', '#00e676', '#ffc107', '#ff5252',
        '#e040fb', '#ff6e40', '#40c4ff', '#69f0ae',
        '#ffff00', '#ff4081', '#7c4dff', '#18ffff',
        '#b2ff59', '#ff9100', '#536dfe', '#f50057'
    ];
    const result = [];
    for (let i = 0; i < count; i++) {
        result.push(base[i % base.length]);
    }
    return result;
}

// ─── Table ───────────────────────────────────────────────

function buildTable(results) {
    const tbody = document.getElementById('resultsTableBody');

    // Trier par temps d'inférence
    const sorted = [...results].sort((a, b) => a.inference_time_ms - b.inference_time_ms);

    tbody.innerHTML = sorted.map((r, i) => {
        const rankClass = i < 3 ? `rank-${i + 1}` : '';
        const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : (i + 1);

        return `
            <tr>
                <td class="${rankClass}">${medal}</td>
                <td class="${rankClass}">${r.model_name}</td>
                <td>${r.model_size_mb} MB</td>
                <td>${r.load_time_ms} ms</td>
                <td><strong>${r.inference_time_ms} ms</strong></td>
                <td>${r.fps}</td>
                <td>${r.num_detections}</td>
                <td>${(r.avg_confidence * 100).toFixed(1)}%</td>
                <td>${(r.max_confidence * 100).toFixed(1)}%</td>
                <td><span class="badge badge-success">✓ OK</span></td>
            </tr>
        `;
    }).join('');

    // Ajouter les modèles en erreur
    const failed = benchmarkResults.results.filter(r => !r.success);
    failed.forEach(r => {
        tbody.innerHTML += `
            <tr style="opacity: 0.5;">
                <td>—</td>
                <td>${r.model_name}</td>
                <td colspan="7">${r.error || 'Erreur inconnue'}</td>
                <td><span class="badge badge-danger">✗ Erreur</span></td>
            </tr>
        `;
    });
}

// ─── Annotated Images ────────────────────────────────────

function buildAnnotatedImages(data) {
    const section = document.getElementById('annotatedSection');
    const grid = document.getElementById('annotatedGrid');

    const withImages = data.results.filter(r => r.success && r.annotated_image);

    if (withImages.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    grid.innerHTML = withImages.map(r => `
        <div class="annotated-item">
            <h4><i class="fas fa-robot"></i> ${r.model_name}
                — ${r.num_detections} détections</h4>
            <img src="/results/${r.annotated_image}"
                 alt="Détections ${r.model_name}"
                 loading="lazy">
        </div>
    `).join('');
}
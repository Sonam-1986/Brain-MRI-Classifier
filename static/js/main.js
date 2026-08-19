/* ==========================================================================
   MRI Brain Tumor Classifier — Interactive Frontend Logic
   ========================================================================== */

/* ─── State ─── */
const MAX_FILES = 4;
let selectedFiles = [];       // Array of File objects
let currentResults = [];      // Array of analysis results from backend

/* ─── DOM Elements ─── */
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewGrid = document.getElementById('previewGrid');
const analyzeSection = document.getElementById('analyzeSection');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearAllBtn = document.getElementById('clearAllBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingStatusText = document.getElementById('loadingStatusText');
const analysisProgressBar = document.getElementById('analysisProgressBar');
const resultsContainer = document.getElementById('resultsContainer');
const resultsCardsWrapper = document.getElementById('resultsCardsWrapper');
const newScanBtn = document.getElementById('newScanBtn');
const exportSummaryBtn = document.getElementById('exportSummaryBtn');

// Modal Elements
const imageModal = document.getElementById('imageModal');
const modalTitle = document.getElementById('modalTitle');
const modalSubtitle = document.getElementById('modalSubtitle');
const modalImg = document.getElementById('modalImg');

/* ─── Navigation Scroll Highlighting ─── */
window.addEventListener('scroll', () => {
  const sections = document.querySelectorAll('section[id]');
  const scrollY = window.pageYOffset;

  sections.forEach(sec => {
    const sectionHeight = sec.offsetHeight;
    const sectionTop = sec.offsetTop - 120;
    const sectionId = sec.getAttribute('id');
    const link = document.querySelector(`.nav-links a[href="#${sectionId}"]`);

    if (link) {
      if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
      }
    }
  });
});

/* ─── Drag & Drop Handlers ─── */
if (dropZone) {
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
  });

  dropZone.addEventListener('click', (e) => {
    if (e.target.closest('.btn-choose-file')) return;
    fileInput.click();
  });
}

if (fileInput) {
  fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
    fileInput.value = ''; // Reset for re-selection
  });
}

/* ─── File Ingestion & Queue ─── */
function handleFiles(files) {
  const allowed = ['image/jpeg', 'image/png', 'image/jpg'];
  const newFiles = Array.from(files);

  newFiles.forEach(f => {
    if (!allowed.includes(f.type)) {
      alert(`"${f.name}" is not a supported file format. Please upload JPG or PNG images.`);
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      alert(`"${f.name}" exceeds the 10MB size limit.`);
      return;
    }
    if (selectedFiles.length >= MAX_FILES) {
      alert(`Maximum of ${MAX_FILES} MRI scans can be analyzed simultaneously.`);
      return;
    }
    // Prevent duplicate files
    if (selectedFiles.some(existing => existing.name === f.name && existing.size === f.size)) {
      return;
    }
    selectedFiles.push(f);
  });

  renderPreviewQueue();
}

function renderPreviewQueue() {
  if (!previewGrid || !analyzeSection) return;

  if (selectedFiles.length === 0) {
    previewGrid.style.display = 'none';
    analyzeSection.style.display = 'none';
    previewGrid.innerHTML = '';
    return;
  }

  previewGrid.style.display = 'grid';
  analyzeSection.style.display = 'flex';
  previewGrid.innerHTML = '';

  selectedFiles.forEach((file, index) => {
    const card = document.createElement('div');
    card.className = 'preview-card';

    const url = URL.createObjectURL(file);
    const sizeKB = (file.size / 1024).toFixed(1);

    card.innerHTML = `
      <img src="${url}" alt="${file.name}" class="preview-thumb" />
      <div class="preview-meta">
        <span class="preview-name" title="${file.name}">${file.name}</span>
        <span class="preview-size">${sizeKB} KB</span>
      </div>
      <button type="button" class="btn-remove-file" onclick="removeFile(${index})" title="Remove file">
        <i class="fas fa-times"></i>
      </button>
    `;

    previewGrid.appendChild(card);
  });
}

window.removeFile = function(index) {
  selectedFiles.splice(index, 1);
  renderPreviewQueue();
};

if (clearAllBtn) {
  clearAllBtn.addEventListener('click', () => {
    selectedFiles = [];
    renderPreviewQueue();
  });
}

/* ─── Analyze Pipeline Execution ─── */
if (analyzeBtn) {
  analyzeBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
      alert('Please upload at least one MRI scan image first.');
      return;
    }

    // Prepare Form Data
    const formData = new FormData();
    selectedFiles.forEach(f => formData.append('files', f));

    // Show Loading Overlay
    loadingOverlay.style.display = 'block';
    resultsContainer.style.display = 'none';
    analyzeBtn.disabled = true;

    // Simulate animated progress steps
    const steps = [
      { pct: 20, text: 'Stage 1: Grayscale conversion, CLAHE & Denoising...' },
      { pct: 45, text: 'Stage 2: Otsu optimal skull stripping & Brain mask...' },
      { pct: 65, text: 'Stage 3: K-Means 5-cluster tissue separation...' },
      { pct: 85, text: 'Stage 4: Deep Residual CNN inference & CAM localization...' },
      { pct: 98, text: 'Stage 5: Multi-layer tumor overlay & Biomarker computation...' }
    ];

    let stepIdx = 0;
    const progressInterval = setInterval(() => {
      if (stepIdx < steps.length) {
        analysisProgressBar.style.width = steps[stepIdx].pct + '%';
        loadingStatusText.textContent = steps[stepIdx].text;
        stepIdx++;
      }
    }, 400);

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData
      });

      clearInterval(progressInterval);
      analysisProgressBar.style.width = '100%';

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Server error occurred during analysis.');
      }

      const data = await response.json();
      currentResults = data.results || [];

      setTimeout(() => {
        loadingOverlay.style.display = 'none';
        analyzeBtn.disabled = false;
        renderAnalysisResults(currentResults);
      }, 500);

    } catch (err) {
      clearInterval(progressInterval);
      loadingOverlay.style.display = 'none';
      analyzeBtn.disabled = false;
      alert(`Analysis Failed: ${err.message}`);
    }
  });
}

/* ─── Render Results Cards ─── */
function renderAnalysisResults(results) {
  if (!resultsCardsWrapper || !resultsContainer) return;

  resultsCardsWrapper.innerHTML = '';
  resultsContainer.style.display = 'block';

  results.forEach((item, idx) => {
    if (!item.success) {
      const errCard = document.createElement('div');
      errCard.className = 'result-scan-card';
      errCard.innerHTML = `
        <div class="scan-card-header">
          <span class="scan-filename">${item.filename || 'Scan ' + (idx + 1)}</span>
          <span class="badge-tag" style="color:#ef4444;">Failed</span>
        </div>
        <p style="color:#ef4444; padding:16px 0;"><i class="fas fa-exclamation-triangle"></i> ${item.error || 'Unable to process this image.'}</p>
      `;
      resultsCardsWrapper.appendChild(errCard);
      return;
    }

    const card = document.createElement('div');
    card.className = 'result-scan-card';

    const isTumor = item.is_tumor;
    const verdictClass = isTumor ? 'tumor' : 'normal';
    const verdictIcon = isTumor ? 'fa-triangle-exclamation' : 'fa-circle-check';
    const verdictTitle = isTumor ? 'Tumor Detected' : 'Normal — No Tumor Detected';
    const verdictDesc = isTumor
      ? `AI Residual CNN and CAM localized suspicious lesion with ${item.confidence}% confidence.`
      : `Bilateral cerebral symmetry preserved. No abnormal mass or vasogenic edema detected (${item.confidence}% confidence).`;

    const metrics = item.metrics || {};
    const factors = item.diagnostic_factors || [];
    const stages = item.stages || {};

    // Build 5-stage preview thumbnails
    const stageNames = [
      { key: 'preprocessed', label: '1. Preprocessing', sub: 'Grayscale + CLAHE' },
      { key: 'brain_mask', label: '2. Brain Extraction', sub: 'Otsu Skull Stripped' },
      { key: 'tissue_segmentation', label: '3. Tissue Separation', sub: 'K-Means 5-Classes' },
      { key: 'deep_learning_cam', label: '4. Deep Learning CAM', sub: 'ResNet Activation Map' },
      { key: 'final_overlay', label: '5. Color Overlay', sub: 'Tumor & Edema Map' }
    ];

    let stagesHTML = '';
    stageNames.forEach(st => {
      const b64 = stages[st.key];
      if (b64) {
        stagesHTML += `
          <div class="stage-card" onclick="openImageModal('${st.label}', '${st.sub}', '${b64}')">
            <div class="stage-header-meta">
              <span class="stage-num-pill">${st.label.split('.')[0]}</span>
              <span class="stage-name">${st.label.split('. ')[1]}</span>
            </div>
            <img src="${b64}" alt="${st.label}" class="stage-img-thumb" />
            <div class="stage-card-footer">
              <i class="fas fa-magnifying-glass-plus"></i>
              <span>${st.sub}</span>
            </div>
          </div>
        `;
      }
    });

    // Build Quantitative Metrics Table
    const metricsHTML = `
      <table class="metrics-table">
        <tbody>
          <tr>
            <td class="m-label">Classification Verdict</td>
            <td class="m-val" style="color:${isTumor ? '#f87171' : '#34d399'}; font-weight:700;">${item.label}</td>
          </tr>
          <tr>
            <td class="m-label">Model Confidence</td>
            <td class="m-val">${item.confidence}%</td>
          </tr>
          <tr>
            <td class="m-label">Brain Parenchyma Area</td>
            <td class="m-val">${(metrics.brain_area_px || 0).toLocaleString()} px</td>
          </tr>
          <tr>
            <td class="m-label">Suspicious Lesion Area</td>
            <td class="m-val">${(metrics.tumor_area_px || 0).toLocaleString()} px</td>
          </tr>
          <tr>
            <td class="m-label">Lesion / Brain Ratio</td>
            <td class="m-val">${metrics.area_ratio_pct || 0}%</td>
          </tr>
          <tr>
            <td class="m-label">Hemispheric Asymmetry Index</td>
            <td class="m-val">${metrics.asymmetry_index_pct || 0}%</td>
          </tr>
          <tr>
            <td class="m-label">Contrast Ratio</td>
            <td class="m-val">${metrics.contrast_ratio || 0}</td>
          </tr>
          <tr>
            <td class="m-label">Ring Enhancement Score</td>
            <td class="m-val">${metrics.ring_score_pct || 0}%</td>
          </tr>
        </tbody>
      </table>
    `;

    // Build Diagnostic Factors
    let factorsHTML = '';
    factors.forEach(f => {
      factorsHTML += `
        <div class="factor-item">
          <i class="fas fa-arrow-right"></i>
          <span>${f}</span>
        </div>
      `;
    });

    card.innerHTML = `
      <div class="scan-card-header">
        <div class="scan-title-group">
          <span class="scan-index-badge">Scan #${idx + 1}</span>
          <span class="scan-filename">${item.filename}</span>
        </div>
        <span class="badge-tag"><i class="fas fa-microchip"></i> PyTorch ResNet-CNN</span>
      </div>

      <!-- Verdict Banner -->
      <div class="verdict-banner ${verdictClass}">
        <div class="verdict-left">
          <div class="verdict-icon">
            <i class="fas ${verdictIcon}"></i>
          </div>
          <div>
            <div class="verdict-title">${verdictTitle}</div>
            <div class="verdict-desc">${verdictDesc}</div>
          </div>
        </div>
        <div class="verdict-confidence-badge">
          <div class="conf-value">${item.confidence}%</div>
          <div class="conf-label">Confidence</div>
        </div>
      </div>

      <!-- 5-Stage Segmentation Pipeline Thumbnails -->
      <h4 style="font-size:0.92rem; font-weight:700; color:#fff; margin-bottom:12px;">
        <i class="fas fa-layer-group" style="color:var(--purple-light); margin-right:6px;"></i> 5-Stage Segmentation Pipeline
      </h4>
      <div class="stages-view-grid">
        ${stagesHTML}
      </div>

      <!-- Biomarkers & Diagnostic Insights Grid -->
      <div class="biomarkers-grid">
        <div class="biomarkers-card">
          <h4><i class="fas fa-chart-column"></i> Quantitative Biomarkers</h4>
          ${metricsHTML}
        </div>
        <div class="biomarkers-card">
          <h4><i class="fas fa-stethoscope"></i> Clinical Insights &amp; Findings</h4>
          <div class="factors-list">
            ${factorsHTML || '<p style="color:var(--text-gray-400); font-size:0.82rem;">No abnormal pathological factors detected.</p>'}
          </div>
        </div>
      </div>
    `;

    resultsCardsWrapper.appendChild(card);
  });

  // Smooth scroll to results
  resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

/* ─── Modal Inspection ─── */
window.openImageModal = function(title, sub, src) {
  if (!imageModal) return;
  modalTitle.textContent = title;
  modalSubtitle.textContent = sub;
  modalImg.src = src;
  imageModal.style.display = 'flex';
};

window.closeImageModal = function() {
  if (!imageModal) return;
  imageModal.style.display = 'none';
  modalImg.src = '';
};

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeImageModal();
});

/* ─── Results Action Buttons ─── */
if (newScanBtn) {
  newScanBtn.addEventListener('click', () => {
    selectedFiles = [];
    renderPreviewQueue();
    resultsContainer.style.display = 'none';
    const workspace = document.getElementById('analyze-workspace');
    if (workspace) workspace.scrollIntoView({ behavior: 'smooth' });
  });
}

if (exportSummaryBtn) {
  exportSummaryBtn.addEventListener('click', () => {
    if (currentResults.length === 0) {
      alert('No results available to export.');
      return;
    }

    const reportData = {
      timestamp: new Date().toISOString(),
      system: 'MRI Brain Tumor Classifier & 5-Stage Segmentation',
      total_scans_analyzed: currentResults.length,
      scans: currentResults.map(r => ({
        filename: r.filename,
        verdict: r.label,
        is_tumor: r.is_tumor,
        confidence_pct: r.confidence,
        metrics: r.metrics,
        findings: r.diagnostic_factors
      }))
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `MRI_Brain_Analysis_Report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

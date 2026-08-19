/* ==========================================================================
   MRI Brain Tumor Classifier — Multi-View SPA & Diagnostic Q&A Logic
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
const resultsCardsWrapper = document.getElementById('resultsCardsWrapper');
const navResultsBtn = document.getElementById('navResultsBtn');
const exportSummaryBtn = document.getElementById('exportSummaryBtn');

// Modal Elements
const imageModal = document.getElementById('imageModal');
const modalTitle = document.getElementById('modalTitle');
const modalSubtitle = document.getElementById('modalSubtitle');
const modalImg = document.getElementById('modalImg');

/* ─── Multi-View Page Navigation Router ─── */
window.navigateTo = function(pageId) {
  // Hide all page views
  const pages = document.querySelectorAll('.page-view');
  pages.forEach(p => {
    p.classList.remove('active');
    p.style.display = 'none';
  });

  // Activate target page
  const targetPage = document.getElementById(pageId);
  if (targetPage) {
    targetPage.style.display = 'block';
    // Trigger transition
    setTimeout(() => {
      targetPage.classList.add('active');
    }, 20);
  }

  // Update Nav Links
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.dataset.page === pageId) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // Scroll to top of the new page cleanly
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Update URL hash
  if (history.pushState) {
    history.pushState(null, null, `#${pageId}`);
  } else {
    location.hash = `#${pageId}`;
  }
};

// Handle Browser Back/Forward buttons and direct hash URLs
window.addEventListener('popstate', () => {
  const hash = location.hash.replace('#', '');
  if (hash && document.getElementById(hash)) {
    navigateTo(hash);
  } else {
    navigateTo('page-home');
  }
});

// Initial load routing
document.addEventListener('DOMContentLoaded', () => {
  const hash = location.hash.replace('#', '');
  if (hash && document.getElementById(hash)) {
    navigateTo(hash);
  } else {
    navigateTo('page-home');
  }
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
    }, 380);

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

        // Render Results & Transition to Results Page View!
        renderAnalysisResults(currentResults);
        if (navResultsBtn) navResultsBtn.style.display = 'inline-flex';
        navigateTo('page-results');
      }, 500);

    } catch (err) {
      clearInterval(progressInterval);
      loadingOverlay.style.display = 'none';
      analyzeBtn.disabled = false;
      alert(`Analysis Failed: ${err.message}`);
    }
  });
}

/* ─── Render Results Cards with Medical Answers to Required Data ─── */
function renderAnalysisResults(results) {
  if (!resultsCardsWrapper) return;

  resultsCardsWrapper.innerHTML = '';

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
      ? `Deep Residual CNN and Class Activation Maps (CAM) identified focal neoplastic tissue (${item.confidence}% confidence).`
      : `Bilateral cerebral symmetry preserved. No abnormal mass or vasogenic edema detected (${item.confidence}% confidence).`;

    const metrics = item.metrics || item.stats || {};
    const factors = item.diagnostic_factors || item.factors || [];
    const stages = item.stages || {};
    const cqa = item.clinical_answers || {};

    // 5-Stage Segmentation Thumbnails
    const stageNames = [
      { key: 'preprocessed', label: '1. Preprocessing', sub: 'Grayscale + CLAHE' },
      { key: 'brain_mask', label: '2. Brain Extraction', sub: 'Otsu Skull Stripped' },
      { key: 'tissue_segmentation', label: '3. Tissue Separation', sub: 'K-Means 5-Classes' },
      { key: 'deep_learning_cam', label: '4. Deep Learning CAM', sub: 'ResNet Feature Map' },
      { key: 'final_overlay', label: '5. Color Overlay', sub: 'Tumor & Edema Map' }
    ];

    let stagesHTML = '';
    stageNames.forEach(st => {
      const b64 = stages[st.key] || (item.pipeline_steps && item.pipeline_steps.find(p => p.stage === st.label.split('.')[0])?.image);
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

    // Clinical Q&A / Answers to Required Data
    const cqaHTML = `
      <div class="clinical-qa-panel">
        <div class="cqa-header">
          <div class="cqa-title-wrap">
            <i class="fas fa-clipboard-check"></i>
            <h3>Required Diagnostic Answers &amp; Findings</h3>
          </div>
          <span class="cqa-badge">${isTumor ? 'Pathology Positive' : 'Normal Tissue Baseline'}</span>
        </div>

        <div class="cqa-grid">
          <div class="cqa-item">
            <div class="cqa-q"><i class="fas fa-stethoscope"></i> Pathological Classification &amp; Status</div>
            <div class="cqa-a ${isTumor ? 'highlight-red' : 'highlight-green'}">
              ${cqa.pathology_verdict || (isTumor ? 'Abnormal Intracranial Mass Detected' : 'Healthy Normal Brain Parenchyma')}
            </div>
          </div>

          <div class="cqa-item">
            <div class="cqa-q"><i class="fas fa-location-crosshairs"></i> Anatomical Lesion Location</div>
            <div class="cqa-a">
              ${cqa.lesion_location || metrics.location || (isTumor ? 'Unilateral Hemispheric Focal Lesion' : 'Bilateral Symmetrical (Normal)')}
            </div>
          </div>

          <div class="cqa-item">
            <div class="cqa-q"><i class="fas fa-ruler-combined"></i> Lesion Size &amp; Intracranial Volume</div>
            <div class="cqa-a">
              ${cqa.lesion_size_and_volume || (isTumor ? `${(metrics.tumor_area_px || metrics.tumor_area || 0).toLocaleString()} px (${metrics.area_ratio_pct || 0}% brain area)` : '0 px (0.0% — No focal lesion detected)')}
            </div>
          </div>

          <div class="cqa-item">
            <div class="cqa-q"><i class="fas fa-droplet"></i> Peritumoral Vasogenic Edema Halo</div>
            <div class="cqa-a ${isTumor && (metrics.edema_ratio_pct > 0) ? 'highlight-red' : ''}">
              ${cqa.vasogenic_edema_halo || (isTumor && (metrics.edema_ratio_pct > 0) ? `Present — Edema halo (${metrics.edema_ratio_pct}%)` : 'Absent — No vasogenic swelling')}
            </div>
          </div>

          <div class="cqa-item">
            <div class="cqa-q"><i class="fas fa-circle-dot"></i> Necrotic Center &amp; Contrast Enhancement Ring</div>
            <div class="cqa-a">
              ${cqa.contrast_ring_enhancement || `Ring Score: ${metrics.ring_score_pct || 0}%`} | Core: ${cqa.necrotic_center_core || 'Normal density'}
            </div>
          </div>

          <div class="cqa-item">
            <div class="cqa-q"><i class="fas fa-scale-balanced"></i> Hemispheric Mass Effect &amp; Asymmetry</div>
            <div class="cqa-a">
              ${cqa.hemispheric_asymmetry || `Asymmetry Index: ${metrics.asymmetry_index_pct || metrics.asymmetry_pct || 0}%`}
            </div>
          </div>

          <div class="cqa-item full-span">
            <div class="cqa-q"><i class="fas fa-user-doctor"></i> Clinical Guidance &amp; Recommended Next Steps</div>
            <div class="cqa-a" style="color:var(--text-gray-100);">
              ${cqa.clinical_recommendation || (isTumor ? 'Immediate neuro-oncological evaluation, contrast-enhanced T1/FLAIR MRI, and MR spectroscopy.' : 'No acute intracranial pathology. Maintain routine clinical screening.')}
            </div>
          </div>
        </div>
      </div>
    `;

    // Quantitative Biomarkers Table
    const metricsHTML = `
      <table class="metrics-table">
        <tbody>
          <tr>
            <td class="m-label">Diagnostic Classification</td>
            <td class="m-val" style="color:${isTumor ? '#f87171' : '#34d399'}; font-weight:700;">${item.label}</td>
          </tr>
          <tr>
            <td class="m-label">Neural Network Confidence</td>
            <td class="m-val">${item.confidence}%</td>
          </tr>
          <tr>
            <td class="m-label">Brain Parenchyma Area</td>
            <td class="m-val">${(metrics.brain_area_px || metrics.brain_area || 0).toLocaleString()} px</td>
          </tr>
          <tr>
            <td class="m-label">Suspicious Lesion Area</td>
            <td class="m-val">${(metrics.tumor_area_px || metrics.tumor_area || 0).toLocaleString()} px</td>
          </tr>
          <tr>
            <td class="m-label">Lesion / Brain Volume Ratio</td>
            <td class="m-val">${metrics.area_ratio_pct || 0}%</td>
          </tr>
          <tr>
            <td class="m-label">Hemispheric Asymmetry Index</td>
            <td class="m-val">${metrics.asymmetry_index_pct || metrics.asymmetry_pct || 0}%</td>
          </tr>
          <tr>
            <td class="m-label">Contrast Ratio</td>
            <td class="m-val">${metrics.contrast_ratio || 1.0}</td>
          </tr>
          <tr>
            <td class="m-label">Ring Enhancement Score</td>
            <td class="m-val">${metrics.ring_score_pct || 0}%</td>
          </tr>
        </tbody>
      </table>
    `;

    // Diagnostic Factors List
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

      <!-- Required Answers to Clinical Data Panel -->
      ${cqaHTML}

      <!-- 5-Stage Segmentation Pipeline Thumbnails -->
      <div class="section-subheading">
        <i class="fas fa-layer-group"></i> 5-Stage Visual Image Segmentation Pipeline
      </div>
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
          <h4><i class="fas fa-microscope"></i> Pathology Factors &amp; Observations</h4>
          <div class="factors-list">
            ${factorsHTML || '<p style="color:var(--text-gray-400); font-size:0.82rem;">No abnormal pathological factors detected.</p>'}
          </div>
        </div>
      </div>
    `;

    resultsCardsWrapper.appendChild(card);
  });
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

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeImageModal();
});

/* ─── Export Report ─── */
if (exportSummaryBtn) {
  exportSummaryBtn.addEventListener('click', () => {
    if (currentResults.length === 0) {
      alert('No analysis results available to export.');
      return;
    }

    const reportData = {
      report_timestamp: new Date().toISOString(),
      clinical_system: 'MRI Brain Tumor Classifier & 5-Stage Segmentation Suite',
      model_architecture: 'Deep Residual CNN (ResBlock x 4) + CAM + Morphological OpenCV',
      total_scans_analyzed: currentResults.length,
      scans: currentResults.map((r, i) => ({
        scan_number: i + 1,
        filename: r.filename,
        classification_verdict: r.label,
        is_tumor_detected: r.is_tumor,
        model_confidence_pct: r.confidence,
        clinical_answers: r.clinical_answers,
        quantitative_biomarkers: r.metrics || r.stats,
        pathology_observations: r.diagnostic_factors || r.factors
      }))
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Clinical_MRI_Analysis_Report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

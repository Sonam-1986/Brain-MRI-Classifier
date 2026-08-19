/* =========================================================
   MRI Brain Analyzer — Frontend Logic
========================================================= */

/* ── State ─────────────────────────────────────────── */
const MAX_FILES = 4;
let selectedFiles = [];   // File objects
let analysisResults = []; // Latest analysis JSON

/* ── DOM refs ──────────────────────────────────────── */
const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const previewGrid   = document.getElementById('previewGrid');
const analyzeSection= document.getElementById('analyzeSection');
const analyzeBtn    = document.getElementById('analyzeBtn');
const loadingOverlay= document.getElementById('loadingOverlay');
const resultsContainer = document.getElementById('resultsContainer');
const backBtn       = document.getElementById('backBtn');

/* ── Navigation ─────────────────────────────────────── */
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    const target = item.dataset.section;
    switchSection(target);
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    item.classList.add('active');
  });
});

function switchSection(name) {
  // "upload" and "dashboard" share the same view
  const mapped = name === 'upload' ? 'dashboard' : name;
  document.querySelectorAll('.section').forEach(s => {
    s.style.display = 'none';
    s.classList.remove('active');
  });
  const el = document.getElementById('section-' + mapped);
  if (el) { el.style.display = 'block'; el.classList.add('active'); }
}

backBtn && backBtn.addEventListener('click', () => {
  switchSection('dashboard');
  setNavActive('dashboard');
});

function setNavActive(name) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const nav = document.querySelector(`.nav-item[data-section="${name}"]`);
  if (nav) nav.classList.add('active');
}

/* ── Drag & Drop ────────────────────────────────────── */
dropZone.addEventListener('click', e => {
  if (e.target.closest('.btn-choose')) return;
  fileInput.click();
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles(Array.from(e.dataTransfer.files));
});

fileInput.addEventListener('change', () => {
  handleFiles(Array.from(fileInput.files));
  fileInput.value = '';
});

/* ── File Handling ──────────────────────────────────── */
function handleFiles(newFiles) {
  const allowed = ['image/jpeg', 'image/png', 'image/jpg'];
  newFiles.forEach(f => {
    if (!allowed.includes(f.type)) {
      alert(`"${f.name}" is not a supported file type (JPG/PNG only).`);
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      alert(`"${f.name}" exceeds the 10MB size limit.`);
      return;
    }
    if (selectedFiles.length >= MAX_FILES) {
      alert('Maximum 4 files allowed.');
      return;
    }
    if (selectedFiles.some(ex => ex.name === f.name && ex.size === f.size)) return;
    selectedFiles.push(f);
  });
  renderPreviews();
}

function renderPreviews() {
  if (selectedFiles.length === 0) {
    previewGrid.style.display = 'none';
    analyzeSection.style.display = 'none';
    return;
  }
  previewGrid.style.display = 'grid';
  analyzeSection.style.display = 'block';
  previewGrid.innerHTML = '';

  selectedFiles.forEach((file, idx) => {
    const reader = new FileReader();
    reader.onload = ev => {
      const item = document.createElement('div');
      item.className = 'preview-item';
      item.innerHTML = `
        <img src="${ev.target.result}" alt="${file.name}" />
        <div class="preview-label" title="${file.name}">${file.name}</div>
        <button class="preview-remove" data-idx="${idx}" title="Remove">
          <i class="fas fa-times"></i>
        </button>`;
      previewGrid.appendChild(item);

      item.querySelector('.preview-remove').addEventListener('click', e => {
        e.stopPropagation();
        removeFile(parseInt(e.currentTarget.dataset.idx));
      });
    };
    reader.readAsDataURL(file);
  });
}

function removeFile(idx) {
  selectedFiles.splice(idx, 1);
  renderPreviews();
}

/* ── Analyze ────────────────────────────────────────── */
analyzeBtn && analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (selectedFiles.length === 0) return;

  analyzeBtn.disabled = true;
  loadingOverlay.style.display = 'flex';

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append('files', f));

  try {
    const resp = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      alert(data.error || 'Server error. Please try again.');
      return;
    }
    analysisResults = data.results || [];
    renderResults(analysisResults, selectedFiles);
    switchSection('results');
    setNavActive('results');
  } catch (err) {
    alert('Network error: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
    loadingOverlay.style.display = 'none';
  }
}

/* ── Render Results ─────────────────────────────────── */
function renderResults(results, files) {
  resultsContainer.innerHTML = '';

  results.forEach((res, i) => {
    const card = document.createElement('div');
    card.className = 'result-card ' + (res.is_tumor ? 'tumor' : 'normal');

    if (!res.success) {
      card.innerHTML = `
        <div class="rc-header">
          <span class="rc-filename"><i class="fas fa-file-image"></i>${res.filename || files[i]?.name}</span>
          <span class="badge badge-tumor"><i class="fas fa-exclamation"></i> Error</span>
        </div>
        <div style="padding:18px 22px;color:#b91c1c;font-size:.88rem;">${res.error}</div>`;
      resultsContainer.appendChild(card);
      return;
    }

    const isTumor    = res.is_tumor;
    const badgeCls   = isTumor ? 'badge-tumor' : 'badge-normal';
    const badgeIcon  = isTumor ? 'fa-circle-exclamation' : 'fa-circle-check';
    const barClass   = isTumor ? 'tumor' : 'normal';
    const confColor  = isTumor ? '#ef4444' : '#10b981';

    // Stats
    const s = res.stats || {};

    // Original preview (use file reader if available)
    const origSrc  = files[i] ? URL.createObjectURL(files[i]) : '';
    const overlaySrc = res.overlay_image || '';

    // Pipeline steps
    const stepsHtml = (res.pipeline_steps || []).map((step, si) => `
      <div class="ps-thumb">
        <img src="${step.image}" alt="${step.title}" loading="lazy" />
        <div class="stage-num">Stage ${step.stage}</div>
        <span>${step.title}</span>
      </div>`).join('');

    // Factors
    const factorsHtml = (res.factors || []).map(f => `
      <div class="factor-item"><i class="fas fa-circle-dot"></i>${f}</div>`).join('');

    card.innerHTML = `
      <div class="rc-header">
        <span class="rc-filename"><i class="fas fa-file-image"></i>${res.filename}</span>
        <span class="badge ${badgeCls}">
          <i class="fas ${badgeIcon}"></i> ${res.label}
        </span>
      </div>
      <div class="rc-body">
        <div class="rc-images">
          <div class="rc-images-row">
            ${origSrc ? `<div class="rc-img-box">
              <img src="${origSrc}" alt="Original" />
              <span>Original MRI</span>
            </div>` : ''}
            ${overlaySrc ? `<div class="rc-img-box">
              <img src="${overlaySrc}" alt="Segmented" />
              <span>Segmentation Overlay</span>
            </div>` : ''}
          </div>
        </div>
        <div class="rc-info">
          <!-- Confidence -->
          <div class="conf-row">
            <span class="conf-label">Confidence</span>
            <div class="conf-bar-wrap">
              <div class="conf-bar ${barClass}" style="width:${res.confidence}%"></div>
            </div>
            <span class="conf-pct" style="color:${confColor}">${res.confidence}%</span>
          </div>

          <!-- Biomarker Stats -->
          <div class="stats-grid">
            <div class="stat-box">
              <div class="sb-val">${s.area_ratio_pct ?? '—'}%</div>
              <div class="sb-lbl">Hyperintense Area</div>
            </div>
            <div class="stat-box">
              <div class="sb-val">${s.intensity_ratio ?? '—'}x</div>
              <div class="sb-lbl">Intensity Contrast</div>
            </div>
            <div class="stat-box ${(s.ring_score_pct > 12) ? 'stat-alert' : ''}">
              <div class="sb-val">${s.ring_score_pct ?? '—'}%</div>
              <div class="sb-lbl">Ring Enhancement</div>
            </div>
            <div class="stat-box">
              <div class="sb-val">${s.edema_ratio_pct ?? '—'}%</div>
              <div class="sb-lbl">Peritumoral Edema</div>
            </div>
            <div class="stat-box ${(s.necrosis_ratio_pct > 5) ? 'stat-alert' : ''}">
              <div class="sb-val">${s.necrosis_ratio_pct ?? '—'}%</div>
              <div class="sb-lbl">Necrotic Core</div>
            </div>
            <div class="stat-box">
              <div class="sb-val">${s.asymmetry_pct ?? '—'}%</div>
              <div class="sb-lbl">Mass Effect</div>
            </div>
            <div class="stat-box">
              <div class="sb-val">${s.border_irregularity ?? '—'}</div>
              <div class="sb-lbl">Border Irregularity</div>
            </div>
            <div class="stat-box">
              <div class="sb-val">${s.suspicious_regions ?? '—'}</div>
              <div class="sb-lbl">Suspicious Regions</div>
            </div>
          </div>

          <!-- Colour legend -->
          ${isTumor ? `<div class="legend-row">
            <div class="legend-item"><span class="legend-dot" style="background:#e6201a"></span>Tumor Core</div>
            <div class="legend-item"><span class="legend-dot" style="background:#f59e0b"></span>Peritumoral Edema</div>
            <div class="legend-item"><span class="legend-dot" style="background:#c314c8"></span>Ring Enhancement</div>
            <div class="legend-item"><span class="legend-dot" style="background:#1450c8"></span>Necrotic Core</div>
          </div>` : `<div class="legend-row">
            <div class="legend-item"><span class="legend-dot" style="background:#3ab85a"></span>Gray Matter</div>
            <div class="legend-item"><span class="legend-dot" style="background:#b0a8d4"></span>White Matter</div>
            <div class="legend-item"><span class="legend-dot" style="background:#a8cae0"></span>CSF</div>
            <div class="legend-item"><span class="legend-dot" style="background:#6496d4"></span>Brain Boundary</div>
          </div>`}

          <!-- Detection factors -->
          <div class="factors-title">Detection Factors</div>
          <div class="factors-list">${factorsHtml}</div>

          <!-- Pipeline steps -->
          <div class="pipeline-title">Segmentation Pipeline</div>
          <div class="pipeline-steps-row">${stepsHtml}</div>
        </div>
      </div>`;

    resultsContainer.appendChild(card);
  });
}

"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  selectedFile: null,
  selectedObjectUrl: null,
  stream: null,
  activeView: "dashboard",
  liveRunning: false,
  liveBusy: false,
  liveTimer: null,
  lastLivePersistedEvent: null,
  liveSessionId: (crypto.randomUUID ? crypto.randomUUID() : `live-${Date.now()}-${Math.random()}`),
  videoFile: null,
  videoJobId: null,
  videoPollTimer: null,
};

const pageMeta = {
  dashboard: ["داشبورد", "نمای کلی سیستم"],
  recognize: ["تشخیص پلاک", "تصویر، دوربین و OCR"],
  video: ["پردازش ویدیو", "Tracking، رأی‌گیری OCR و بهترین فریم"],
  history: ["سوابق", "رویدادهای ذخیره‌شده"],
  search: ["جستجو", "پیدا کردن پلاک در دیتابیس"],
  settings: ["تنظیمات", "وضعیت Runtime و مدل‌ها"],
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function apiUrl(path) { return path; }

async function fetchJson(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  let payload = null;
  try { payload = await response.json(); } catch (_) { /* non-json */ }
  if (!response.ok) {
    const message = payload?.detail || `خطای HTTP ${response.status}`;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return payload;
}

function toFaDigits(value) {
  return String(value ?? "").replace(/\d/g, d => "۰۱۲۳۴۵۶۷۸۹"[Number(d)]);
}

function normalizeDigits(value) {
  return String(value ?? "")
    .replace(/[۰-۹]/g, d => String("۰۱۲۳۴۵۶۷۸۹".indexOf(d)))
    .replace(/[٠-٩]/g, d => String("٠١٢٣٤٥٦٧٨٩".indexOf(d)));
}

function plateParts(raw) {
  const value = normalizeDigits(raw).replace(/\s+/g, "");
  if (value.length !== 8) return null;
  return { left: value.slice(0,2), letter: value[2], middle: value.slice(3,6), iran: value.slice(6,8), raw: value };
}

function plateVisualHtml(raw, compact = false) {
  const p = plateParts(raw);
  if (!p) return `<bdi class="plate-fallback" dir="ltr">${escapeHtml(raw || "—")}</bdi>`;
  return `
    <div class="iran-plate ${compact ? "compact" : ""}" dir="ltr" aria-label="پلاک ${escapeHtml(raw)}">
      <span class="iran-blue"><b>IR</b><small>ایران</small></span>
      <span class="iran-num">${escapeHtml(toFaDigits(p.left))}</span>
      <span class="iran-letter" dir="rtl">${escapeHtml(p.letter)}</span>
      <span class="iran-num iran-middle">${escapeHtml(toFaDigits(p.middle))}</span>
      <span class="iran-code"><small>ایران</small><b>${escapeHtml(toFaDigits(p.iran))}</b></span>
    </div>`;
}

function plateInlineHtml(raw) {
  const p = plateParts(raw);
  if (!p) return `<bdi dir="ltr">${escapeHtml(raw || "—")}</bdi>`;
  return `<bdi class="plate-inline" dir="ltr">${escapeHtml(toFaDigits(p.left))} <span dir="rtl">${escapeHtml(p.letter)}</span> ${escapeHtml(toFaDigits(p.middle))} | ایران ${escapeHtml(toFaDigits(p.iran))}</bdi>`;
}

function plateInlineText(raw) {
  const p = plateParts(raw);
  if (!p) return String(raw || "—");
  return `${toFaDigits(p.left)} ${p.letter} ${toFaDigits(p.middle)} | ایران ${toFaDigits(p.iran)}`;
}

function sourceTypeLabel(type) {
  return ({ image: "تصویر", live: "دوربین زنده", video: "ویدیو" })[type] || type || "تصویر";
}

function formatSeconds(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const minutes = Math.floor(n / 60);
  const seconds = Math.floor(n % 60);
  return `${toFaDigits(minutes)}:${toFaDigits(String(seconds).padStart(2, "0"))}`;
}

function regionLabel(region) {
  if (!region?.known) return "مبدأ پلاک نامشخص";
  if (region.city) return `${region.city} — ${region.province}`;
  return region.province || region.label || "نامشخص";
}

function regionHtml(region) {
  if (!region) return "";
  const code = region.iran_code ? `ایران ${toFaDigits(region.iran_code)}` : "";
  return `<div class="region-line"><span class="region-pin">⌖</span><strong>${escapeHtml(regionLabel(region))}</strong>${code ? `<small>${escapeHtml(code)}</small>` : ""}</div>`;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("fa-IR", {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit",
  }).format(date);
}

function formatConfidence(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

let toastTimer = null;
function showToast(message, type = "ok") {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", type === "error");
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3200);
}

function switchView(view) {
  if (!pageMeta[view]) return;
  state.activeView = view;
  $$(".view").forEach(el => el.classList.toggle("active", el.id === `view-${view}`));
  $$('[data-view]').forEach(el => el.classList.toggle("active", el.dataset.view === view));
  $("#pageTitle").textContent = pageMeta[view][0];
  $("#pageEyebrow").textContent = pageMeta[view][1];
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (view === "dashboard") loadDashboard();
  if (view === "history") loadHistory();
  if (view === "settings") loadHealth();
}

function bindNavigation() {
  $$('[data-view]').forEach(el => el.addEventListener("click", () => switchView(el.dataset.view)));
  $$('[data-go-recognize]').forEach(el => el.addEventListener("click", () => switchView("recognize")));
  $$('[data-view-link]').forEach(el => el.addEventListener("click", () => switchView(el.dataset.viewLink)));
}

async function loadHealth() {
  try {
    const data = await fetchJson("/health");
    const ready = data.status === "ok" && data.detector_exists && data.ocr_exists;
    $("#sidebarStatusDot").className = `status-dot ${ready ? "ok" : "bad"}`;
    $("#sidebarStatusText").textContent = ready ? "سیستم آماده است" : "نیاز به بررسی";
    $("#statHealth").textContent = ready ? "آماده پردازش" : "مدل ناقص";
    $("#statDevice").textContent = `Device: ${data.device ?? "—"}`;
    if ($("#videoLimitLabel") && data.max_video_mb) $("#videoLimitLabel").textContent = `حداکثر حجم ویدیو: ${toFaDigits(data.max_video_mb)} مگابایت`;
    const rows = [
      ["Backend", data.status], ["نسخه API", data.version], ["Device", data.device],
      ["Detector", data.detector_exists ? "موجود ✓" : "یافت نشد"],
      ["OCR", data.ocr_exists ? "موجود ✓" : "یافت نشد"],
      ["Live dedup", `${data.live_dedup_seconds ?? "—"} sec`],
      ["Video dedup", `${data.video_dedup_seconds ?? "—"} sec`],
      ["حداکثر ویدیو", `${data.max_video_mb ?? "—"} MB`],
      ["Database", data.database], ["Storage", data.storage],
    ];
    $("#healthDetails").innerHTML = rows.map(([k,v]) => `<div class="kv-row"><span>${escapeHtml(k)}</span><strong>${escapeHtml(v)}</strong></div>`).join("");
    return data;
  } catch (error) {
    $("#sidebarStatusDot").className = "status-dot bad";
    $("#sidebarStatusText").textContent = "Backend در دسترس نیست";
    $("#statHealth").textContent = "قطع ارتباط";
    $("#healthDetails").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    return null;
  }
}

async function loadDashboard() {
  loadHealth();
  try {
    const [stats, detections] = await Promise.all([fetchJson("/stats"), fetchJson("/detections?limit=6&offset=0")]);
    $("#statUnique").textContent = Number(stats.unique_plates ?? 0).toLocaleString("fa-IR");
    $("#statDetections").textContent = Number(stats.detection_events ?? 0).toLocaleString("fa-IR");
    $("#statLatest").textContent = formatDate(stats.latest_detection_at);
    const items = detections.items || [];
    $("#recentList").innerHTML = items.length ? items.map(item => `
      <div class="list-item">
        ${item.crop_url ? `<img class="list-thumb" src="${escapeHtml(item.crop_url)}" alt="Crop پلاک" />` : `<div class="list-thumb"></div>`}
        <div><strong>${plateInlineHtml(item.plate_text)}</strong><span>${escapeHtml(regionLabel(item.region))} · ${escapeHtml(formatDate(item.detected_at))}</span></div>
        <span class="conf-pill">${escapeHtml(formatConfidence(item.det_confidence))}</span>
      </div>`).join("") : `<div class="empty-state">هنوز تشخیصی در دیتابیس ثبت نشده است.</div>`;
  } catch (error) {
    $("#recentList").innerHTML = `<div class="empty-state">خطا در دریافت Dashboard: ${escapeHtml(error.message)}</div>`;
  }
}

function sourceSwitch(source) {
  $$(".source-tab").forEach(btn => btn.classList.toggle("active", btn.dataset.source === source));
  $$(".source-pane").forEach(pane => pane.classList.toggle("active", pane.id === `source-${source}`));
}

function cleanupObjectUrl() {
  if (state.selectedObjectUrl) URL.revokeObjectURL(state.selectedObjectUrl);
  state.selectedObjectUrl = null;
}

function setSelectedFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) { showToast("فایل انتخاب‌شده تصویر نیست.", "error"); return; }
  cleanupObjectUrl();
  state.selectedFile = file;
  state.selectedObjectUrl = URL.createObjectURL(file);
  $("#previewImage").src = state.selectedObjectUrl;
  $("#previewWrap").classList.remove("hidden");
  $("#recognizeBtn").disabled = false;
  resetResult();
}

function clearSelectedFile() {
  cleanupObjectUrl();
  state.selectedFile = null;
  $("#imageInput").value = "";
  $("#previewImage").removeAttribute("src");
  $("#previewWrap").classList.add("hidden");
  $("#recognizeBtn").disabled = true;
}

function resetResult() {
  $("#resultStatus").className = "result-status neutral";
  $("#resultStatus").textContent = "آماده";
  $("#resultEmpty").classList.remove("hidden");
  $("#resultContent").classList.add("hidden");
  $("#annotatedFrame").classList.remove("hidden");
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("این مرورگر دسترسی Camera API ندارد.", "error");
    return false;
  }
  stopLiveRecognition();
  stopCameraTracks();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60, max: 60 } },
      audio: false,
    });
    state.stream = stream;
    const video = $("#cameraVideo");
    video.srcObject = stream;
    await video.play();
    $("#cameraPlaceholder").classList.add("hidden");
    $("#captureBtn").disabled = false;
    $("#startLiveBtn").disabled = false;
    $("#stopCameraBtn").disabled = false;
    $("#startCameraBtn").textContent = "تعویض / شروع مجدد";
    $("#liveStatus").textContent = "دوربین آماده است";
    return true;
  } catch (error) {
    showToast(`دسترسی به دوربین ممکن نشد: ${error.message}`, "error");
    return false;
  }
}

function stopCameraTracks() {
  if (state.stream) state.stream.getTracks().forEach(track => track.stop());
  state.stream = null;
  $("#cameraVideo").srcObject = null;
  $("#cameraPlaceholder").classList.remove("hidden");
  $("#captureBtn").disabled = true;
  $("#startLiveBtn").disabled = true;
  $("#stopCameraBtn").disabled = true;
  clearLiveOverlay();
}

function stopCamera() {
  stopLiveRecognition();
  stopCameraTracks();
  $("#liveStatus").textContent = "دوربین متوقف است";
}

function captureFrameToCanvas(maxWidth = null) {
  const video = $("#cameraVideo");
  if (!video.videoWidth || !video.videoHeight) return null;
  const sourceW = video.videoWidth;
  const sourceH = video.videoHeight;
  const scale = maxWidth && sourceW > maxWidth ? maxWidth / sourceW : 1;
  const width = Math.round(sourceW * scale);
  const height = Math.round(sourceH * scale);
  const canvas = $("#cameraCanvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.drawImage(video, 0, 0, width, height);
  return { canvas, width, height };
}

function captureCamera() {
  const captured = captureFrameToCanvas();
  if (!captured) return;
  captured.canvas.toBlob(blob => {
    if (!blob) return;
    const file = new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" });
    setSelectedFile(file);
    showToast("عکس دوربین آماده پردازش است.");
  }, "image/jpeg", 0.94);
}

function frameBlob(maxWidth = 1280, quality = 0.82) {
  const captured = captureFrameToCanvas(maxWidth);
  if (!captured) return Promise.resolve(null);
  return new Promise(resolve => captured.canvas.toBlob(blob => resolve(blob ? { blob, width: captured.width, height: captured.height } : null), "image/jpeg", quality));
}

function clearLiveOverlay() {
  const canvas = $("#cameraOverlay");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawLiveOverlay(data, width, height) {
  const canvas = $("#cameraOverlay");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  for (const item of (data.results || [])) {
    const accepted = Boolean(item.accepted);
    ctx.strokeStyle = accepted ? "#36d38a" : "#ff667d";
    ctx.lineWidth = Math.max(2, width / 500);
    const x = Number(item.x1), y = Number(item.y1), w = Number(item.x2) - x, h = Number(item.y2) - y;
    ctx.strokeRect(x, y, w, h);
    if (accepted) {
      const text = plateParts(item.raw_text)?.raw || item.raw_text || "PLATE";
      ctx.font = `bold ${Math.max(16, width / 42)}px Tahoma`;
      const pad = 8;
      const metrics = ctx.measureText(text);
      const boxH = Math.max(28, width / 30);
      const labelY = Math.max(boxH, y - 4);
      ctx.fillStyle = "rgba(4,18,15,.86)";
      ctx.fillRect(x, labelY - boxH, metrics.width + pad * 2, boxH);
      ctx.fillStyle = "#6ff1b0";
      ctx.fillText(text, x + pad, labelY - 7);
    }
  }
}

function renderPlateCards(results, live = false) {
  if (!results.length) return `<div class="empty-state">${live ? "در این فریم پلاکی دیده نشد؛ پایش ادامه دارد." : "هیچ Bounding Box پلاکی پیدا نشد."}</div>`;
  return results.map(item => `
    <article class="plate-result-card ${item.accepted ? "accepted" : "rejected"}">
      ${item.crop_url ? `<img class="plate-crop" src="${escapeHtml(item.crop_url)}?t=${Date.now()}" alt="Crop پلاک" />` : `<div class="plate-crop plate-crop-placeholder">LIVE</div>`}
      <div class="plate-result-main">
        ${plateVisualHtml(item.raw_text)}
        ${regionHtml(item.region)}
        <div class="plate-meta">DET ${escapeHtml(formatConfidence(item.det_confidence))} · Sharp ${escapeHtml(Math.round(Number(item.sharpness || 0)))} · RAW <bdi dir="ltr">${escapeHtml(item.raw_text || "—")}</bdi></div>
        ${live && item.track_id ? `<div class="track-note">Track #${escapeHtml(toFaDigits(item.track_id))} · ${escapeHtml(toFaDigits(item.temporal_valid_hits ?? 0))}/${escapeHtml(toFaDigits(item.temporal_hits ?? 1))} فریم معتبر${item.consensus_text ? ` · رأی: ${escapeHtml(plateInlineText(item.consensus_text))}` : ""}</div>` : ""}
        ${live && item.accepted ? `<div class="live-db-note ${item.live_persisted ? "saved" : "seen"}">${item.live_persisted ? "ثبت نهایی در دیتابیس" : "در حال Tracking / تثبیت"}</div>` : ""}
      </div>
      <span class="${item.accepted ? "accept-pill" : "reject-pill"}">${item.accepted ? "معتبر" : escapeHtml(item.status || "رد شد")}</span>
    </article>`).join("");
}

function renderRecognition(data, live = false) {
  $("#resultEmpty").classList.add("hidden");
  $("#resultContent").classList.remove("hidden");
  if (data.annotated_url) {
    $("#annotatedFrame").classList.remove("hidden");
    $("#annotatedImage").src = `${data.annotated_url}?t=${Date.now()}`;
  } else {
    $("#annotatedFrame").classList.add("hidden");
  }
  $("#resultStatus").className = `result-status ${data.accepted > 0 ? "success" : "neutral"}`;
  $("#resultStatus").textContent = live ? (data.accepted > 0 ? `${data.accepted} پلاک در تصویر زنده` : "پایش زنده") : (data.accepted > 0 ? `${data.accepted} پلاک معتبر` : "پلاک معتبر یافت نشد");
  $("#resultSummary").innerHTML = `
    <span class="summary-chip">تشخیص: ${escapeHtml(data.detections)}</span>
    <span class="summary-chip">معتبر: ${escapeHtml(data.accepted)}</span>
    <span class="summary-chip">ردشده: ${escapeHtml(data.rejected)}</span>
    <span class="summary-chip">ثبت DB: ${escapeHtml(data.database?.inserted ?? 0)}</span>
    ${live ? `<span class="summary-chip live-chip">LIVE</span>` : ""}
  `;
  $("#plateResults").innerHTML = renderPlateCards(data.results || [], live);
}

async function recognizeSelected() {
  if (!state.selectedFile) return;
  const btn = $("#recognizeBtn");
  btn.disabled = true;
  $("#recognizeBtnText").textContent = "در حال پردازش...";
  $("#recognizeSpinner").classList.remove("hidden");
  $("#resultStatus").className = "result-status neutral";
  $("#resultStatus").textContent = "در حال پردازش";
  const form = new FormData();
  form.append("file", state.selectedFile, state.selectedFile.name || "image.jpg");
  form.append("det_conf", $("#confRange").value);
  form.append("allow_edge", $("#allowEdge").checked ? "true" : "false");
  try {
    const data = await fetchJson("/api/recognize/image", { method: "POST", body: form });
    renderRecognition(data, false);
    showToast(data.accepted > 0 ? "تشخیص انجام شد و پلاک معتبر ثبت شد." : "پردازش انجام شد؛ پلاک معتبر پیدا نشد.");
    loadDashboard();
  } catch (error) {
    $("#resultStatus").className = "result-status error";
    $("#resultStatus").textContent = "خطا";
    showToast(`خطا در تشخیص: ${error.message}`, "error");
  } finally {
    btn.disabled = !state.selectedFile;
    $("#recognizeBtnText").textContent = "تشخیص پلاک";
    $("#recognizeSpinner").classList.add("hidden");
  }
}

async function processLiveFrame() {
  if (!state.liveRunning || state.liveBusy || !state.stream) return;
  state.liveBusy = true;
  try {
    const captured = await frameBlob(1280, 0.82);
    if (!captured || !state.liveRunning) return;
    const form = new FormData();
    form.append("file", captured.blob, `live-${Date.now()}.jpg`);
    form.append("session_id", state.liveSessionId);
    form.append("det_conf", $("#confRange").value);
    form.append("allow_edge", $("#allowEdge").checked ? "true" : "false");
    form.append("min_hits", $("#liveMinHits")?.value || "2");
    const started = performance.now();
    const data = await fetchJson("/api/recognize/live-frame", { method: "POST", body: form });
    const elapsed = performance.now() - started;
    drawLiveOverlay(data, captured.width, captured.height);
    renderRecognition(data, true);
    $("#liveStatus").textContent = `پایش فعال · ${data.active_tracks ?? 0} Track · ${data.accepted} پلاک · ${Math.round(elapsed)} ms`;
    $("#liveIndicator").className = "live-indicator on";
    if (data.database?.inserted > 0 && data.event_id && data.event_id !== state.lastLivePersistedEvent) {
      state.lastLivePersistedEvent = data.event_id;
      const first = data.committed?.[0]?.result || (data.results || []).find(x => x.live_persisted);
      showToast(first ? `پلاک ${plateInlineText(first.raw_text)} ثبت شد · ${regionLabel(first.region)}` : "پلاک جدید از دوربین ثبت شد.");
      loadDashboard();
    }
  } catch (error) {
    $("#liveStatus").textContent = `خطا: ${error.message}`;
    $("#liveIndicator").className = "live-indicator error";
    if (state.liveRunning) showToast(`خطا در پایش زنده: ${error.message}`, "error");
  } finally {
    state.liveBusy = false;
    if (state.liveRunning) {
      const delay = Number($("#liveDelay").value || 0);
      clearTimeout(state.liveTimer);
      state.liveTimer = setTimeout(processLiveFrame, delay);
    }
  }
}

async function startLiveRecognition() {
  if (state.liveRunning) return;
  if (!state.stream) {
    const ready = await startCamera();
    if (!ready) return;
  }
  state.liveRunning = true;
  $("#startLiveBtn").disabled = true;
  $("#stopLiveBtn").disabled = false;
  $("#captureBtn").disabled = false;
  $("#liveStatus").textContent = "در حال شروع پایش زنده...";
  $("#liveIndicator").className = "live-indicator on";
  sourceSwitch("camera");
  processLiveFrame();
}

function stopLiveRecognition() {
  state.liveRunning = false;
  state.liveBusy = false;
  clearTimeout(state.liveTimer);
  state.liveTimer = null;
  const startBtn = $("#startLiveBtn");
  const stopBtn = $("#stopLiveBtn");
  if (startBtn) startBtn.disabled = !state.stream;
  if (stopBtn) stopBtn.disabled = true;
  const indicator = $("#liveIndicator");
  if (indicator) indicator.className = "live-indicator";
  const status = $("#liveStatus");
  if (status && state.stream) status.textContent = "پایش متوقف است؛ دوربین همچنان روشن است";
  clearLiveOverlay();
}

async function deleteDetectionFromDb(id) {
  if (!confirm("این رویداد از دیتابیس و تصاویر مربوط به آن حذف شود؟")) return;
  try {
    await fetchJson(`/detections/${encodeURIComponent(id)}?delete_media=true`, { method: "DELETE" });
    showToast("رویداد حذف شد.");
    await Promise.all([loadHistory(), loadDashboard()]);
  } catch (error) { showToast(`حذف انجام نشد: ${error.message}`, "error"); }
}

async function deletePlateFromDb(text) {
  const display = plateInlineText(text);
  if (!confirm(`تمام سوابق پلاک ${display} از دیتابیس حذف شود؟`)) return;
  try {
    await fetchJson(`/plates/${encodeURIComponent(text)}?delete_media=true`, { method: "DELETE" });
    showToast("تمام سوابق این پلاک حذف شد.");
    $("#searchResults").innerHTML = `<div class="empty-state">پلاک حذف شد. دوباره جستجو کن.</div>`;
    await loadDashboard();
  } catch (error) { showToast(`حذف انجام نشد: ${error.message}`, "error"); }
}

async function clearWholeDatabase() {
  if (!confirm("هشدار: تمام پلاک‌ها و سوابق تشخیص حذف می‌شوند. ادامه می‌دهی؟")) return;
  const typed = prompt("برای تأیید نهایی عبارت DELETE_ALL را دقیق وارد کن:");
  if (typed !== "DELETE_ALL") { showToast("پاک‌سازی لغو شد.", "error"); return; }
  try {
    const data = await fetchJson(`/api/database?confirm=DELETE_ALL&delete_media=true`, { method: "DELETE" });
    showToast(`${toFaDigits(data.deleted?.detections ?? 0)} رویداد از دیتابیس پاک شد.`);
    await Promise.all([loadDashboard(), loadHistory()]);
  } catch (error) { showToast(`پاک‌سازی انجام نشد: ${error.message}`, "error"); }
}

async function loadHistory() {
  const body = $("#historyBody");
  body.innerHTML = `<tr><td colspan="8" class="empty-cell">در حال دریافت...</td></tr>`;
  try {
    const data = await fetchJson("/detections?limit=100&offset=0");
    const items = data.items || [];
    body.innerHTML = items.length ? items.map(item => `
      <tr>
        <td class="plate-cell">${plateInlineHtml(item.plate_text)}</td>
        <td>${escapeHtml(regionLabel(item.region))}</td>
        <td>${escapeHtml(formatDate(item.detected_at))}${item.source_type === "video" ? `<small class="table-sub">${formatSeconds(item.source_time_seconds)} در ویدیو</small>` : ""}</td>
        <td><span class="source-pill ${escapeHtml(item.source_type || "image")}">${escapeHtml(sourceTypeLabel(item.source_type))}</span></td>
        <td dir="ltr">${escapeHtml(formatConfidence(item.det_confidence))}${item.temporal_hits ? `<small class="table-sub" dir="rtl">${toFaDigits(item.temporal_hits)} فریم</small>` : ""}</td>
        <td>${item.crop_url ? `<a href="${escapeHtml(item.crop_url)}" target="_blank"><img class="table-thumb" src="${escapeHtml(item.crop_url)}" alt="Crop پلاک" /></a>` : "—"}</td>
        <td>${item.source_url ? `<a class="text-btn" href="${escapeHtml(item.source_url)}" target="_blank">تصویر مدرک</a>` : "—"}</td>
        <td><button class="icon-danger" type="button" data-delete-detection="${escapeHtml(item.id)}" title="حذف رویداد">حذف</button></td>
      </tr>`).join("") : `<tr><td colspan="8" class="empty-cell">هنوز سابقه‌ای وجود ندارد.</td></tr>`;
  } catch (error) {
    body.innerHTML = `<tr><td colspan="8" class="empty-cell">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function runSearch(event) {
  event.preventDefault();
  const q = normalizeDigits($("#searchInput").value.trim()).replace(/\s+/g, "");
  if (!q) return;
  $("#searchResults").innerHTML = `<div class="empty-state">در حال جستجو...</div>`;
  try {
    const data = await fetchJson(`/search?q=${encodeURIComponent(q)}&limit=100`);
    const items = data.items || [];
    $("#searchResults").innerHTML = items.length ? items.map(item => `
      <article class="search-card">
        <div class="search-card-main">${plateVisualHtml(item.plate_text, true)}${regionHtml(item.region)}<small>آخرین مشاهده: ${escapeHtml(formatDate(item.last_seen_at))}</small></div>
        <div class="search-card-actions"><span class="count-bubble" title="تعداد تشخیص">${escapeHtml(toFaDigits(item.detection_count))}</span><button class="icon-danger" type="button" data-delete-plate="${escapeHtml(item.plate_text)}">حذف همه</button></div>
      </article>`).join("") : `<div class="empty-state">نتیجه‌ای برای «${escapeHtml(q)}» پیدا نشد.</div>`;
  } catch (error) {
    $("#searchResults").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function setVideoFile(file) {
  if (!file) return;
  if (!(file.type || "").startsWith("video/") && !/\.(mp4|mov|avi|mkv|m4v|webm)$/i.test(file.name || "")) {
    showToast("فایل انتخاب‌شده ویدیو نیست.", "error"); return;
  }
  state.videoFile = file;
  $("#videoFileLabel").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  $("#processVideoBtn").disabled = false;
  $("#videoResults").innerHTML = `<div class="empty-state">ویدیو آماده پردازش است.</div>`;
}

function renderVideoEvents(events) {
  if (!events?.length) return `<div class="empty-state">هنوز پلاک نهایی از این ویدیو ثبت نشده است.</div>`;
  return events.map(item => `
    <article class="video-event-card">
      ${item.source_url ? `<img src="${escapeHtml(item.source_url)}?t=${Date.now()}" alt="بهترین فریم خودرو" />` : ""}
      <div class="video-event-body">
        ${plateVisualHtml(item.raw_text, true)}
        ${regionHtml(item.region)}
        <div class="video-event-meta"><span>زمان ویدیو: ${formatSeconds(item.timestamp_seconds ?? item.source_time_seconds)}</span><span>${toFaDigits(item.temporal_hits ?? 1)} فریم Track</span><span>Sharp ${toFaDigits(Math.round(Number(item.sharpness || 0)))}</span></div>
      </div>
    </article>`).join("");
}

async function pollVideoJob() {
  if (!state.videoJobId) return;
  try {
    const job = await fetchJson(`/api/video/jobs/${encodeURIComponent(state.videoJobId)}`);
    const progress = Number(job.progress || 0);
    $("#videoProgressBar").style.width = `${Math.max(0, Math.min(100, progress))}%`;
    $("#videoProgressText").textContent = `${toFaDigits(progress.toFixed(1))}٪`;
    const labels = { queued: "در صف", running: "در حال پردازش", completed: "تکمیل شد", failed: "خطا" };
    $("#videoJobStatus").textContent = labels[job.status] || job.status;
    $("#videoJobMeta").innerHTML = `<span>نمونه‌های پردازش‌شده: ${toFaDigits(job.processed_samples ?? 0)}</span><span>Detection: ${toFaDigits(job.detections ?? 0)}</span><span>پلاک ثبت‌شده: ${toFaDigits(job.saved_events ?? 0)}</span>${job.duration_seconds ? `<span>مدت ویدیو: ${formatSeconds(job.duration_seconds)}</span>` : ""}`;
    $("#videoResults").innerHTML = renderVideoEvents(job.events || []);
    if (job.status === "completed") {
      $("#processVideoBtn").disabled = !state.videoFile;
      showToast(`پردازش ویدیو تمام شد؛ ${toFaDigits(job.saved_events ?? 0)} رویداد ثبت شد.`);
      loadDashboard();
      return;
    }
    if (job.status === "failed") {
      $("#processVideoBtn").disabled = !state.videoFile;
      showToast(`پردازش ویدیو شکست خورد: ${job.error || "خطای نامشخص"}`, "error");
      return;
    }
    clearTimeout(state.videoPollTimer);
    state.videoPollTimer = setTimeout(pollVideoJob, 900);
  } catch (error) {
    showToast(`دریافت وضعیت ویدیو ناموفق بود: ${error.message}`, "error");
  }
}

async function startVideoProcessing() {
  if (!state.videoFile) return;
  const form = new FormData();
  form.append("file", state.videoFile, state.videoFile.name || "video.mp4");
  form.append("det_conf", $("#confRange").value);
  form.append("allow_edge", $("#allowEdge").checked ? "true" : "false");
  form.append("sample_fps", $("#videoSampleFps").value);
  form.append("min_hits", $("#videoMinHits").value);
  $("#processVideoBtn").disabled = true;
  $("#videoProgressWrap").classList.remove("hidden");
  $("#videoJobStatus").textContent = "در حال آپلود...";
  $("#videoProgressBar").style.width = "1%";
  try {
    const job = await fetchJson("/api/recognize/video", { method: "POST", body: form });
    state.videoJobId = job.job_id;
    $("#videoResults").innerHTML = `<div class="empty-state">پردازش شروع شد. پلاک‌ها به‌محض نهایی‌شدن Track ظاهر می‌شوند.</div>`;
    pollVideoJob();
  } catch (error) {
    $("#processVideoBtn").disabled = false;
    showToast(`آپلود ویدیو انجام نشد: ${error.message}`, "error");
  }
}

function bindRecognition() {
  $$(".source-tab").forEach(btn => btn.addEventListener("click", () => sourceSwitch(btn.dataset.source)));
  const input = $("#imageInput");
  input.addEventListener("change", () => setSelectedFile(input.files?.[0]));
  const zone = $("#dropZone");
  ["dragenter", "dragover"].forEach(type => zone.addEventListener(type, e => { e.preventDefault(); zone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach(type => zone.addEventListener(type, e => { e.preventDefault(); zone.classList.remove("dragging"); }));
  zone.addEventListener("drop", e => setSelectedFile(e.dataTransfer?.files?.[0]));
  $("#clearImageBtn").addEventListener("click", clearSelectedFile);
  $("#confRange").addEventListener("input", e => $("#confValue").textContent = Number(e.target.value).toFixed(2));
  $("#startCameraBtn").addEventListener("click", startCamera);
  $("#stopCameraBtn").addEventListener("click", stopCamera);
  $("#captureBtn").addEventListener("click", captureCamera);
  $("#startLiveBtn").addEventListener("click", startLiveRecognition);
  $("#stopLiveBtn").addEventListener("click", stopLiveRecognition);
  $("#recognizeBtn").addEventListener("click", recognizeSelected);
}

function bindOther() {
  $("#refreshHistoryBtn").addEventListener("click", loadHistory);
  $("#searchForm").addEventListener("submit", runSearch);
  $("#clearDatabaseBtn")?.addEventListener("click", clearWholeDatabase);
  const videoInput = $("#videoInput");
  const videoZone = $("#videoDropZone");
  videoInput?.addEventListener("change", () => setVideoFile(videoInput.files?.[0]));
  if (videoZone) {
    ["dragenter", "dragover"].forEach(type => videoZone.addEventListener(type, e => { e.preventDefault(); videoZone.classList.add("dragging"); }));
    ["dragleave", "drop"].forEach(type => videoZone.addEventListener(type, e => { e.preventDefault(); videoZone.classList.remove("dragging"); }));
    videoZone.addEventListener("drop", e => setVideoFile(e.dataTransfer?.files?.[0]));
  }
  $("#processVideoBtn")?.addEventListener("click", startVideoProcessing);
  document.addEventListener("click", event => {
    const det = event.target.closest?.("[data-delete-detection]");
    if (det) deleteDetectionFromDb(det.dataset.deleteDetection);
    const plate = event.target.closest?.("[data-delete-plate]");
    if (plate) deletePlateFromDb(plate.dataset.deletePlate);
  });
  window.addEventListener("beforeunload", () => { stopCamera(); cleanupObjectUrl(); clearTimeout(state.videoPollTimer); });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state.liveRunning) stopLiveRecognition();
  });
}

function init() {
  bindNavigation();
  bindRecognition();
  bindOther();
  loadDashboard();
}

document.addEventListener("DOMContentLoaded", init);

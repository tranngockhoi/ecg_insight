/* engine.js – chạy ECGNetX ngay trong trình duyệt (onnxruntime-web).
   Port từ ecgmodel.py của bản máy tính: tiền xử lý đúng như lúc huấn luyện, chấm 137 nhãn, cổng "không phải ECG",
   vùng giấy ECG, bản đồ chú ý Grad-CAM (tính giải tích từ đặc trưng, khớp autograd), chất lượng ảnh, nhận định.
   Chỉ để NGHIÊN CỨU. */
(function (global) {
  'use strict';
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  const THRESH_HIGH = 0.5, THRESH_MID = 0.3, EPS = 0.05;
  const BENIGN = new Set(['sup_NORM', 'sub_NORM', 'scp_NORM', 'scp_SR']);   // nhãn "bình thường": không đưa vào danh sách đáng lưu ý

  /* ---------- resize song tuyến tính có khử răng cưa, giống PIL.Image.resize(BILINEAR) ---------- */
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize, fs = Math.max(scale, 1), support = fs;
    const out = [];
    for (let x = 0; x < outSize; x++) {
      const c = (x + 0.5) * scale;
      const xmin = Math.max(Math.trunc(c - support + 0.5), 0), xmax = Math.min(Math.trunc(c + support + 0.5), inSize);
      const w = []; let s = 0;
      for (let k = xmin; k < xmax; k++) { let t = (k - c + 0.5) / fs; t = Math.abs(t); const v = t < 1 ? 1 - t : 0; w.push(v); s += v; }
      for (let k = 0; k < w.length; k++) w[k] /= (s || 1);
      out.push({ xmin, w });
    }
    return out;
  }
  /* src: Uint8 mảng có `ch` kênh xen kẽ; trả về Uint8 cùng số kênh */
  function resize(src, sw, sh, dw, dh, ch) {
    let cur = src, cw = sw;
    if (dw !== sw) {
      const cf = coeffs(sw, dw), tmp = new Uint8ClampedArray(dw * sh * ch);
      for (let y = 0; y < sh; y++) {
        const rowS = y * sw * ch, rowD = y * dw * ch;
        for (let x = 0; x < dw; x++) {
          const { xmin, w } = cf[x];
          for (let c = 0; c < ch; c++) { let a = 0; for (let k = 0; k < w.length; k++) a += w[k] * cur[rowS + (xmin + k) * ch + c]; tmp[rowD + x * ch + c] = a + 0.5; }
        }
      }
      cur = tmp; cw = dw;
    }
    if (dh !== sh) {
      const cf = coeffs(sh, dh), out = new Uint8ClampedArray(cw * dh * ch);
      for (let y = 0; y < dh; y++) {
        const { xmin, w } = cf[y];
        for (let x = 0; x < cw; x++) for (let c = 0; c < ch; c++) {
          let a = 0; for (let k = 0; k < w.length; k++) a += w[k] * cur[((xmin + k) * cw + x) * ch + c];
          out[(y * cw + x) * ch + c] = a + 0.5;
        }
      }
      cur = out;
    }
    return cur;
  }
  const rgbaToRgb = d => { const n = d.length >> 2, o = new Uint8ClampedArray(n * 3); for (let i = 0, j = 0; i < n; i++) { o[j++] = d[4 * i]; o[j++] = d[4 * i + 1]; o[j++] = d[4 * i + 2]; } return o; };
  const toGray = (rgb, n) => { const g = new Uint8ClampedArray(n); for (let i = 0; i < n; i++) g[i] = (299 * rgb[3 * i] + 587 * rgb[3 * i + 1] + 114 * rgb[3 * i + 2] + 500) / 1000; return g; };
  function percentile(sorted, p) { const i = p / 100 * (sorted.length - 1), lo = Math.floor(i), hi = Math.min(lo + 1, sorted.length - 1); return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo); }

  /* ---------- chất lượng ảnh ---------- */
  function qualityReport(rgb, w, h) {
    const gw = 1024, gh = Math.max(1, Math.trunc(1024 * h / w));
    const gSrc = toGray(rgb, w * h);
    const g8 = resize(gSrc, w, h, gw, gh, 1), g = Float32Array.from(g8);
    let sum = 0, clip = 0, dark = 0; for (let i = 0; i < g.length; i++) { sum += g[i]; if (g[i] >= 252) clip++; if (g[i] < 90) dark++; }
    const bright = sum / g.length, clipped = clip / g.length, darkF = dark / g.length;
    let ls = 0, ls2 = 0, n = 0;
    for (let y = 1; y < gh - 1; y++) for (let x = 1; x < gw - 1; x++) {
      const i = y * gw + x, v = 4 * g[i] - g[i - gw] - g[i + gw] - g[i - 1] - g[i + 1]; ls += v; ls2 += v * v; n++;
    }
    const sharp = n ? ls2 / n - (ls / n) * (ls / n) : 0;
    const sorted = Float32Array.from(g).sort(), contrast = percentile(sorted, 95) - percentile(sorted, 5);
    const aw = 256, ah = Math.max(1, Math.trunc(256 * h / w)), a = resize(rgb, w, h, aw, ah, 3);
    let col = 0; for (let i = 0; i < aw * ah; i++) { const r = a[3 * i], gg = a[3 * i + 1], b = a[3 * i + 2], mx = Math.max(r, gg, b), mn = Math.min(r, gg, b); if (mx - mn > 18 && mx > 140) col++; }
    const colored = col / (aw * ah);
    const items = []; let score = 100;
    const add = (level, title, text) => items.push({ level, title, text });
    if (sharp < 25) { add('bad', 'Độ nét', 'Ảnh mờ. Giữ máy cố định và chạm màn hình để lấy nét.'); score -= 30; }
    else if (sharp < 60) { add('warn', 'Độ nét', 'Ảnh hơi mềm; chụp rõ nét hơn sẽ đọc tốt hơn.'); score -= 12; }
    else add('ok', 'Độ nét', 'Nét ECG sắc nét.');
    if (bright < 90) { add('bad', 'Độ sáng', 'Ảnh quá tối. Chụp ở nơi đủ sáng.'); score -= 25; }
    else if (bright < 130) { add('warn', 'Độ sáng', 'Ảnh hơi tối.'); score -= 10; }
    else if (clipped > 0.7) { add('warn', 'Độ sáng', 'Ảnh có thể bị cháy sáng. Tránh đèn chiếu thẳng vào giấy.'); score -= 10; }
    else add('ok', 'Độ sáng', 'Độ sáng phù hợp.');
    if (contrast < 60) { add('warn', 'Tương phản', 'Tương phản thấp, nét ECG có thể bị chìm.'); score -= 12; }
    if (h > w * 1.1) { add('warn', 'Hướng ảnh', 'Ảnh đang nằm dọc. ECG thường nằm ngang; xoay ảnh trước khi phân tích.'); score -= 15; }
    if (darkF < 0.002) { add('bad', 'Nét ECG', 'Hầu như không thấy nét ECG trong ảnh.'); score -= 30; }
    else if (colored > 0.01) add('ok', 'Lưới', 'Nhận thấy lưới ô vuông có màu.');
    else add('info', 'Lưới', 'Không thấy lưới màu (có thể là bản in trắng đen).');
    if (Math.min(w, h) < 500) { add('warn', 'Độ phân giải', 'Độ phân giải thấp.'); score -= 12; }
    score = Math.max(0, Math.min(100, Math.round(score)));
    return { score, items, sharpness: +sharp.toFixed(1), brightness: +bright.toFixed(1), width: w, height: h };
  }

  /* ---------- gom kết quả (khớp build_result của bản Python) ---------- */
  function buildResult(cols, probs, labels) {
    const sup = {};
    for (const k of cols) if (k.startsWith('sup_')) sup[k.slice(4)] = probs[k];
    const abn = ['MI', 'STTC', 'CD', 'HYP'].filter(c => c in sup).map(c => [c, sup[c]]).sort((a, b) => b[1] - a[1]);
    const flagged = abn.filter(t => t[1] >= THRESH_HIGH).map(t => t[0]);
    const detail = cols.filter(k => !k.startsWith('sup_') && !labels[k].hidden).map(k => [k, probs[k]]).sort((a, b) => b[1] - a[1]);
    const high = detail.filter(t => t[1] >= THRESH_HIGH && !BENIGN.has(t[0])).map(t => t[0]);
    const norm = sup.NORM || 0;
    const level = flagged.length ? 'abnormal' : norm >= THRESH_HIGH ? 'normal' : 'uncertain';
    let cam_default = null;
    if (flagged.length) cam_default = 'sup_' + flagged[0];
    else if (detail.length) cam_default = norm >= THRESH_HIGH ? 'sup_NORM' : (abn.length ? 'sup_' + abn[0][0] : detail[0][0]);
    const scores = {}; for (const k of cols) scores[k] = Math.round(probs[k] * 1e4) / 1e4;
    return { scores, cam_default, impression: { level, groups: flagged, high: high.slice(0, 8), n_mid: detail.filter(t => t[1] >= THRESH_MID && t[1] < THRESH_HIGH).length, norm: Math.round(norm * 1e4) / 1e4 } };
  }

  /* ---------- bản đồ nhiệt / vùng giấy: bilinear (align_corners=false) từ lưới nhỏ lên OW×OH ---------- */
  function upsample(src, fh, fw, OH, OW) {
    const out = new Float32Array(OH * OW);
    for (let y = 0; y < OH; y++) {
      const fy = Math.min(Math.max((y + 0.5) * fh / OH - 0.5, 0), fh - 1), y0 = Math.floor(fy), y1 = Math.min(y0 + 1, fh - 1), ty = fy - y0;
      for (let x = 0; x < OW; x++) {
        const fx = Math.min(Math.max((x + 0.5) * fw / OW - 0.5, 0), fw - 1), x0 = Math.floor(fx), x1 = Math.min(x0 + 1, fw - 1), tx = fx - x0;
        out[y * OW + x] = src[y0 * fw + x0] * (1 - tx) * (1 - ty) + src[y0 * fw + x1] * tx * (1 - ty) + src[y1 * fw + x0] * (1 - tx) * ty + src[y1 * fw + x1] * tx * ty;
      }
    }
    return out;
  }
  async function toPngUrl(imgData, w, h) {
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
    cv.getContext('2d').putImageData(imgData, 0, 0);
    const blob = await new Promise(r => cv.toBlob(r, 'image/png'));
    return URL.createObjectURL(blob);
  }

  /* ---------- engine ---------- */
  const E = { cfg: null, session: null, backend: null, rt: new Map(), KEEP: 12, resize, qualityReport, buildResult, THRESH_HIGH, THRESH_MID };

  E.setConfig = cfg => {
    E.cfg = cfg;
    const bin = atob(cfg.cls_w), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    E.clsW = new Float32Array(u8.buffer);                                   // (số nhãn × 1024)
  };

  E.loadModel = async function (bytes, useGpu) {
    ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2)) : 1;
    const base = { graphOptimizationLevel: 'all' };
    const tries = useGpu && navigator.gpu ? [['webgpu'], ['wasm']] : [['wasm']];
    let lastErr = null;
    for (const ep of tries) {
      try {
        E.session = await ort.InferenceSession.create(bytes, { ...base, executionProviders: ep });
        E.backend = ep[0] + (ep[0] === 'wasm' ? ` · ${ort.env.wasm.numThreads} luồng` : '');
        if (useGpu && ep[0] === 'wasm' && navigator.gpu) E.gpuFailed = true;
        return E.backend;
      } catch (e) { lastErr = e; console.warn('Không tạo được phiên với', ep, e); }
    }
    throw lastErr;
  };

  E.analyze = async function (blob) {
    if (!E.session) throw new Error('Mô hình chưa sẵn sàng.');
    const t0 = performance.now(), { H, W, cols, labels } = E.cfg;
    const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
    const sw = bmp.width, sh = bmp.height;
    const cv = document.createElement('canvas'); cv.width = sw; cv.height = sh;
    const cx = cv.getContext('2d', { willReadFrequently: true });
    cx.fillStyle = '#fff'; cx.fillRect(0, 0, sw, sh); cx.drawImage(bmp, 0, 0);
    const rgb = rgbaToRgb(cx.getImageData(0, 0, sw, sh).data);
    // letterbox nền trắng
    const s = Math.min(W / sw, H / sh), nw = Math.max(1, Math.round(sw * s)), nh = Math.max(1, Math.round(sh * s));
    const small = resize(rgb, sw, sh, nw, nh, 3), view = new Uint8ClampedArray(W * H * 3).fill(255);
    const ox = Math.floor((W - nw) / 2), oy = Math.floor((H - nh) / 2);
    for (let y = 0; y < nh; y++) view.set(small.subarray(y * nw * 3, (y + 1) * nw * 3), ((y + oy) * W + ox) * 3);
    // tensor CHW chuẩn hoá ImageNet
    const plane = W * H, x = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++) x[c * plane + i] = (view[3 * i + c] / 255 - MEAN[c]) / STD[c];
    const out = await E.session.run({ x: new ort.Tensor('float32', x, [1, 3, H, W]) });
    const z = out.logits.data, pf = out.paper_frac.data[0];
    const quality = qualityReport(rgb, sw, sh);
    // ảnh hiển thị
    const jid = Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
    const vc = document.createElement('canvas'); vc.width = W; vc.height = H;
    const vx = vc.getContext('2d'), id = vx.createImageData(W, H);
    for (let i = 0; i < plane; i++) { id.data[4 * i] = view[3 * i]; id.data[4 * i + 1] = view[3 * i + 1]; id.data[4 * i + 2] = view[3 * i + 2]; id.data[4 * i + 3] = 255; }
    vx.putImageData(id, 0, 0);
    const viewBlob = await new Promise(r => vc.toBlob(r, 'image/jpeg', 0.9));
    const rt = { feat: Float32Array.from(out.feat.data), fd: out.feat.dims, w32: Float32Array.from(out.w32.data), seg: Float32Array.from(out.seg.data), sd: out.seg.dims,
                 urls: { orig: URL.createObjectURL(blob), view: URL.createObjectURL(viewBlob) }, cam: new Map(), paper: null };
    E.rt.set(jid, rt);
    while (E.rt.size > E.KEEP) { const k = E.rt.keys().next().value, o = E.rt.get(k); Object.values(o.urls).forEach(URL.revokeObjectURL); o.cam.forEach(URL.revokeObjectURL); if (o.paper) URL.revokeObjectURL(o.paper); E.rt.delete(k); }
    const base = { id: jid, version: E.cfg.version, quality, paper_frac: +pf.toFixed(4), gate: E.cfg.gate, seconds: +((performance.now() - t0) / 1000).toFixed(2), t: Date.now() };
    if (pf < E.cfg.gate) return Object.assign(base, { rejected: true });
    const probs = {}; cols.forEach((c, i) => probs[c] = 1 / (1 + Math.exp(-z[i])));
    return Object.assign(base, { rejected: false }, buildResult(cols, probs, labels));
  };

  /* Grad-CAM của nhãn `key` -> URL ảnh PNG RGBA W×H (cùng khung với ảnh "mô hình thấy").
     Công thức giải tích cho đầu ra gộp có trọng số theo mặt nạ giấy (khớp ecgnetx.gradcam, sai số ~1e-6):
       alpha_c = W_jc · Σ_p[1(f_cp>0)·w_p] / (N·(Σw+eps));  cam = ReLU(Σ_c alpha_c·f_c) · w */
  E.cam = async function (jid, key) {
    const rt = E.rt.get(jid); if (!rt) return null;
    if (rt.cam.has(key)) return rt.cam.get(key);
    const j = E.cfg.cols.indexOf(key); if (j < 0) return null;
    const { H: OH, W: OW } = E.cfg, [, C, fh, fw] = rt.fd, N = fh * fw, A = rt.feat, w = rt.w32, Wt = E.clsW.subarray(j * C, (j + 1) * C);
    let sw = 0; for (let p = 0; p < N; p++) sw += w[p];
    const cam = new Float32Array(N);
    for (let c = 0; c < C; c++) {
      const o = c * N; let s = 0; for (let p = 0; p < N; p++) if (A[o + p] > 0) s += w[p];
      const al = Wt[c] * s / N / (sw + EPS); if (!al) continue;
      for (let p = 0; p < N; p++) cam[p] += al * A[o + p];
    }
    for (let p = 0; p < N; p++) cam[p] = Math.max(cam[p], 0) * w[p];
    const up = upsample(cam, fh, fw, OH, OW);
    let mx = 0; for (let p = 0; p < up.length; p++) { if (up[p] < 0) up[p] = 0; if (up[p] > mx) mx = up[p]; }
    const img = new ImageData(OW, OH), d = img.data, stops = [[255, 214, 10], [255, 138, 0], [200, 30, 40]];
    for (let p = 0; p < up.length; p++) {
      const c = mx > 0 ? up[p] / mx : 0, t = Math.min(Math.max(c * 2, 0), 2), i = Math.min(Math.floor(t), 1), f = t - i;
      const a = Math.pow(Math.min(Math.max((c - 0.18) / 0.82, 0), 1), 0.9) * 0.78, q = 4 * p;
      d[q] = stops[i][0] * (1 - f) + stops[i + 1][0] * f; d[q + 1] = stops[i][1] * (1 - f) + stops[i + 1][1] * f; d[q + 2] = stops[i][2] * (1 - f) + stops[i + 1][2] * f; d[q + 3] = a * 255;
    }
    const url = await toPngUrl(img, OW, OH); rt.cam.set(key, url); return url;
  };

  /* Vùng giấy ECG: làm tối phần ngoài giấy, viền mảnh quanh tờ giấy */
  E.paper = async function (jid) {
    const rt = E.rt.get(jid); if (!rt) return null;
    if (rt.paper) return rt.paper;
    const { H: OH, W: OW } = E.cfg, [, , sh, sw] = rt.sd, up = upsample(rt.seg, sh, sw, OH, OW);
    const inside = new Uint8Array(up.length); for (let p = 0; p < up.length; p++) inside[p] = up[p] >= 0.5 ? 1 : 0;
    const img = new ImageData(OW, OH), d = img.data;
    for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) {
      const p = y * OW + x, q = 4 * p;
      if (inside[p]) {
        const inner = y > 0 && y < OH - 1 && x > 0 && x < OW - 1 && inside[p - 1] && inside[p + 1] && inside[p - OW] && inside[p + OW];
        if (!inner) { d[q] = 20; d[q + 1] = 160; d[q + 2] = 160; d[q + 3] = 255; }
      } else { d[q] = 8; d[q + 1] = 16; d[q + 2] = 32; d[q + 3] = 120; }
    }
    rt.paper = await toPngUrl(img, OW, OH); return rt.paper;
  };

  global.Engine = E;
})(self);

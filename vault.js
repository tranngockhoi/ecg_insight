/* vault.js – tải mô hình đã mã hoá (model.enc), giải mã bằng mật khẩu, lưu trong máy (IndexedDB), kiểm tra hạn dùng.
   Định dạng model.enc: "ECGS1"(5) | lặp PBKDF2 uint32(4) | salt(16) | iv(12) | hạn dùng ms uint64(8) | AES-256-GCM(ciphertext+tag)
   45 byte đầu là dữ liệu xác thực (AAD), nên không sửa được ngày hết hạn mà không làm hỏng giải mã. */
(function (global) {
  'use strict';
  const MAGIC = 'ECGS1', HLEN = 45, URL_ENC = 'model.enc', DB = 'ecginsight', STORE = 'kv';

  const idb = () => new Promise((res, rej) => { const r = indexedDB.open(DB, 1); r.onupgradeneeded = () => r.result.createObjectStore(STORE); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  const kv = async (mode, fn) => { const db = await idb(); return new Promise((res, rej) => { const tx = db.transaction(STORE, mode), st = tx.objectStore(STORE), rq = fn(st); tx.oncomplete = () => res(rq && rq.result); tx.onerror = () => rej(tx.error); }); };
  const dbGet = k => kv('readonly', s => s.get(k)), dbSet = (k, v) => kv('readwrite', s => s.put(v, k)), dbDel = k => kv('readwrite', s => s.delete(k));

  function parseHeader(u8) {
    if (u8.length < HLEN) throw new Error('Tệp mô hình không hợp lệ.');
    for (let i = 0; i < 5; i++) if (u8[i] !== MAGIC.charCodeAt(i)) throw new Error('Tệp mô hình không đúng định dạng.');
    const dv = new DataView(u8.buffer, u8.byteOffset, HLEN);
    const iter = dv.getUint32(5), expiry = Number(dv.getBigUint64(37));
    const salt = u8.slice(9, 25), iv = u8.slice(25, 37);
    const ver = Array.from(salt.slice(0, 6)).map(b => b.toString(16).padStart(2, '0')).join('');
    return { iter, salt, iv, expiry, ver, aad: u8.slice(0, HLEN) };
  }

  async function fetchHeader(timeoutMs = 7000) {
    const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const r = await fetch(URL_ENC, { headers: { Range: 'bytes=0-44' }, cache: 'no-store', signal: ctl.signal });
      if (!r.ok && r.status !== 206) throw new Error('HTTP ' + r.status);
      const rd = r.body.getReader(), buf = new Uint8Array(HLEN); let n = 0;
      while (n < HLEN) { const { value, done } = await rd.read(); if (done) break; const take = Math.min(value.length, HLEN - n); buf.set(value.subarray(0, take), n); n += take; }
      try { ctl.abort(); } catch (e) { }
      return parseHeader(buf.subarray(0, n));
    } finally { clearTimeout(t); }
  }

  async function download(onProgress, expectMB) {
    const chunks = []; let got = 0;
    for (let part = 1; part <= 20; part++) {
      const name = part === 1 ? URL_ENC : 'model.p' + part;
      const r = await fetch(name, { cache: 'no-store' });
      if (!r.ok) { if (part === 1) throw new Error('Không tải được mô hình (HTTP ' + r.status + ').'); break; }
      const rd = r.body.getReader();
      for (; ;) { const { value, done } = await rd.read(); if (done) break; chunks.push(value); got += value.length; onProgress && onProgress(expectMB ? Math.min(0.99, got / (expectMB * 1e6)) : 0, got, expectMB * 1e6); }
      if (r.headers.get('content-type') && /text\/html/.test(r.headers.get('content-type'))) throw new Error('Phần mô hình bị thay bằng trang web (404).');
    }
    const all = new Uint8Array(got); let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
    return all;
  }

  async function decrypt(u8, password) {
    const h = parseHeader(u8);
    if (Date.now() > h.expiry) { const e = new Error('Bản mô hình này đã hết hạn.'); e.code = 'expired'; throw e; }
    const km = await crypto.subtle.importKey('raw', new TextEncoder().encode(password), 'PBKDF2', false, ['deriveKey']);
    const key = await crypto.subtle.deriveKey({ name: 'PBKDF2', salt: h.salt, iterations: h.iter, hash: 'SHA-256' }, km, { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
    try {
      const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: h.iv, additionalData: h.aad }, key, u8.subarray(HLEN));
      return { bytes: new Uint8Array(plain), header: h };
    } catch (e) { const er = new Error('Sai mật khẩu.'); er.code = 'badpass'; throw er; }
  }

  /* Trạng thái khởi động:
       {state:'ready', bytes, expiry, ver, offline}      -> dùng ngay
       {state:'need-password', header}                    -> cần nhập mật khẩu + tải
       {state:'expired'}                                  -> tệp trên máy chủ đã hết hạn
       {state:'offline-empty'}                            -> chưa có mô hình và không có mạng */
  async function status() {
    let stored = null;
    try { stored = await dbGet('model'); } catch (e) { }
    if (stored && Date.now() > stored.expiry) { try { await dbDel('model'); } catch (e) { } stored = null; }
    let head = null, netErr = null;
    try { head = await fetchHeader(); } catch (e) { netErr = e; }
    if (head) {
      if (Date.now() > head.expiry) { try { await dbDel('model'); } catch (e) { } return { state: 'expired', header: head }; }
      if (stored && stored.ver === head.ver) return { state: 'ready', bytes: stored.bytes, expiry: stored.expiry, ver: stored.ver, offline: false };
      return { state: 'need-password', header: head };
    }
    if (stored) return { state: 'ready', bytes: stored.bytes, expiry: stored.expiry, ver: stored.ver, offline: true };
    return { state: 'offline-empty', error: netErr };
  }

  async function unlock(password, onProgress, expectMB) {
    const enc = await download(onProgress, expectMB);
    onProgress && onProgress(1, 0, 0, 'decrypt');
    const { bytes, header } = await decrypt(enc, password);
    try { await dbSet('model', { ver: header.ver, expiry: header.expiry, bytes }); } catch (e) { console.warn('Không lưu được mô hình vào máy', e); }
    return { bytes, expiry: header.expiry, ver: header.ver };
  }

  const forget = async () => { try { await dbDel('model'); } catch (e) { } };
  global.Vault = { status, unlock, forget, parseHeader, HLEN };
})(self);

/* sw.js – chạy offline + bật "cách ly origin" (COOP/COEP) để onnxruntime dùng được nhiều luồng trên GitHub Pages */
const V = 'ecginsight-v1', SHELL = ['./', 'index.html', 'engine.js', 'vault.js', 'config.json', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png'];
self.addEventListener('install', e => { e.waitUntil(caches.open(V).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
function iso(r) {
  if (!r || r.status === 0 || r.type === 'opaque') return r;
  const h = new Headers(r.headers); h.set('Cross-Origin-Opener-Policy', 'same-origin'); h.set('Cross-Origin-Embedder-Policy', 'require-corp'); h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
}
self.addEventListener('fetch', e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin) return;
  if (/\/model\.(enc|p\d+)$/.test(url.pathname)) { e.respondWith(fetch(req).then(iso)); return; }
  if (url.pathname.includes('/ort/')) {
    e.respondWith(caches.open(V).then(async c => { const m = await c.match(req); if (m) return iso(m); const r = await fetch(req); if (r.ok) c.put(req, r.clone()); return iso(r); }));
    return;
  }
  e.respondWith(fetch(req, { cache: 'no-cache' }).then(async r => { if (r.ok) { const c = await caches.open(V); c.put(req, r.clone()); } return iso(r); })
    .catch(async () => iso((await caches.match(req, { ignoreSearch: true })) || (req.mode === 'navigate' ? await caches.match('index.html') : undefined) || Response.error())));
});

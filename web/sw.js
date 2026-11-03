// Offline shell for clinics with unreliable connections between the screening laptop and the server.
// Network first (so an upgrade is picked up immediately), falling back to the cached shell.
// API calls and images are never cached: no patient data is written to the browser's cache.
const CACHE = "clinictune-shell-v2";
const SHELL = [
  "./", "index.html", "styles.css", "app.js", "manifest.webmanifest", "icon.svg",
  "fonts/fonts.css", "fonts/anek-latin-latin.woff2", "fonts/anek-latin-latin-ext.woff2",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin || url.pathname.startsWith("/v1/")) return;
  event.respondWith(
    fetch(event.request)
      .then((res) => {
        if (res.ok && (SHELL.some((p) => url.pathname.endsWith(p.replace("./", "/"))) || url.pathname.startsWith("/fonts/"))) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(event.request, copy));
        }
        return res;
      })
      .catch(() => caches.match(event.request).then((hit) => hit || caches.match("index.html"))),
  );
});

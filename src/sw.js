// Service worker template; vite.config.ts fills in the version and shell list
// at build time. Model weights are cached by the inference worker itself.
const VERSION = "__VERSION__";
const SHELL = __SHELL__;
const SHELL_CACHE = `depth-light-shell-${VERSION}`;
// Runtime files under /models/ carry their version in the name, so this cache
// survives deploys and only drops entries that a newer file replaced.
const RUNTIME_CACHE = "depth-light-runtime";

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("depth-light-shell-") && k !== SHELL_CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

async function cacheRuntime(request, response) {
  const cache = await caches.open(RUNTIME_CACHE);
  const stem = (url) => new URL(url).pathname.replace(/-[\d.]+(\.\w+)$/, "$1");
  for (const old of await cache.keys()) if (stem(old.url) === stem(request.url) && old.url !== request.url) await cache.delete(old);
  await cache.put(request, response);
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;
  if (url.pathname.endsWith(".onnx")) return;

  if (request.mode === "navigate") {
    // Network first so a deploy shows up at once; the cached shell offline.
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(SHELL_CACHE).then((cache) => cache.put("/", copy));
          return response;
        })
        .catch(() => caches.match("/")),
    );
    return;
  }

  event.respondWith(
    caches.match(request).then(
      (hit) =>
        hit ||
        fetch(request).then((response) => {
          if (response.ok && url.pathname.startsWith("/models/")) event.waitUntil(cacheRuntime(request, response.clone()));
          return response;
        }),
    ),
  );
});

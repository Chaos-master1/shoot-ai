var CACHE = 'KAWKAB_CACHE_V2';
var STATIC_ASSETS = [
  'index.html',
  'offline.html',
  'css/main.css',
  'css/accessibility.css',
  'js/kawkab_polish.js',
  'js/app.js',
  'js/app-offline.js',
  'js/app-router.js',
  'js/app-skeletons.js',
  'js/app-perf.js',
  'js/app-ux.js',
  'js/app-charts.js',
  'js/app-coding.js',
  'js/app-tactics.js',
  'js/app-squad.js',
  'js/app-ai.js',
  'js/app-scout.js',
  'js/app-sparklines.js',
  'js/app-data-providers.js',
  'js/app-search-compare.js',
  'js/app-tooltips.js',
  'js/app-opponent.js',
  'js/app-marketplace.js',
  'js/app-error-boundary.js',
  'manifest.json',
  'icons/icon-192.svg',
  'icons/icon-512.svg'
];

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(CACHE).then(function (cache) {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(
        keys.filter(function (k) { return k !== CACHE && k.startsWith('KAWKAB_CACHE_'); })
          .map(function (k) { return caches.delete(k); })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', function (event) {
  var url = new URL(event.request.url);
  var isStatic = /\.(css|js|png|jpg|jpeg|gif|svg|woff2?|ttf|eot|ico)$/i.test(url.pathname);
  var isApi = url.pathname.indexOf('/bridge/') !== -1 || url.pathname.indexOf('/api/') !== -1 || url.pathname.indexOf('/data/') !== -1;
  var isNavigation = event.request.mode === 'navigate';

  if (isStatic) {
    event.respondWith(
      caches.match(event.request).then(function (cached) {
        return cached || fetch(event.request).then(function (response) {
          return caches.open(CACHE).then(function (cache) {
            cache.put(event.request, response.clone());
            return response;
          });
        });
      })
    );
    return;
  }

  if (isApi) {
    event.respondWith(
      fetch(event.request).then(function (response) {
        return caches.open(CACHE).then(function (cache) {
          cache.put(event.request, response.clone());
          return response;
        });
      }).catch(function () {
        return caches.match(event.request);
      })
    );
    return;
  }

  if (isNavigation) {
    event.respondWith(
      fetch(event.request).catch(function () {
        return caches.match('offline.html');
      })
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then(function (cached) {
      return cached || fetch(event.request);
    })
  );
});

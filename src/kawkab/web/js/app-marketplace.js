    // ── Phase 15 — Community Marketplace ──

    // Not wrapped in its own IIFE (split out of app.js without one) -- the
    // bundler is a plain concatenation, so this and every escapeHtml(...)
    // call below resolved to nothing at runtime (ReferenceError) until this
    // import was added. Matches app-data-providers.js's own fix for the
    // same gap.
    var escapeHtml = window.__kawkab.escapeHtml;

    function initMarketplace() {
        var currentType = 'drill';

        function switchMpTab(tabId) {
            currentType = tabId.replace('mp-', '');
            document.querySelectorAll('#marketplace-tabs .tab').forEach(function(t) { t.classList.remove('active'); });
            var tab = document.querySelector('#marketplace-tabs .tab[data-mp-tab="' + tabId + '"]');
            if (tab) tab.classList.add('active');
            loadCategories(currentType);
            loadItems();
        }

        document.querySelectorAll('#marketplace-tabs .tab').forEach(function(tab) {
            tab.addEventListener('click', function() { switchMpTab(this.dataset.mpTab); });
        });

        function loadCategories(itemType) {
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.marketplace_categories(itemType, function(result) {
                try {
                    var data = JSON.parse(result);
                    if (!data.success) return;
                    var sel = document.getElementById('mp-category-select');
                    sel.innerHTML = '<option value="">All categories</option>';
                    (data.categories || []).forEach(function(c) {
                        sel.innerHTML += '<option value="' + escapeHtml(c) + '">' + escapeHtml(c) + '</option>';
                    });
                } catch(e) {}
            });
        }

        function loadItems() {
            if (typeof bridge === 'undefined' || !bridge) return;
            var query = document.getElementById('mp-search-input').value;
            var category = document.getElementById('mp-category-select').value;
            bridge.marketplace_list(currentType, category, query, '', function(result) {
                try {
                    var data = JSON.parse(result);
                    if (!data.success) return;
                    var grid = document.getElementById('mp-items-grid');
                    grid.innerHTML = '';
                    (data.items || []).forEach(function(item) {
                        var card = document.createElement('div');
                        card.className = 'pro-card';
                        card.style.cursor = 'pointer';
                        var icon = item.item_type === 'drill' ? '🏃' : item.item_type === 'template' ? '📐' : '🔌';
                        card.innerHTML = '<div style="font-weight:700">' + icon + ' ' + escapeHtml(item.name) + '</div>' +
                            '<div style="font-size:0.75rem;color:var(--text-muted);margin:2px 0">' +
                            escapeHtml(item.description) + '</div>' +
                            '<div style="display:flex;gap:8px;font-size:0.7rem;color:var(--text-muted)">' +
                            '<span>⭐ ' + item.rating + '</span>' +
                            '<span>⬇ ' + item.download_count + '</span>' +
                            (item.category ? '<span>📂 ' + escapeHtml(item.category) + '</span>' : '') +
                            '</div>';
                        card.addEventListener('click', function() { showItemDetail(item.id); });
                        grid.appendChild(card);
                    });
                    if (!data.items || data.items.length === 0) {
                        grid.innerHTML = '<p class="hint">No items found. Be the first to submit!</p>';
                    }
                } catch(e) {}
            });
        }

        function showItemDetail(itemId) {
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.marketplace_get(itemId, function(result) {
                try {
                    var data = JSON.parse(result);
                    if (!data.success) return;
                    var item = data.item;
                    document.getElementById('mp-detail-modal').classList.remove('hidden');
                    document.getElementById('mp-detail-modal').style.display = 'flex';
                    document.getElementById('mp-detail-name').textContent = item.name;
                    document.getElementById('mp-detail-type').textContent = item.item_type + ' | ' + (item.category || 'General');
                    document.getElementById('mp-detail-body').textContent = item.description;
                    document.getElementById('mp-detail-rating').textContent = '⭐ ' + item.rating + ' | ⬇ ' + item.download_count + ' downloads | By: ' + (item.author || 'Community');
                    var dataEl = document.getElementById('mp-detail-data');
                    try {
                        var parsed = JSON.parse(item.data || '{}');
                        dataEl.textContent = JSON.stringify(parsed, null, 2);
                    } catch(e) {
                        dataEl.textContent = item.data || 'No data';
                    }
                    var delBtn = document.getElementById('mp-detail-delete-btn');
                    if (item.source === 'local') {
                        delBtn.classList.remove('hidden');
                        delBtn.onclick = function() {
                            if (!confirm('Delete this item?')) return;
                            bridge.marketplace_delete(itemId, function() {
                                document.getElementById('mp-detail-modal').classList.add('hidden');
                                document.getElementById('mp-detail-modal').style.display = '';
                                loadItems();
                                showToast('Deleted', 'info');
                            });
                        };
                    } else {
                        delBtn.classList.add('hidden');
                    }
                    document.getElementById('mp-download-btn').onclick = function() {
                        bridge.marketplace_rate(itemId, '5.0', function() {
                            showToast('Downloaded! Thanks for your interest.', 'success');
                            loadItems();
                        });
                    };
                } catch(e) {}
            });
        }

        document.getElementById('mp-detail-close').onclick = function() {
            document.getElementById('mp-detail-modal').classList.add('hidden');
            document.getElementById('mp-detail-modal').style.display = '';
        };
        document.getElementById('mp-search-btn').onclick = loadItems;
        document.getElementById('mp-search-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') loadItems();
        });
        document.getElementById('mp-category-select').onchange = loadItems;
        document.getElementById('mp-add-btn').onclick = function() {
            var name = prompt('Item name:');
            if (!name) return;
            var desc = prompt('Description:') || '';
            var category = prompt('Category (e.g. possession, finishing, defense, formation):') || '';
            bridge.marketplace_add(currentType, name, desc, 'Local User', category, '[]', '{}', 'local', function() {
                loadItems();
                showToast('Item submitted!', 'success');
            });
        };

        // Load stats
        if (typeof bridge !== 'undefined' && bridge) {
            bridge.marketplace_stats(function(result) {
                try {
                    var data = JSON.parse(result);
                    if (data.success && data.stats) {
                        document.getElementById('marketplace-stats').textContent =
                            data.stats.total + ' items (' + data.stats.drills + ' drills, ' + data.stats.templates + ' templates, ' + data.stats.plugins + ' plugins)';
                    }
                } catch(e) {}
            });
            loadCategories('drill');
            loadItems();
        }
    }


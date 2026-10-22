import * as THREE from './lib/three.module.min.js';
import { OrbitControls } from './lib/OrbitControls.js';

(function () {
    'use strict';

    var _instance = null;
    var _matchId = null;
    var _currentTime = 0;
    var _speed = 1;
    var _mode = 'dots';
    var _showHome = true;
    var _showAway = true;
    var _playerNameMap = {};
    var _pendingUpdate = false;

    var PITCH_W = 105;
    var PITCH_H = 68;
    var HOME_COLOR = 0x2563eb;
    var AWAY_COLOR = 0xdc2626;
    var BALL_COLOR = 0xffffff;
    var PITCH_COLOR = 0x2d7d3a;
    var LINE_COLOR = 0xffffff;
    var LINE_OPACITY = 0.6;

    function Kawkab3DPitch(containerId) {
        if (_instance) { _instance.dispose(); }
        _instance = this;

        this.container = document.getElementById(containerId);
        if (!this.container) { console.warn('3D pitch container not found:', containerId); return; }

        this.matchId = null;
        this.clock = new THREE.Clock();
        this.clock.stop();
        this.animating = false;
        this.players = {};
        this.playerLabels = {};
        this.ballMesh = null;
        this.ballTrail = [];
        this.lastTimestamp = -1;

        this._setupScene();
        this._setupLights();
        this._setupPitch();
        this._setupControls();
        this._setupResize();

        var self = this;
        this._animate = function () {
            if (!self.animating) return;
            if (!document.contains(self.container)) { self.animating = false; return; }
            requestAnimationFrame(self._animate);
            self.controls.update();
            self.renderer.render(self.scene, self.camera);
        };
    }

    Kawkab3DPitch.prototype._setupScene = function () {
        var rect = this.container.getBoundingClientRect();
        var w = rect.width || 800;
        var h = rect.height || 500;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        this.camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 200);
        this.camera.position.set(0, 55, 60);
        this.camera.lookAt(0, 0, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        this.container.appendChild(this.renderer.domElement);
    };

    Kawkab3DPitch.prototype._setupLights = function () {
        var ambient = new THREE.AmbientLight(0x404060, 0.6);
        this.scene.add(ambient);

        var dir = new THREE.DirectionalLight(0xffffff, 1.2);
        dir.position.set(50, 80, 30);
        dir.castShadow = true;
        dir.shadow.mapSize.width = 2048;
        dir.shadow.mapSize.height = 2048;
        dir.shadow.camera.near = 1;
        dir.shadow.camera.far = 200;
        dir.shadow.camera.left = -60;
        dir.shadow.camera.right = 60;
        dir.shadow.camera.top = 60;
        dir.shadow.camera.bottom = -60;
        this.scene.add(dir);

        var fill = new THREE.DirectionalLight(0x8888ff, 0.4);
        fill.position.set(-30, 20, -30);
        this.scene.add(fill);

        var hemi = new THREE.HemisphereLight(0x87ceeb, 0x362d1e, 0.3);
        this.scene.add(hemi);
    };

    Kawkab3DPitch.prototype._setupPitch = function () {
        // Pitch surface
        var pitchGeo = new THREE.PlaneGeometry(PITCH_W, PITCH_H);
        var pitchMat = new THREE.MeshStandardMaterial({
            color: PITCH_COLOR,
            roughness: 0.8,
            metalness: 0.0,
            side: THREE.DoubleSide,
        });
        var pitch = new THREE.Mesh(pitchGeo, pitchMat);
        pitch.rotation.x = -Math.PI / 2;
        pitch.position.y = -0.05;
        pitch.receiveShadow = true;
        this.scene.add(pitch);

        // Pitch markings
        this._drawLine(-PITCH_W / 2, -PITCH_H / 2, PITCH_W / 2, -PITCH_H / 2); // bottom touch
        this._drawLine(-PITCH_W / 2, PITCH_H / 2, PITCH_W / 2, PITCH_H / 2);   // top touch
        this._drawLine(-PITCH_W / 2, -PITCH_H / 2, -PITCH_W / 2, PITCH_H / 2); // left goal
        this._drawLine(PITCH_W / 2, -PITCH_H / 2, PITCH_W / 2, PITCH_H / 2);   // right goal

        // Center line
        this._drawLine(0, -PITCH_H / 2, 0, PITCH_H / 2);

        // Center circle
        var circlePoints = [];
        for (var a = 0; a <= Math.PI * 2; a += 0.05) {
            circlePoints.push(new THREE.Vector3(9.15 * Math.cos(a), 0.02, 9.15 * Math.sin(a)));
        }
        var circleGeo = new THREE.BufferGeometry().setFromPoints(circlePoints);
        var circleMat = new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: LINE_OPACITY });
        var circle = new THREE.Line(circleGeo, circleMat);
        circle.position.y = 0;
        this.scene.add(circle);

        // Center dot
        var dotGeo = new THREE.CircleGeometry(0.3, 16);
        var dotMat = new THREE.MeshBasicMaterial({ color: LINE_COLOR });
        var dot = new THREE.Mesh(dotGeo, dotMat);
        dot.rotation.x = -Math.PI / 2;
        dot.position.y = 0.01;
        this.scene.add(dot);

        // Penalty areas
        this._drawRect(-PITCH_W / 2, -20.15, -PITCH_W / 2 + 16.5, 20.15); // left
        this._drawRect(PITCH_W / 2 - 16.5, -20.15, PITCH_W / 2, 20.15);    // right

        // Six-yard boxes
        this._drawRect(-PITCH_W / 2, -9.15, -PITCH_W / 2 + 5.5, 9.15);    // left
        this._drawRect(PITCH_W / 2 - 5.5, -9.15, PITCH_W / 2, 9.15);       // right

        // Goals
        this._drawGoal(-PITCH_W / 2 - 0.3, -3.66, -PITCH_W / 2 + 0.3, 3.66); // left
        this._drawGoal(PITCH_W / 2 - 0.3, -3.66, PITCH_W / 2 + 0.3, 3.66);   // right

        // Ball
        var ballGeo = new THREE.SphereGeometry(0.4, 12, 12);
        var ballMat = new THREE.MeshStandardMaterial({
            color: BALL_COLOR,
            roughness: 0.3,
            metalness: 0.1,
        });
        this.ballMesh = new THREE.Mesh(ballGeo, ballMat);
        this.ballMesh.position.set(0, 0.4, 0);
        this.ballMesh.castShadow = true;
        this.scene.add(this.ballMesh);
    };

    Kawkab3DPitch.prototype._drawLine = function (x1, z1, x2, z2) {
        var points = [new THREE.Vector3(x1, 0.01, z1), new THREE.Vector3(x2, 0.02, z2)];
        var geo = new THREE.BufferGeometry().setFromPoints(points);
        var mat = new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: LINE_OPACITY });
        var line = new THREE.Line(geo, mat);
        this.scene.add(line);
    };

    Kawkab3DPitch.prototype._drawRect = function (x1, z1, x2, z2) {
        var pts = [
            new THREE.Vector3(x1, 0.01, z1), new THREE.Vector3(x2, 0.01, z1),
            new THREE.Vector3(x2, 0.01, z2), new THREE.Vector3(x1, 0.01, z2),
            new THREE.Vector3(x1, 0.01, z1),
        ];
        var geo = new THREE.BufferGeometry().setFromPoints(pts);
        var mat = new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: LINE_OPACITY * 0.8 });
        var line = new THREE.Line(geo, mat);
        this.scene.add(line);
    };

    Kawkab3DPitch.prototype._drawGoal = function (x1, z1, x2, z2) {
        var pts = [
            new THREE.Vector3(x1, 0.01, z1), new THREE.Vector3(x1, 2.44, z1),
            new THREE.Vector3(x1, 2.44, z2), new THREE.Vector3(x1, 0.01, z2),
            new THREE.Vector3(x2, 0.01, z2), new THREE.Vector3(x2, 2.44, z2),
            new THREE.Vector3(x2, 2.44, z1), new THREE.Vector3(x2, 0.01, z1),
            new THREE.Vector3(x2, 2.44, z1), new THREE.Vector3(x1, 2.44, z1),
        ];
        var geo = new THREE.BufferGeometry().setFromPoints(pts);
        var mat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 });
        var line = new THREE.Line(geo, mat);
        this.scene.add(line);
    };

    Kawkab3DPitch.prototype._setupControls = function () {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.maxPolarAngle = Math.PI / 2.2;
        this.controls.minDistance = 15;
        this.controls.maxDistance = 120;
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    };

    Kawkab3DPitch.prototype._setupResize = function () {
        var self = this;
        var observer = new ResizeObserver(function () {
            if (!self.container || !self.renderer) return;
            var rect = self.container.getBoundingClientRect();
            var w = rect.width || 800;
            var h = rect.height || 500;
            self.camera.aspect = w / h;
            self.camera.updateProjectionMatrix();
            self.renderer.setSize(w, h);
        });
        observer.observe(this.container);
        this._resizeObserver = observer;
    };

    Kawkab3DPitch.prototype.setMatchId = function (matchId) {
        this.matchId = matchId;
    };

    Kawkab3DPitch.prototype.setSpeed = function (speed) {
        _speed = speed;
        this.clock.setDelta(this.clock.getDelta());
    };

    Kawkab3DPitch.prototype.setMode = function (mode) {
        _mode = mode;
        this._rebuildPlayers();
    };

    Kawkab3DPitch.prototype.showTeam = function (team, visible) {
        if (team === 'home') _showHome = visible;
        if (team === 'away') _showAway = visible;
        for (var id in this.players) {
            var p = this.players[id];
            if (p.userData.team === 'h') p.visible = _showHome;
            if (p.userData.team === 'a') p.visible = _showAway;
        }
    };

    Kawkab3DPitch.prototype.start = function () {
        if (this.animating) return;
        this.animating = true;
        this.clock.start();
        this._animate();
    };

    Kawkab3DPitch.prototype.stop = function () {
        this.animating = false;
        this.clock.stop();
    };

    Kawkab3DPitch.prototype.updateScene = function (timestamp) {
        if (!this.animating || !this.matchId) {
            this._queueUpdate(timestamp);
            return;
        }
        this._fetchAndRender(timestamp);
    };

    Kawkab3DPitch.prototype._queueUpdate = function (timestamp) {
        var self = this;
        self._pendingTimestamp = timestamp;
        if (!self._pendingTimer) {
            self._pendingTimer = setTimeout(function () {
                self._pendingTimer = null;
                if (self.animating && self.matchId && self._pendingTimestamp != null) {
                    self._fetchAndRender(self._pendingTimestamp);
                }
            }, 200);
        }
    };

    Kawkab3DPitch.prototype._fetchAndRender = function (timestamp) {
        if (_pendingUpdate) return;
        _pendingUpdate = true;
        _currentTime = timestamp;
        var self = this;
        var bridge = (window.KawkabUtils && KawkabUtils.getBridge) ? KawkabUtils.getBridge() : (window.bridge || window.kawkabBridge);
        if (!bridge || !self.matchId) { _pendingUpdate = false; return; }

        bridge.get_overlay_data(self.matchId, timestamp).then(function (json) {
            _pendingUpdate = false;
            if (!json) return;
            try {
                var data = typeof json === 'string' ? JSON.parse(json) : json;
                if (data && (data.p || data.players)) {
                    self._updatePositions(data.p || data.players, data.b || data.ball, timestamp);
                }
            } catch (e) {
                console.warn('3D overlay parse error:', e);
            }
        }).catch(function () {
            _pendingUpdate = false;
        });
    };

    Kawkab3DPitch.prototype._updatePositions = function (players, ball, timestamp) {
        if (!players) return;

        // Track which IDs we saw this frame
        var activeIds = {};

        players.forEach(function (p) {
            var id = p.i || p.track_id || p.id;
            if (id == null) return;
            activeIds[id] = true;

            // Convert normalized 0-1 coords to pitch coords
            var x = (p.x - 0.5) * PITCH_W;
            var z = (p.y - 0.5) * PITCH_H;

            if (!_instance.players[id]) {
                _instance._createPlayer(id, x, z, p.m || 'u', p.name || '');
            }
            var mesh = _instance.players[id];
            mesh.position.x += (x - mesh.position.x) * 0.3;
            mesh.position.z += (z - mesh.position.z) * 0.3;
            mesh.userData.team = p.m || 'u';
            mesh.visible = (p.m === 'h' && _showHome) || (p.m === 'a' && _showAway) || (p.m !== 'h' && p.m !== 'a');
        });

        // Remove players not in current frame (within reason — keep buffer)
        for (var id in _instance.players) {
            if (!activeIds[id]) {
                if (_instance.players[id].userData.missedFrames == null) {
                    _instance.players[id].userData.missedFrames = 0;
                }
                _instance.players[id].userData.missedFrames++;
                if (_instance.players[id].userData.missedFrames > 30) {
                    _instance._removePlayer(id);
                }
            } else {
                if (_instance.players[id]) {
                    _instance.players[id].userData.missedFrames = 0;
                }
            }
        }

        // Update ball
        if (ball && _instance.ballMesh) {
            var bx = (ball.x - 0.5) * PITCH_W;
            var bz = (ball.y - 0.5) * PITCH_H;
            _instance.ballMesh.position.x += (bx - _instance.ballMesh.position.x) * 0.4;
            _instance.ballMesh.position.z += (bz - _instance.ballMesh.position.z) * 0.4;
            _instance.ballMesh.position.y = 0.4;

            // Ball trail (3D)
            _instance.ballTrail.push(_instance.ballMesh.position.clone());
            if (_instance.ballTrail.length > 20) _instance.ballTrail.shift();
            // Render ball trail as small spheres
            _instance._renderBallTrail();
        }
    };

    Kawkab3DPitch.prototype._createPlayer = function (id, x, z, team, name) {
        var isHome = team === 'h';
        var color = isHome ? HOME_COLOR : (team === 'a' ? AWAY_COLOR : 0x9ca3af);

        if (_mode === 'cards') {
            this._createPlayerCard(id, x, z, team, name, color);
        } else {
            this._createPlayerDot(id, x, z, team, name, color);
        }
    };

    Kawkab3DPitch.prototype._createPlayerDot = function (id, x, z, team, name, color) {
        var geo = new THREE.SphereGeometry(0.6, 12, 12);
        var mat = new THREE.MeshStandardMaterial({
            color: color,
            roughness: 0.4,
            metalness: 0.2,
            emissive: color,
            emissiveIntensity: 0.1,
        });
        var mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x, 0.6, z);
        mesh.castShadow = true;
        mesh.userData = { id: id, team: team, name: name || '', isCard: false, missedFrames: 0 };
        this.scene.add(mesh);
        this.players[id] = mesh;

        // Label sprite (jersey number / name)
        this._createLabel(id, mesh, name);
    };

    Kawkab3DPitch.prototype._createPlayerCard = function (id, x, z, team, name, color) {
        var isHome = team === 'h';
        var canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 160;
        var ctx = canvas.getContext('2d');

        // Card background
        var grad = ctx.createLinearGradient(0, 0, 0, 160);
        if (isHome) {
            grad.addColorStop(0, '#1e40af');
            grad.addColorStop(0.6, '#2563eb');
            grad.addColorStop(1, '#1e3a5f');
        } else if (team === 'a') {
            grad.addColorStop(0, '#991b1b');
            grad.addColorStop(0.6, '#dc2626');
            grad.addColorStop(1, '#7f1d1d');
        } else {
            grad.addColorStop(0, '#4b5563');
            grad.addColorStop(0.6, '#6b7280');
            grad.addColorStop(1, '#374151');
        }
        ctx.fillStyle = grad;
        ctx.roundRect(4, 4, 120, 152, 8);
        ctx.fill();

        // Border
        ctx.strokeStyle = isHome ? '#60a5fa' : (team === 'a' ? '#f87171' : '#9ca3af');
        ctx.lineWidth = 2;
        ctx.roundRect(4, 4, 120, 152, 8);
        ctx.stroke();

        // Rating circle
        var rating = _playerNameMap[id] && _playerNameMap[id].rating != null ? _playerNameMap[id].rating : 75;
        ctx.beginPath();
        ctx.arc(64, 32, 18, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 20px Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(rating.toString(), 64, 32);

        // Player photo area (silhouette if no photo)
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.beginPath();
        ctx.arc(64, 85, 30, 0, Math.PI * 2);
        ctx.fill();

        // Player name
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 11px Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        var displayName = name || '#' + id;
        if (displayName.length > 10) displayName = displayName.substring(0, 9) + '…';
        ctx.fillText(displayName, 64, 140);

        // Jersey number
        ctx.font = 'bold 10px Arial, sans-serif';
        ctx.textBaseline = 'top';
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.fillText('#' + id, 64, 142);

        // Stat bars at bottom
        var stats = _playerNameMap[id] && _playerNameMap[id].stats ? _playerNameMap[id].stats : { pac: 50, sho: 50, pas: 50, dri: 50, def: 50, phy: 50 };
        var statLabels = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY'];
        var statKeys = ['pac', 'sho', 'pas', 'dri', 'def', 'phy'];
        for (var s = 0; s < 6; s++) {
            var sy = 112 + s * 7;
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.font = '6px Arial, sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(statLabels[s], 12, sy + 1);

            var val = Math.min(99, Math.max(0, stats[statKeys[s]] || 50));
            ctx.fillStyle = 'rgba(255,255,255,0.1)';
            ctx.fillRect(40, sy, 70, 4);
            ctx.fillStyle = val > 70 ? '#22c55e' : (val > 45 ? '#eab308' : '#ef4444');
            ctx.fillRect(40, sy, 70 * (val / 99), 4);

            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.font = '5px Arial, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(val.toString(), 112, sy + 1);
        }

        var texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;

        var material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthWrite: false,
            sizeAttenuation: true,
        });

        var sprite = new THREE.Sprite(material);
        sprite.position.set(x, 4, z);
        sprite.scale.set(4, 5, 1);
        sprite.userData = { id: id, team: team, name: name || '', isCard: true, missedFrames: 0 };
        this.scene.add(sprite);
        this.players[id] = sprite;
    };

    Kawkab3DPitch.prototype._createLabel = function (id, parentMesh, name) {
        var canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 32;
        var ctx = canvas.getContext('2d');

        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.roundRect(4, 4, 56, 24, 4);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        var label = name || '#' + id;
        if (label.length > 6) label = label.substring(0, 5) + '…';
        ctx.fillText(label, 32, 16);

        var texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;

        var material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthWrite: false,
            sizeAttenuation: true,
        });

        var sprite = new THREE.Sprite(material);
        sprite.position.set(0, 1.6, 0);
        sprite.scale.set(2, 1, 1);
        parentMesh.add(sprite);
    };

    Kawkab3DPitch.prototype._removePlayer = function (id) {
        var mesh = this.players[id];
        if (!mesh) return;
        this.scene.remove(mesh);
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) mesh.material.dispose();
        delete this.players[id];
    };

    Kawkab3DPitch.prototype._rebuildPlayers = function () {
        // Remove all players
        for (var id in this.players) {
            this._removePlayer(id);
        }
        this.players = {};
        // Re-fetch current frame
        if (_currentTime > 0) {
            this._fetchAndRender(_currentTime);
        }
    };

    Kawkab3DPitch.prototype._renderBallTrail = function () {
        // Remove old trail
        if (this._trailMeshes) {
            this._trailMeshes.forEach(function (m) {
                _instance.scene.remove(m);
                if (m.geometry) m.geometry.dispose();
                if (m.material) m.material.dispose();
            });
        }
        this._trailMeshes = [];

        var trail = this.ballTrail;
        for (var i = 0; i < trail.length; i++) {
            var alpha = (i / trail.length) * 0.6;
            var geo = new THREE.SphereGeometry(0.15, 6, 6);
            var mat = new THREE.MeshBasicMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: alpha,
            });
            var mesh = new THREE.Mesh(geo, mat);
            mesh.position.copy(trail[i]);
            mesh.position.y = 0.2;
            this._trailMeshes.push(mesh);
            this.scene.add(mesh);
        }
    };

    Kawkab3DPitch.prototype.setPlayerNameMap = function (nameMap) {
        _playerNameMap = nameMap || {};
    };

    Kawkab3DPitch.prototype.dispose = function () {
        this.stop();
        if (this._resizeObserver) this._resizeObserver.disconnect();
        if (this.controls) this.controls.dispose();
        if (this.renderer) {
            this.renderer.dispose();
            if (this.renderer.domElement && this.container) {
                this.container.removeChild(this.renderer.domElement);
            }
        }
        for (var id in this.players) this._removePlayer(id);
        if (this.ballMesh) {
            this.scene.remove(this.ballMesh);
            if (this.ballMesh.geometry) this.ballMesh.geometry.dispose();
            if (this.ballMesh.material) this.ballMesh.material.dispose();
        }
        _instance = null;
    };

    // Expose globally
    window.Kawkab3DPitch = Kawkab3DPitch;

    // Init function called from app.js
    window.init3DPitch = function () {
        var pitch = new Kawkab3DPitch('pitch3d-container');
        if (!pitch.renderer) return;
        pitch.start();

        // Wire up controls
        var matchSelect = document.getElementById('pitch3d-match-select');
        var speedControl = document.getElementById('pitch3d-speed');
        var modeToggle = document.getElementById('pitch3d-mode');
        var homeToggle = document.getElementById('pitch3d-home-toggle');
        var awayToggle = document.getElementById('pitch3d-away-toggle');
        var playBtn = document.getElementById('pitch3d-play-btn');
        var seekSlider = document.getElementById('pitch3d-seek');

        if (matchSelect) {
            matchSelect.addEventListener('change', function () {
                var v = parseInt(this.value, 10);
                if (v) {
                    pitch.setMatchId(v);
                    // Load player name map
                    loadPlayerNameMap(v, pitch);
                }
            });
        }

        if (speedControl) {
            speedControl.addEventListener('change', function () {
                pitch.setSpeed(parseFloat(this.value));
            });
        }

        if (modeToggle) {
            modeToggle.addEventListener('click', function () {
                var newMode = pitch.players && Object.keys(pitch.players).length > 0 && pitch.players[Object.keys(pitch.players)[0]].userData.isCard ? 'dots' : 'cards';
                pitch.setMode(newMode);
                this.textContent = newMode === 'cards' ? '🎴 Cards' : '⚫ Dots';
            });
        }

        if (homeToggle) {
            homeToggle.addEventListener('click', function () {
                _showHome = !_showHome;
                pitch.showTeam('home', _showHome);
                this.classList.toggle('active');
            });
        }

        if (awayToggle) {
            awayToggle.addEventListener('click', function () {
                _showAway = !_showAway;
                pitch.showTeam('away', _showAway);
                this.classList.toggle('active');
            });
        }

        if (playBtn) {
            playBtn.addEventListener('click', function () {
                if (pitch.animating) {
                    pitch.stop();
                    this.textContent = '▶ Play';
                } else {
                    pitch.start();
                    this.textContent = '⏸ Pause';
                }
            });
        }

        if (seekSlider) {
            seekSlider.addEventListener('input', function () {
                pitch.updateScene(parseFloat(this.value));
            });
        }

        // Export for external seek calls
        window.__pitch3d = pitch;
    };

    function loadPlayerNameMap(matchId, pitch) {
        var bridge = (window.KawkabUtils && KawkabUtils.getBridge) ? KawkabUtils.getBridge() : (window.bridge || window.kawkabBridge);
        if (!bridge) return;
        bridge.get_match_players(matchId, function (result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (data && data.players) {
                    var map = {};
                    (data.players || data).forEach(function (p) {
                        var tid = p.track_id || p.trackId;
                        if (tid == null) return;
                        map[tid] = {
                            name: p.name || 'Player ' + tid,
                            rating: p.rating != null ? p.rating : computeRating(p),
                            stats: {
                                pac: p.sprint_speed || p.pac || 50,
                                sho: p.shooting || p.sho || 50,
                                pas: p.pass_accuracy || p.pas || 50,
                                dri: p.dribbling || p.dri || 50,
                                def: p.defending || p.def || 50,
                                phy: p.physical || p.phy || 50,
                            },
                        };
                    });
                    pitch.setPlayerNameMap(map);
                }
            } catch (e) {
                console.warn('Failed to load player map:', e);
            }
        });
    }

    function computeRating(p) {
        var stats = [
            p.pass_accuracy || 0, p.shooting || 0, p.dribbling || 0,
            p.defending || 0, p.physical || 0, p.sprint_speed || 0
        ];
        var sum = 0;
        for (var i = 0; i < stats.length; i++) sum += stats[i];
        return stats.length > 0 ? Math.round(sum / stats.length) : 75;
    }

    // roundRect polyfill for canvas
    if (!CanvasRenderingContext2D.prototype.roundRect) {
        CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
            if (typeof r === 'number') r = [r];
            var tl = r[0] || 0;
            var tr = (r[1] || tl);
            var br = (r[2] || tl);
            var bl = (r[3] || tl);
            this.beginPath();
            this.moveTo(x + tl, y);
            this.lineTo(x + w - tr, y);
            this.quadraticCurveTo(x + w, y, x + w, y + tr);
            this.lineTo(x + w, y + h - br);
            this.quadraticCurveTo(x + w, y + h, x + w - br, y + h);
            this.lineTo(x + bl, y + h);
            this.quadraticCurveTo(x, y + h, x, y + h - bl);
            this.lineTo(x, y + tl);
            this.quadraticCurveTo(x, y, x + tl, y);
            this.closePath();
            return this;
        };
    }

    console.log('Kawkab3DPitch loaded. Use init3DPitch() to start.');
})();

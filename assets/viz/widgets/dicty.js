/* Dictyostelium aggregation: amoebae relay a cAMP signal and walk toward it.
   Port of amlab/collective_motion/dicty.py (same equations, same parameters).
   Field (n x n, no-flux walls, dx = 1):
       du/dt = D lap(u) + k g(x) u (1 - u) (u - theta - gamma v) - mu u,     dv/dt = eps (u - v)
   g(x) = smoothed local cell density / rho_c, clipped to [0, 1]  (relay capacity of the cells).
   Cells: persistent random walk (speed v0); while cAMP at the cell is RISING (a wave front) the cell turns
   toward grad u and walks with speed vChem for tChem; every cell fires spontaneously at a small rate.
   Teaching model: model units, not fitted to data. Drawn on IE Tech black. */
(function () {
  "use strict";
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  // IE colours: Tech black stage, electric/sea blue field, VIG red (lightened) resting cells, Tech green walkers
  const C = { bg: "#000024", floor: "#08083F", rest: "#FF8A91", move: "#6DC201", white: "#FFFFFF", muted: "#A9B4D6", grid: "#1B1B5C" };

  const K = {
    n: 100, ny: 0, dt: 0.2,
    D: 1.0, k: 1.0, theta: 0.15, mu: 0.005, gamma: 1.2, eps: 0.025,
    density: 0.30, rhoC: 0.06, v0: 0.02, dRot: 0.05, vChem: 0.30, tChem: 6.0, rise: 0.04,
    fireRate: 4e-5, fireRadius: 3.5, chemotaxis: true, cells: true, meanEvery: 5,
  };
  const WINDOW = 600;

  /* ---------------------------------------------------------------- model (testable without a DOM) */
  function makeModel(p, rng) {
    const nx = p.n, ny = p.ny || p.n, N = nx * ny;
    const nc = p.cells ? Math.round(p.density * N) : 0;
    const m = {
      p, nx, ny, nc, time: 0, steps: 0, nFired: 0,
      u: new Float64Array(N), v: new Float64Array(N), nu: new Float64Array(N),
      x: new Float64Array(nc), y: new Float64Array(nc), h: new Float64Array(nc),
      timer: new Float64Array(nc), uPrev: new Float64Array(nc), active: new Uint8Array(nc),
      rho: new Float64Array(N), gain: new Float64Array(N).fill(1), tmp: new Float64Array(N),
    };
    for (let c = 0; c < nc; c++) { m.x[c] = rng() * nx; m.y[c] = rng() * ny; m.h[c] = (rng() * 2 - 1) * Math.PI; }
    const ix = new Int32Array(nc), iy = new Int32Array(nc);
    const sites = () => {
      for (let c = 0; c < nc; c++) { ix[c] = Math.min(Math.floor(m.x[c]), nx - 1); iy[c] = Math.min(Math.floor(m.y[c]), ny - 1); }
    };

    // [1, 2, 1] / 4 smoothing along both axes, edge values mirrored (as in the Python version)
    function blur(a, passes) {
      const t = m.tmp;
      for (let ps = 0; ps < passes; ps++) {
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
          const im = i > 0 ? i - 1 : 0, ip = i < nx - 1 ? i + 1 : nx - 1;
          t[i * ny + j] = 0.25 * (2 * a[i * ny + j] + a[im * ny + j] + a[ip * ny + j]);
        }
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
          const jm = j > 0 ? j - 1 : 0, jp = j < ny - 1 ? j + 1 : ny - 1;
          a[i * ny + j] = 0.25 * (2 * t[i * ny + j] + t[i * ny + jm] + t[i * ny + jp]);
        }
      }
    }
    m.refreshDensity = function () {
      if (!p.cells) { m.gain.fill(1); return; }
      sites();
      m.rho.fill(0);
      for (let c = 0; c < nc; c++) m.rho[ix[c] * ny + iy[c]] += 1;
      blur(m.rho, 2);
      for (let q = 0; q < N; q++) m.gain[q] = clamp(m.rho[q] / p.rhoC, 0, 1);
    };
    m.refreshDensity();

    const poisson = (mean) => {
      if (mean <= 0) return 0;
      if (mean > 30) return Math.max(0, Math.round(mean + Math.sqrt(mean) * rng.normal()));
      const L = Math.exp(-mean); let k = 0, q = 1;
      do { k++; q *= rng(); } while (q > L);
      return k - 1;
    };

    m.fire = function () {
      if (!p.cells || nc === 0 || p.fireRate <= 0) return;
      const cnt = poisson(p.fireRate * nc * p.dt);
      const r = p.fireRadius, R = Math.ceil(4 * r), s2 = 2 * r * r;
      for (let e = 0; e < cnt; e++) {
        const c = Math.floor(rng() * nc), cx = m.x[c], cy = m.y[c];
        const i0 = Math.max(0, Math.floor(cx) - R), i1 = Math.min(nx - 1, Math.floor(cx) + R);
        const j0 = Math.max(0, Math.floor(cy) - R), j1 = Math.min(ny - 1, Math.floor(cy) + R);
        for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
          const dx = i + 0.5 - cx, dy = j + 0.5 - cy;
          const q = i * ny + j; m.u[q] = Math.min(1, m.u[q] + Math.exp(-(dx * dx + dy * dy) / s2));
        }
        m.nFired++;
      }
    };

    m.stepField = function () {
      const u = m.u, v = m.v, nu = m.nu, dt = p.dt;
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < ny; j++) {
          const q = i * ny + j, a = u[q];
          let lap = 0;
          if (i > 0) lap += u[q - ny] - a;
          if (i < nx - 1) lap += u[q + ny] - a;
          if (j > 0) lap += u[q - 1] - a;
          if (j < ny - 1) lap += u[q + 1] - a;
          const react = p.k * m.gain[q] * a * (1 - a) * (a - p.theta - p.gamma * v[q]) - p.mu * a;
          nu[q] = a + dt * (p.D * lap + react);
          v[q] += dt * p.eps * (a - v[q]);
        }
      }
      m.u = nu; m.nu = u;
    };

    m.moveCells = function () {
      if (nc === 0) return;
      sites();
      const u = m.u, dt = p.dt, sq = Math.sqrt(2 * p.dRot * dt);
      for (let c = 0; c < nc; c++) {
        const i = ix[c], j = iy[c], uh = u[i * ny + j];
        const du = (uh - m.uPrev[c]) / dt;
        m.uPrev[c] = uh;
        m.h[c] += sq * rng.normal();
        let speed = p.v0;
        if (p.chemotaxis) {
          if (du > p.rise) {
            const gx = u[Math.min(i + 1, nx - 1) * ny + j] - u[Math.max(i - 1, 0) * ny + j];
            const gy = u[i * ny + Math.min(j + 1, ny - 1)] - u[i * ny + Math.max(j - 1, 0)];
            if (Math.hypot(gx, gy) > 1e-6) m.h[c] = Math.atan2(gy, gx);
            m.timer[c] = p.tChem;
          }
          m.active[c] = m.timer[c] > 0 ? 1 : 0;
          if (m.active[c]) speed = p.vChem;
          m.timer[c] = Math.max(m.timer[c] - dt, 0);
        } else m.active[c] = 0;
        let X = m.x[c] + speed * Math.cos(m.h[c]) * dt, Y = m.y[c] + speed * Math.sin(m.h[c]) * dt;
        if (X < 0) { X = -X; m.h[c] = Math.PI - m.h[c]; } else if (X >= nx) { X = 2 * nx - X - 1e-9; m.h[c] = Math.PI - m.h[c]; }
        if (Y < 0) { Y = -Y; m.h[c] = -m.h[c]; } else if (Y >= ny) { Y = 2 * ny - Y - 1e-9; m.h[c] = -m.h[c]; }
        m.x[c] = X; m.y[c] = Y;
      }
    };

    m.step = function () {
      if (m.steps % p.meanEvery === 0) m.refreshDensity();
      m.fire();
      m.stepField();
      if (p.cells) m.moveCells();
      m.steps++; m.time += p.dt;
    };
    m.run = function (t) { const k = Math.round(t / p.dt); for (let i = 0; i < k; i++) m.step(); };

    /* share of cells in crowded blocks, number of crowded clusters (4-connected groups of blocks) */
    m.aggregation = function (block, factor) {
      block = block || 5; factor = factor || 4;
      const bx = Math.floor(nx / block), by = Math.floor(ny / block), cnt = new Float64Array(bx * by);
      sites();
      for (let c = 0; c < nc; c++) cnt[Math.min(Math.floor(ix[c] / block), bx - 1) * by + Math.min(Math.floor(iy[c] / block), by - 1)] += 1;
      const mean = nc / (bx * by), dense = new Uint8Array(bx * by);
      let inDense = 0;
      for (let q = 0; q < dense.length; q++) if (cnt[q] > factor * mean) { dense[q] = 1; inDense += cnt[q]; }
      const seen = new Uint8Array(bx * by); let clusters = 0;
      for (let q0 = 0; q0 < dense.length; q0++) {
        if (!dense[q0] || seen[q0]) continue;
        clusters++; const st = [q0]; seen[q0] = 1;
        while (st.length) {
          const q = st.pop(), a = Math.floor(q / by), b = q % by;
          const nb = [[a + 1, b], [a - 1, b], [a, b + 1], [a, b - 1]];
          for (const [c1, d1] of nb) {
            if (c1 < 0 || c1 >= bx || d1 < 0 || d1 >= by) continue;
            const r = c1 * by + d1;
            if (dense[r] && !seen[r]) { seen[r] = 1; st.push(r); }
          }
        }
      }
      return { share: nc ? inDense / nc : 0, clusters };
    };
    m.waveFraction = function () { let s = 0; for (let q = 0; q < N; q++) if (m.u[q] > 0.5) s++; return s / N; };
    m.finite = function () { for (let q = 0; q < N; q++) if (!isFinite(m.u[q]) || !isFinite(m.v[q])) return false; return true; };
    return m;
  }

  /* front position along the middle row (right-most crossing of 0.5) and the front speed of the pure Nagumo equation */
  function frontPosition(m) {
    const j = Math.floor(m.ny / 2); let i = -1;
    for (let a = 0; a < m.nx; a++) if (m.u[a * m.ny + j] >= 0.5) i = a;
    if (i < 0 || i + 1 >= m.nx) return i;
    const r0 = m.u[i * m.ny + j], r1 = m.u[(i + 1) * m.ny + j];
    return i + (r0 - 0.5) / (r0 - r1 + 1e-12);
  }
  function frontSpeed(D, theta) {
    const p = Object.assign({}, K, { n: 400, ny: 5, D, theta, mu: 0, eps: 0, cells: false, density: 0, dt: Math.min(0.2, 0.125 / D) });
    const m = makeModel(p, VIZ_RNG(1));
    for (let i = 0; i < 6; i++) for (let j = 0; j < 5; j++) m.u[i * 5 + j] = 1;
    m.run(40); const a = frontPosition(m); m.run(80); const b = frontPosition(m);
    return (b - a) / 80;
  }
  function VIZ_RNG(seed) { return (typeof window !== "undefined" && window.VIZ && window.VIZ.rng) ? window.VIZ.rng(seed) : xorRng(seed); }
  function xorRng(seed) {   // same generator as VIZ.rng, for use in node
    let s = seed >>> 0 || 1;
    const u = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) + 0.5) / 4294967296; };
    u.normal = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
    return u;
  }

  const api = { makeModel, defaults: K, frontSpeed, nagumoSpeed: (D, k, th) => Math.sqrt(D * k / 2) * (1 - 2 * th), rng: xorRng };
  if (typeof module !== "undefined" && module.exports) module.exports = api;   // node tests
  if (typeof window !== "undefined" && window.VIZ) window.VIZ.dicty = api;
  if (typeof window === "undefined" || !window.VIZ || !window.VIZ.register) return;

  /* ---------------------------------------------------------------- widget */
  VIZ.register("dicty", {
    title: "Dictyostelium: amoebae relay a signal and gather",
    init(ui) {
      const S = { seed: 3, speed: 3 };
      const P = Object.assign({}, K);
      let m, hist, acc = 0, paused = false, lastReadout = 0;

      const sDens = VIZ.slider(ui.controls, { label: "Cell density", min: 0.05, max: 0.6, step: 0.05, value: P.density,
        fmt: (v) => v.toFixed(2) });
      sDens.input.addEventListener("change", () => { P.density = sDens.get(); reset(); });   // restarts the run when released
      VIZ.slider(ui.controls, { label: "Threshold θ (higher: harder to excite)", min: 0.05, max: 0.4, step: 0.01, value: P.theta,
        fmt: (v) => v.toFixed(2), onInput: (v) => { P.theta = v; } });
      VIZ.slider(ui.controls, { label: "Chemotactic speed", min: 0, max: 0.6, step: 0.05, value: P.vChem,
        fmt: (v) => v.toFixed(2), onInput: (v) => { P.vChem = Math.max(v, 1e-9); P.chemotaxis = v > 0; } });
      VIZ.slider(ui.controls, { label: "Speed", min: 1, max: 8, step: 1, value: S.speed, fmt: (v) => v + "×", onInput: (v) => { S.speed = v; } });
      const g = VIZ.group(ui.controls);
      const pauseBtn = VIZ.button(g, "Pause", () => { paused = !paused; pauseBtn.textContent = paused ? "Resume" : "Pause"; pauseBtn.setAttribute("aria-pressed", String(paused)); redraw(); });
      pauseBtn.setAttribute("aria-pressed", "false");
      VIZ.button(g, "New run", () => { S.seed++; reset(); }, { primary: true });

      const stage = ui.panel({ title: "The dish: cAMP wave (blue to white); cell dot = at rest, triangle = walking toward a wave", aspect: 1, minHeight: 300,
        label: "Square dish with expanding waves of cAMP in blue and white, and small cells that gather into clusters", onDraw: drawStage });
      const plot = ui.panel({ title: "Share of cells in crowded blocks and area excited", aspect: 1, minHeight: 300,
        label: "Share of cells in crowded blocks and excited area over the last 600 time units", onDraw: drawPlot });
      stage.canvas.tabIndex = 0;
      stage.canvas.addEventListener("keydown", (e) => {
        if (e.key === " ") { e.preventDefault(); pauseBtn.click(); }
        else if (e.key === "n" || e.key === "N") { S.seed++; reset(); }
      });

      function reset() {
        m = makeModel(P, VIZ.rng(S.seed * 7919 + Math.round(P.density * 100)));
        hist = { t: [], share: [], wave: [], clusters: 0, share0: 0 };
        sample(); redraw();
      }
      function sample() {
        const a = m.aggregation();
        hist.t.push(m.time); hist.share.push(a.share); hist.wave.push(m.waveFraction()); hist.clusters = a.clusters;
        if (hist.t.length > 400) { hist.t.shift(); hist.share.shift(); hist.wave.shift(); }
      }

      // ---------------- drawing ----------------
      const fieldCanvas = document.createElement("canvas");
      const fctx = fieldCanvas.getContext("2d");
      let img = null;
      const stops = [[8, 8, 63], [0, 0, 219], [71, 191, 255], [255, 255, 255]];
      const lut = new Uint8ClampedArray(256 * 3);
      for (let i = 0; i < 256; i++) {
        const v = (i / 255) * 3, k = Math.min(2, Math.floor(v)), f = v - k;
        for (let t = 0; t < 3; t++) lut[i * 3 + t] = stops[k][t] + (stops[k + 1][t] - stops[k][t]) * f;
      }
      function drawStage(c) {
        if (!m) return;
        const ctx = c.ctx, nx = m.nx, ny = m.ny, sc = Math.min(c.w, c.h) / nx, ox = (c.w - sc * nx) / 2, oy = (c.h - sc * ny) / 2;
        if (!img || img.width !== nx) { fieldCanvas.width = nx; fieldCanvas.height = ny; img = fctx.createImageData(nx, ny); }
        const d = img.data;
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const val = clamp(m.u[i * ny + (ny - 1 - j)], 0, 1), li = Math.round(val * 255) * 3, o = (j * nx + i) * 4;
          d[o] = lut[li]; d[o + 1] = lut[li + 1]; d[o + 2] = lut[li + 2]; d[o + 3] = 255;
        }
        fctx.putImageData(img, 0, 0);
        ctx.fillStyle = C.bg; ctx.fillRect(0, 0, c.w, c.h);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(fieldCanvas, ox, oy, sc * nx, sc * ny);
        const X = (x) => ox + sc * x, Y = (y) => oy + sc * (ny - y);
        ctx.fillStyle = C.rest;
        const r = Math.max(1.2, sc * 0.45);
        for (let q = 0; q < m.nc; q++) if (!m.active[q]) ctx.fillRect(X(m.x[q]) - r / 2, Y(m.y[q]) - r / 2, r, r);
        const tr = Math.max(3.2, sc * 1.3);
        ctx.fillStyle = C.move; ctx.strokeStyle = C.white; ctx.lineWidth = 0.8;
        ctx.beginPath();
        for (let q = 0; q < m.nc; q++) if (m.active[q]) {
          const px = X(m.x[q]), py = Y(m.y[q]);
          ctx.moveTo(px, py - tr); ctx.lineTo(px + 0.87 * tr, py + 0.5 * tr); ctx.lineTo(px - 0.87 * tr, py + 0.5 * tr); ctx.closePath();
        }
        ctx.fill(); ctx.stroke();
        const a = hist.t.length ? hist.share[hist.share.length - 1] : 0;
        const label = `${paused ? "PAUSED · " : ""}t = ${m.time.toFixed(0)} · ${Math.round(100 * a)} % crowded · ${hist.clusters} clusters${P.chemotaxis ? "" : " · control"}`;
        ctx.font = "700 12px Montserrat, Arial, sans-serif"; ctx.textAlign = "left";
        ctx.fillStyle = "rgba(0,0,36,0.78)"; ctx.fillRect(6, 6, ctx.measureText(label).width + 12, 22);
        ctx.fillStyle = C.white; ctx.fillText(label, 12, 21);
      }

      function drawPlot(c) {
        if (!m) return;
        const ax = VIZ.axes(c, { xlim: [-WINDOW, 0], ylim: [0, 1.02], xlabel: "time before now (t.u.)", ylabel: "share of cells / area excited × 4" });
        const T = ax.T;
        const now = m.time, xs = [], s = [], w = [];
        for (let i = 0; i < hist.t.length; i++) if (hist.t[i] > now - WINDOW) { xs.push(hist.t[i] - now); s.push(hist.share[i]); w.push(Math.min(1, 4 * hist.wave[i])); }
        ax.clip();
        ax.line(xs, w, { color: "#E2000F", width: 2, dash: [5, 4] });
        ax.line(xs, s, { color: T.c[0], width: 2.5 });
        c.ctx.restore();
        VIZ.legend(c, [{ label: "share of cells in crowded blocks", color: T.c[0], width: 2.5 }, { label: "area excited (u > 0.5), times 4", color: "#E2000F", dash: [5, 4], width: 2 }], ax.pad.l + 10, ax.pad.t + 16);
      }

      function readout(force) {
        const now = performance.now();
        if (!force && now - lastReadout < 700) return;
        lastReadout = now;
        const a = hist.share[hist.share.length - 1] || 0;
        ui.readout.textContent = `time ${m.time.toFixed(0)} · cells ${m.nc} · share of cells in crowded blocks ${(100 * a).toFixed(0)} % · crowded clusters ${hist.clusters} · ` +
          `area excited ${(100 * m.waveFraction()).toFixed(0)} % · density ${P.density.toFixed(2)}, θ = ${P.theta.toFixed(2)}, chemotaxis ${P.chemotaxis ? P.vChem.toFixed(2) : "off"}${paused ? " · paused" : ""}`;
      }
      function redraw() { stage.draw(); plot.draw(); readout(true); }

      let frame = 0;
      const loop = VIZ.loop(ui.el, (dt) => {
        if (paused) return;
        m.p = P;                                        // sliders change the live parameters
        for (let i = 0; i < S.speed; i++) m.step();
        if ((frame++ % 3) === 0) sample();
        stage.draw(); plot.draw(); readout(false);
      });

      reset();
      if (VIZ.reduceMotion) { m.run(300); sample(); redraw(); }
      else loop.start();
    },
  });
})();

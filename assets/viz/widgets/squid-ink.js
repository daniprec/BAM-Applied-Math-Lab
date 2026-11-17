/* Squid ink: a school of fish hunts one squid that releases ink.
   Port of amlab/collective_motion/squid_ink.py (same rules, same parameters).
   Fish (predators): alignment, cohesion, separation and pursuit, limited turning, noise.
   Squid: cruises, flees from the nearest fish within the flight distance and releases a bolus of ink
   when the gland has reloaded. Ink: in every step each cell passes `spread` of its ink to its four
   neighbours (half stays, half goes out), and a cell below `threshold` is reset to 0. Vision: the
   pursuit weight of a fish is multiplied by exp(-kappa * mean ink on the sightline * distance).
   A fish within the capture radius catches the squid. The stage is drawn on IE Tech black. */
(function () {
  "use strict";
  const TWO_PI = 2 * Math.PI;
  const wrapPi = (a) => ((((a + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI) - Math.PI;
  const mod2pi = (a) => ((a % TWO_PI) + TWO_PI) % TWO_PI;
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  // IE colours (brand sheet): Tech black stage, VIG red (lightened) fish, Tech green squid, electric-blue ink
  const C = { bg: "#000024", floor: "#08083F", dot: "#1B1B5C", fish: "#FF5A63", squid: "#6DC201", spent: "#8A8FA8", white: "#FFFFFF", muted: "#A9B4D6" };

  const K = {
    n: 24, L: 40, G: 100, dt: 0.12,
    spread: 0.5, threshold: 0.002, inkSteps: 3, thickness: 3.0, boWidth: 0.9, kappa: 2.8, water: 0.05, reload: 5.0, useFraction: 0.6, emptyBelow: 0.05, ink: true,
    vFish: 0.70, fishTurn: 0.20, fishNoise: 0.035, rSep: 1.15, rSchool: 5.0, wAlign: 1.1, wCohere: 0.35, wSep: 1.4, wPursue: 1.9,
    vCruise: 0.30, vFlee: 0.45, flight: 8.0, squidTurn: 0.33, squidNoise: 0.045, capture: 0.8,
  };
  const VIEW = 15, WINDOW = 60, T_MAX = 250, SQUID_SIZE = 3.0, SEE = 0.5;

  /* ---------------------------------------------------------------- model (testable without a DOM) */
  function makeModel(p, rng) {
    const L = p.L, G = p.G, dx = L / G, n = p.n;
    const wrap = (d) => ((((d + L / 2) % L) + L) % L) - L / 2;
    const m = {
      p, dx, time: 0, releases: 0, captured: false, captureTime: NaN, cooldown: 0, closest: 0, stock: 1,
      fx: new Float64Array(n), fy: new Float64Array(n), fh: new Float64Array(n), T: new Float64Array(n).fill(1),
      sx: 0.28 * L, sy: 0.42 * L, sh: -0.15,
      ink: new Float32Array(G * G), tmp: new Float32Array(G * G),
    };
    for (let i = 0; i < n; i++) {
      m.fx[i] = (((0.68 * L + 2.6 * rng.normal()) % L) + L) % L;
      m.fy[i] = (((0.52 * L + 2.6 * rng.normal()) % L) + L) % L;
      m.fh[i] = (rng() * 2 - 1) * Math.PI;
    }
    const nearest = () => {
      let best = Infinity;
      for (let i = 0; i < n; i++) best = Math.min(best, Math.hypot(wrap(m.fx[i] - m.sx), wrap(m.fy[i] - m.sy)));
      return best;
    };
    m.closest = nearest();
    m.wrap = wrap;

    m.release = function () {
      const amount = p.thickness * m.stock;
      const s2 = 2 * p.boWidth * p.boWidth, R = Math.ceil((5 * p.boWidth) / dx);
      const cx = Math.round(m.sx / dx), cy = Math.round(m.sy / dx);
      for (let j = -R; j <= R; j++) {
        for (let i = -R; i <= R; i++) {
          const gi = (((cx + i) % G) + G) % G, gj = (((cy + j) % G) + G) % G;
          const rx = wrap(gi * dx - m.sx), ry = wrap(gj * dx - m.sy);
          m.ink[gj * G + gi] += amount * Math.exp(-(rx * rx + ry * ry) / s2);
        }
      }
      m.releases++;
      m.stock *= 1 - p.useFraction;
    };

    m.spreadInk = function () {
      const s = p.spread, keep = 1 - s, q = s / 4, thr = p.threshold;
      for (let k = 0; k < p.inkSteps; k++) {
        const a = m.ink, b = m.tmp;
        for (let j = 0; j < G; j++) {
          const jm = ((j + G - 1) % G) * G, jp = ((j + 1) % G) * G, jj = j * G;
          for (let i = 0; i < G; i++) {
            const im = (i + G - 1) % G, ip = (i + 1) % G;
            const v = keep * a[jj + i] + q * (a[jm + i] + a[jp + i] + a[jj + im] + a[jj + ip]);
            b[jj + i] = v < thr ? 0 : v;
          }
        }
        m.ink = b; m.tmp = a;
      }
    };

    const sample = (x, y) => {
      const gx = (((x % L) + L) % L) / dx, gy = (((y % L) + L) % L) / dx;
      const x0 = Math.floor(gx), y0 = Math.floor(gy), fx = gx - x0, fy = gy - y0;
      const x1 = (x0 + 1) % G, y1 = (y0 + 1) % G, X0 = x0 % G, Y0 = y0 % G;
      return m.ink[Y0 * G + X0] * (1 - fx) * (1 - fy) + m.ink[Y0 * G + x1] * fx * (1 - fy)
        + m.ink[y1 * G + X0] * (1 - fx) * fy + m.ink[y1 * G + x1] * fx * fy;
    };

    m.hasInk = () => { for (let i = 0; i < m.ink.length; i++) if (m.ink[i] > 0) return true; return false; };

    m.transmission = function () {
      if (!m.hasInk()) { for (let i = 0; i < n; i++) m.T[i] = Math.exp(-p.water * Math.hypot(wrap(m.sx - m.fx[i]), wrap(m.sy - m.fy[i]))); return; }
      const rx = new Float64Array(n), ry = new Float64Array(n), dist = new Float64Array(n);
      let dmax = 0;
      for (let i = 0; i < n; i++) {
        rx[i] = wrap(m.sx - m.fx[i]); ry[i] = wrap(m.sy - m.fy[i]); dist[i] = Math.hypot(rx[i], ry[i]); dmax = Math.max(dmax, dist[i]);
      }
      const ns = Math.max(2, Math.ceil(dmax / (dx * 1.5)));
      for (let i = 0; i < n; i++) {
        let acc = 0;
        for (let k = 0; k < ns; k++) {
          const f = 0.08 + (0.84 * k) / (ns - 1);
          acc += sample(m.fx[i] + rx[i] * f, m.fy[i] + ry[i] * f);
        }
        m.T[i] = Math.exp(-(p.water + p.kappa * (acc / ns)) * dist[i]);
      }
    };

    m.fishUpdate = function () {
      const nh = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let ax = 0, ay = 0, nS = 0, awx = 0, awy = 0, cx = 0, cy = 0;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const dx_ = wrap(m.fx[j] - m.fx[i]), dy_ = wrap(m.fy[j] - m.fy[i]), d = Math.hypot(dx_, dy_);
          if (d < p.rSep) { awx -= dx_ / Math.max(d, 1e-8); awy -= dy_ / Math.max(d, 1e-8); }
          if (d < p.rSchool) { nS++; cx += dx_; cy += dy_; ax += Math.cos(m.fh[j]); ay += Math.sin(m.fh[j]); }
        }
        const nn = Math.max(nS, 1);
        let px = wrap(m.sx - m.fx[i]), py = wrap(m.sy - m.fy[i]);
        const pn = Math.max(Math.hypot(px, py), 1e-8); px /= pn; py /= pn;
        const vx = p.wAlign * (ax / nn) + p.wCohere * (cx / nn) + p.wSep * awx + p.wPursue * m.T[i] * px;
        const vy = p.wAlign * (ay / nn) + p.wCohere * (cy / nn) + p.wSep * awy + p.wPursue * m.T[i] * py;
        const turn = clamp(wrapPi(Math.atan2(vy, vx) - m.fh[i]), -p.fishTurn, p.fishTurn) + p.fishNoise * rng.normal();
        nh[i] = mod2pi(m.fh[i] + turn);
      }
      m.fh = nh;
    };

    m.squidUpdate = function () {
      let k = 0, best = Infinity, rx = 0, ry = 0;
      for (let i = 0; i < n; i++) {
        const dx_ = wrap(m.fx[i] - m.sx), dy_ = wrap(m.fy[i] - m.sy), d = Math.hypot(dx_, dy_);
        if (d < best) { best = d; k = i; rx = dx_; ry = dy_; }
      }
      const threatened = best < p.flight;
      if (threatened) {
        const turn = clamp(wrapPi(Math.atan2(-ry, -rx) - m.sh), -p.squidTurn, p.squidTurn);
        m.sh = mod2pi(m.sh + turn);
        if (p.ink && m.cooldown <= 0 && (p.useFraction === 0 || m.stock >= p.emptyBelow)) { m.release(); m.cooldown = p.reload; }
      } else {
        m.sh = mod2pi(m.sh + p.squidNoise * rng.normal());
      }
      const v = threatened ? p.vFlee : p.vCruise;
      m.sx = (((m.sx + v * p.dt * Math.cos(m.sh)) % L) + L) % L;
      m.sy = (((m.sy + v * p.dt * Math.sin(m.sh)) % L) + L) % L;
    };

    m.step = function () {
      m.spreadInk();
      m.cooldown = Math.max(0, m.cooldown - p.dt);
      m.transmission();
      m.fishUpdate();
      for (let i = 0; i < n; i++) {
        m.fx[i] = (((m.fx[i] + p.vFish * p.dt * Math.cos(m.fh[i])) % L) + L) % L;
        m.fy[i] = (((m.fy[i] + p.vFish * p.dt * Math.sin(m.fh[i])) % L) + L) % L;
      }
      m.squidUpdate();
      m.time += p.dt;
      m.closest = nearest();
      if (m.closest < p.capture && !m.captured) { m.captured = true; m.captureTime = m.time; }
    };
    m.meanT = () => { let s = 0; for (let i = 0; i < n; i++) s += m.T[i]; return s / n; };
    return m;
  }

  // For tests and for the lesson page's own experiments: VIZ.squidInk.makeModel(params, VIZ.rng(seed)).
  if (window.VIZ) window.VIZ.squidInk = { makeModel, defaults: K };

  /* ---------------------------------------------------------------- widget */
  if (!window.VIZ || !window.VIZ.register) return;
  VIZ.register("squid-ink", {
    title: "Squid ink: a school hunts a squid",
    init(ui) {
      const S = { seed: 4, n: 24 };
      const P = Object.assign({}, K);
      let m, hist, ended, endFrames, acc = 0;
      const tally = { captured: 0, escaped: 0 };

      const sKappa = VIZ.slider(ui.controls, { label: "Ink opacity κ", min: 0, max: 8, step: 0.2, value: P.kappa,
        fmt: (v) => v.toFixed(1), onInput: (v) => { P.kappa = v; } });
      const sThick = VIZ.slider(ui.controls, { label: "Ink thickness", min: 0.5, max: 8, step: 0.5, value: P.thickness,
        fmt: (v) => v.toFixed(1), onInput: (v) => { P.thickness = v; } });
      const sUse = VIZ.slider(ui.controls, { label: "Ink used per release", min: 0, max: 0.9, step: 0.1, value: P.useFraction,
        fmt: (v) => v.toFixed(1), onInput: (v) => { P.useFraction = v; } });
      VIZ.select(ui.controls, { label: "School size", value: "24", options: [["12", "12"], ["24", "24"], ["40", "40"]],
        onChange: (v) => { S.n = +v; reset(); } });
      const g = VIZ.group(ui.controls);
      const inkBtn = VIZ.button(g, "Ink: on", () => { P.ink = !P.ink; inkBtn.textContent = P.ink ? "Ink: on" : "Ink: off"; resetTally(); });
      let follow = true;
      const camBtn = VIZ.button(g, "Camera: follows squid", () => {
        follow = !follow; camBtn.textContent = follow ? "Camera: follows squid" : "Camera: fixed"; redraw();
      });
      VIZ.button(g, "Recharge ink", () => { m.stock = 1; redraw(); });
      VIZ.button(g, "New hunt", () => { S.seed++; reset(); }, { primary: true });

      const stage = ui.panel({ title: "The hunt (use the camera button to follow or fix the view)", aspect: 1, minHeight: 280,
        label: "A green squid with arms and tentacles releases an ink cloud while red fish chase it", onDraw: drawStage });
      const plot = ui.panel({ title: "What the fish can see", aspect: 1, minHeight: 280,
        label: "Mean visibility of the squid and the distance of the closest fish over the last 60 time units", onDraw: drawPlot });

      function resetTally() { tally.captured = 0; tally.escaped = 0; }

      function reset() {
        P.n = S.n;
        m = makeModel(P, VIZ.rng(S.seed * 7919 + S.n));
        hist = { t: [], vis: [], dist: [] };
        ended = ""; endFrames = 0;
        redraw();
      }

      // ---------------- drawing ----------------
      const inkCanvas = document.createElement("canvas");
      const inkCtx = inkCanvas.getContext("2d");
      let inkImg = null;
      // cmap: transparent -> #3A3AFF -> #8E7BFF -> #F0EBFF, like the Matplotlib version
      const stops = [[41, 41, 255, 0], [58, 58, 255, 255], [142, 123, 255, 255], [240, 235, 255, 255]];
      function inkColour(c, out, o) {
        const v = clamp(c / 1.2, 0, 1) * 3, k = Math.min(2, Math.floor(v)), f = v - k, a = stops[k], b = stops[k + 1];
        for (let t = 0; t < 4; t++) out[o + t] = a[t] + (b[t] - a[t]) * f;
      }
      function drawInkFixed(ctx, X, Y, sc) {
        const G = P.G;
        if (!inkImg || inkImg.width !== G) { inkCanvas.width = G; inkCanvas.height = G; inkImg = inkCtx.createImageData(G, G); }
        const d = inkImg.data;
        for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) inkColour(m.ink[(G - 1 - j) * G + i], d, (j * G + i) * 4);
        inkCtx.putImageData(inkImg, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(inkCanvas, X(-0.5 * m.dx), Y(P.L - 0.5 * m.dx), P.L * sc, P.L * sc);
      }
      function drawInk(ctx, X, Y, sc) {
        const G = P.G, half = Math.ceil(VIEW / m.dx) + 1, w = 2 * half;
        if (!inkImg || inkImg.width !== w) { inkCanvas.width = w; inkCanvas.height = w; inkImg = inkCtx.createImageData(w, w); }
        const cx = Math.round(m.sx / m.dx), cy = Math.round(m.sy / m.dx), d = inkImg.data;
        for (let j = 0; j < w; j++) {
          const gj = (((cy + (w - 1 - j) - half) % G) + G) % G;     // canvas y grows downwards
          for (let i = 0; i < w; i++) {
            const gi = (((cx + i - half) % G) + G) % G;
            inkColour(m.ink[gj * G + gi], d, (j * w + i) * 4);
          }
        }
        inkCtx.putImageData(inkImg, 0, 0);
        ctx.imageSmoothingEnabled = true;
        const offx = cx * m.dx - m.sx, offy = cy * m.dx - m.sy;     // the squid is not exactly on a grid point
        ctx.drawImage(inkCanvas, X((-half - 0.5) * m.dx + offx), Y((half - 0.5) * m.dx + offy), w * m.dx * sc, w * m.dx * sc);
      }

      function drawSquid(ctx, px, py, theta, size, fill) {
        const mix = (a, b, t) => Math.round(a + (b - a) * t);
        const tint = `rgb(${mix(138, 109, fill)},${mix(143, 194, fill)},${mix(168, 1, fill)})`; // grey when empty, green when full
        ctx.save(); ctx.translate(px, py); ctx.rotate(-theta); ctx.scale(size, size);
        const poly = (pts) => { ctx.beginPath(); ctx.moveTo(pts[0][0], -pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], -pts[i][1]); ctx.closePath(); };
        const line = (pts) => { ctx.beginPath(); ctx.moveTo(pts[0][0], -pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], -pts[i][1]); ctx.stroke(); };
        ctx.lineJoin = "round"; ctx.lineCap = "round";
        ctx.strokeStyle = tint; ctx.lineWidth = 1.8 / size;
        for (let k = 0; k < 8; k++) { const y = -0.09 + (0.18 * k) / 7; line([[-0.14, y], [-0.45, 1.6 * y], [-0.78, 2.4 * y]]); }
        ctx.lineWidth = 2.4 / size;
        [1, -1].forEach((s) => line([[-0.14, s * 0.04], [-0.65, s * 0.10], [-1.2, s * 0.20]]));
        ctx.strokeStyle = C.white; ctx.lineWidth = 3.2 / size;
        [1, -1].forEach((s) => line([[-1.2, s * 0.20], [-1.32, s * 0.24]]));
        ctx.fillStyle = tint; ctx.strokeStyle = C.white; ctx.lineWidth = 1.2 / size;
        const mantle = [[1, 0], [0.82, 0.10], [0.38, 0.17], [0, 0.17], [-0.14, 0.10], [-0.14, -0.10], [0, -0.17], [0.38, -0.17], [0.82, -0.10]];
        const fin = [[0.98, 0], [0.74, 0.30], [0.50, 0.32], [0.38, 0.17]];
        [mantle, fin, fin.map((q) => [q[0], -q[1]])].forEach((pts) => { poly(pts); ctx.fill(); ctx.stroke(); });
        ctx.restore();
      }

      function drawStage(c) {
        if (!m) return;
        const ctx = c.ctx, L = P.L, sc = Math.min(c.w, c.h) / (2 * VIEW), scF = Math.min(c.w, c.h) / L;
        ctx.fillStyle = C.floor; ctx.fillRect(0, 0, c.w, c.h);
        // follow: coordinates relative to the squid; fixed: absolute coordinates of the whole sea
        const s = follow ? sc : scF;
        const X = follow ? (wx) => c.w / 2 + sc * wx : (wx) => (c.w - scF * L) / 2 + scF * wx;
        const Y = follow ? (wy) => c.h / 2 - sc * wy : (wy) => (c.h + scF * L) / 2 - scF * wy;
        const pos = (x, y) => (follow ? [m.wrap(x - m.sx), m.wrap(y - m.sy)] : [x, y]);
        const lim = follow ? VIEW + 2 : Infinity;
        ctx.fillStyle = C.dot;
        for (let gx = 0; gx < L; gx += 4) for (let gy = 0; gy < L; gy += 4) {
          const [rx, ry] = pos(gx, gy);
          if (Math.abs(rx) < (follow ? VIEW : Infinity) && Math.abs(ry) < (follow ? VIEW : Infinity)) ctx.fillRect(X(rx) - 1, Y(ry) - 1, 2, 2);
        }
        if (follow) drawInk(ctx, X, Y, sc); else drawInkFixed(ctx, X, Y, scF);
        if (!follow) { ctx.strokeStyle = C.dot; ctx.lineWidth = 1; ctx.strokeRect(X(0), Y(L), scF * L, scF * L); }
        const body = (follow ? 0.95 : 1.6) * s;
        for (let i = 0; i < P.n; i++) {
          const [rx, ry] = pos(m.fx[i], m.fy[i]);
          if (Math.abs(rx) > lim || Math.abs(ry) > lim) continue;
          const cs = Math.cos(m.fh[i]), sn = -Math.sin(m.fh[i]), px = X(rx), py = Y(ry);
          ctx.beginPath();
          ctx.moveTo(px + body * cs, py + body * sn);
          ctx.lineTo(px - 0.65 * body * cs - 0.45 * body * sn, py - 0.65 * body * sn + 0.45 * body * cs);
          ctx.lineTo(px - 0.65 * body * cs + 0.45 * body * sn, py - 0.65 * body * sn - 0.45 * body * cs);
          ctx.closePath();
          if (m.T[i] > SEE) { ctx.fillStyle = C.fish; ctx.fill(); }
          ctx.strokeStyle = C.fish; ctx.lineWidth = 1.3; ctx.stroke();
        }
        const [qx, qy] = pos(m.sx, m.sy), empty = P.useFraction > 0 && m.stock < P.emptyBelow;
        const size = follow ? SQUID_SIZE * sc : SQUID_SIZE * 0.8 * scF;
        if (P.ink && m.cooldown > 0 && !empty) { ctx.strokeStyle = C.white; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(X(qx), Y(qy), size * 0.55, 0, TWO_PI); ctx.stroke(); }
        drawSquid(ctx, X(qx), Y(qy), m.sh, size, P.useFraction === 0 ? 1 : Math.max(0, Math.min(1, m.stock)));
        ctx.font = "700 13px Montserrat, Arial, sans-serif"; ctx.textAlign = "left"; ctx.fillStyle = C.white;
        const supply = P.useFraction === 0 ? "unlimited ink" : `ink left ${Math.round(100 * m.stock)} %`;
        const label = ended === "captured" ? "✕ CAPTURED" : ended === "escaped" ? "✓ ESCAPED" : (P.ink ? (empty ? "● INK SAC EMPTY" : `${m.cooldown > 0 ? "●" : "○"} ${supply}`) : "○ NO INK");
        ctx.fillText(label, 10, 20);
        ctx.font = "500 11px Montserrat, Arial, sans-serif"; ctx.fillStyle = C.muted;
        ctx.fillText("filled arrow: sees the squid · hollow arrow: lost it in the ink", 10, c.h - 10);
      }

      function drawPlot(c) {
        if (!m) return;
        const ax = VIZ.axes(c, { xlim: [-WINDOW, 0], ylim: [0, 1.02], xlabel: "time before now (t.u.)", ylabel: "mean visibility",
          pad: { r: 38 } });
        const T = ax.T, ctx = c.ctx;
        if (hist.t.length) {
          const now = m.time, xs = [], v = [], dd = [];
          for (let i = 0; i < hist.t.length; i++) if (hist.t[i] > now - WINDOW) { xs.push(hist.t[i] - now); v.push(hist.vis[i]); dd.push(Math.min(hist.dist[i], 30) / 30 * 1.02); }
          ax.clip();
          ax.line(xs, dd, { color: C.fish, width: 2, dash: [5, 4] });
          ax.line(xs, v, { color: T.c[0], width: 2.5 });
          ctx.restore();
        }
        // right axis: distance of the closest fish, 0 to 30 body lengths
        ctx.save(); ctx.font = T.font(11); ctx.fillStyle = T.text; ctx.textAlign = "left";
        [0, 10, 20, 30].forEach((d) => ctx.fillText(String(d), ax.pad.l + ax.pw + 5, ax.sy(d / 30 * 1.02) + 4));
        ctx.restore();
        VIZ.legend(c, [{ label: "mean visibility", color: T.c[0], width: 2.5 }, { label: "closest fish, right axis (BL)", color: C.fish, dash: [5, 4], width: 2 }], ax.pad.l + 10, ax.pad.t + 16);
      }

      function readout() {
        const n = tally.captured + tally.escaped;
        const pc = n ? Math.round((100 * tally.captured) / n) + " %" : "n/a";
        ui.readout.textContent = `time ${m.time.toFixed(1)} · closest fish ${m.closest.toFixed(1)} BL · mean visibility ${m.meanT().toFixed(2)} · ink releases ${m.releases} · ` +
          `hunts: captured ${tally.captured}, escaped ${tally.escaped}, capture probability ${pc} · ink ${P.ink ? "on" : "off"}, κ = ${P.kappa.toFixed(1)}, thickness ${P.thickness.toFixed(1)}, ink used per release ${P.useFraction.toFixed(1)}`;
      }
      function redraw() { stage.draw(); plot.draw(); readout(); }

      function advance() {
        if (ended) { if (++endFrames > 90) { S.seed++; reset(); } return; }
        m.step();
        hist.t.push(m.time); hist.vis.push(m.meanT()); hist.dist.push(m.closest);
        if (hist.t.length > 600) { hist.t.shift(); hist.vis.shift(); hist.dist.shift(); }
        if (m.captured) { ended = "captured"; tally.captured++; }
        else if (m.time >= T_MAX) { ended = "escaped"; tally.escaped++; }
      }

      const loop = VIZ.loop(ui.el, (dt) => {
        if (ended) { advance(); redraw(); return; }
        acc += dt * 60;                           // 60 steps (7 time units) per second
        let k = 0;
        while (acc >= 1 && k < 4) { advance(); acc -= 1; k++; }
        acc = Math.min(acc, 2);
        redraw();
      });

      reset();
      if (VIZ.reduceMotion) { for (let i = 0; i < 400; i++) advance(); redraw(); }
      else loop.start();
    },
  });
})();

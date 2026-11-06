/* Vicsek model, as in amlab.collective_motion.vicsek.vicsek_equations:
   θ_i ← atan2(⟨sin θ⟩, ⟨cos θ⟩) over neighbours with |x_j − x_i| ≤ r (self included,
   distances NOT wrapped across the boundary, like the lab's distance matrix),
   plus noise η·(U − 1/2); x_i ← x_i + v0 Δt (cos θ_i, sin θ_i), then mod L.
   Defaults from the lab: N = 100, L = 10, η = 0.5, r = 1, v0 = 0.03, Δt = 1. */
(function () {
  "use strict";
  const R = 1, V0 = 0.03, TWO_PI = 2 * Math.PI, HIST = 3000;

  VIZ.register("vicsek", {
    title: "Vicsek model: alignment against noise",
    init(ui) {
      const S = { N: 100, L: 10, eta: 0.5, x: [], y: [], th: [], n: 0, hist: [], meas: [], since: 0, seed: 3, sweep: null, va: 0 };
      let u = VIZ.rng(S.seed);

      const sEta = VIZ.slider(ui.controls, { label: "Noise <i>η</i>", min: 0, max: 5, step: 0.05, value: S.eta,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.eta = v; S.since = 0; S.sweep = null; } });
      const sN = VIZ.slider(ui.controls, { label: "<i>N</i>", min: 20, max: 400, step: 10, value: S.N,
        fmt: (v) => String(v), onInput: (v) => { S.N = v; S.sweep = null; reset(); } });
      const sL = VIZ.slider(ui.controls, { label: "Box <i>L</i>", min: 3, max: 25, step: 0.5, value: S.L,
        fmt: (v) => v.toFixed(1), onInput: (v) => { S.L = v; S.sweep = null; reset(); } });
      const g = VIZ.group(ui.controls);
      VIZ.button(g, "Restart", () => { S.seed++; reset(); }, { primary: true });
      VIZ.button(g, "Measure va", measure);
      const sweepBtn = VIZ.button(g, "Sweep η from 0 to 5", startSweep);
      VIZ.button(g, "Clear points", () => { S.meas = []; draw(); });

      const box = ui.panel({ title: "Particles in the periodic box (colour = heading)", aspect: 1, minHeight: 240,
        label: "Self-propelled particles drawn as short arrows in a square box", onDraw: drawBox });
      const trace = ui.panel({ title: "Order parameter  va(t)", aspect: 1, minHeight: 240,
        label: "Recent history of the Vicsek order parameter", onDraw: drawTrace });
      const curve = ui.panel({ title: "va against η: your measurements (dots); dashed: 1/√N, the value for random headings",
        cls: "viz__panel--wide", aspect: 2.8, minHeight: 200, label: "Measured order parameter against noise", onDraw: drawCurve });

      function reset() {
        u = VIZ.rng(S.seed * 101 + S.N);
        S.x = []; S.y = []; S.th = [];
        for (let i = 0; i < S.N; i++) { S.th.push(TWO_PI * u()); }
        for (let i = 0; i < S.N; i++) { S.x.push(S.L * u()); S.y.push(S.L * u()); }
        S.n = 0; S.since = 0; S.hist = []; S.va = order(); draw();
      }
      function order() {
        let c = 0, s = 0;
        for (let i = 0; i < S.N; i++) { c += Math.cos(S.th[i]); s += Math.sin(S.th[i]); }
        return Math.hypot(c, s) / S.N;
      }
      function step() {
        const N = S.N, x = S.x, y = S.y, th = S.th, sn = new Float64Array(N), cs = new Float64Array(N);
        for (let i = 0; i < N; i++) { sn[i] = Math.sin(th[i]); cs[i] = Math.cos(th[i]); }
        const nt = new Array(N);
        for (let i = 0; i < N; i++) {
          let ss = 0, cc = 0;
          for (let j = 0; j < N; j++) {
            const dx = x[j] - x[i], dy = y[j] - y[i];
            if (dx * dx + dy * dy <= R * R) { ss += sn[j]; cc += cs[j]; }
          }
          const t = Math.atan2(ss, cc) + S.eta * (u() - 0.5);
          nt[i] = ((t % TWO_PI) + TWO_PI) % TWO_PI;
        }
        for (let i = 0; i < N; i++) {
          th[i] = nt[i];
          x[i] = (((x[i] + V0 * Math.cos(nt[i])) % S.L) + S.L) % S.L;
          y[i] = (((y[i] + V0 * Math.sin(nt[i])) % S.L) + S.L) % S.L;
        }
        S.n++; S.since++;
        S.va = order();
        S.hist.push([S.n, S.va]);
        if (S.hist.length > HIST) S.hist.shift();
      }
      function avgLast(k) {
        const pts = S.hist.slice(-Math.min(k, S.since));
        return pts.reduce((a, p) => a + p[1], 0) / Math.max(1, pts.length);
      }
      function measure() {
        // average over the last 500 steps, after at least 1500 steps at the current η
        if (S.since >= 2000) S.meas.push([S.eta, avgLast(500), S.N, S.L]);
        else S.pendingMeasure = true;
        draw();
      }
      function startSweep() {
        S.meas = []; S.sweep = { etas: VIZ.linspace(0, 5, 11), i: 0 };
        S.eta = 0; sEta.set(0); S.since = 0;
        sweepBtn.setAttribute("aria-pressed", "true");
      }

      const loop = VIZ.loop(ui.el, () => {
        // ~200k pair checks per frame at most
        const fast = S.sweep || S.pendingMeasure;
        const per = fast ? Math.max(1, Math.min(30, Math.floor(2e5 / (S.N * S.N)))) : Math.max(1, Math.min(3, Math.floor(6e4 / (S.N * S.N))));
        for (let k = 0; k < per; k++) {
          step();
          if (S.pendingMeasure && S.since >= 2000) { S.pendingMeasure = false; S.meas.push([S.eta, avgLast(500), S.N, S.L]); }
          if (S.sweep && S.since >= 2000) {
            S.meas.push([S.eta, avgLast(500), S.N, S.L]);
            S.sweep.i++;
            if (S.sweep.i >= S.sweep.etas.length) { S.sweep = null; sweepBtn.setAttribute("aria-pressed", "false"); }
            else { S.eta = S.sweep.etas[S.sweep.i]; sEta.set(S.eta); S.since = 0; }
          }
        }
        draw();
      });

      function draw() { box.draw(); trace.draw(); curve.draw(); readout(); }

      function drawBox(c) {
        const T = VIZ.theme(), ctx = c.ctx, pad = 8, side = Math.min(c.w, c.h) - 2 * pad;
        const ox = (c.w - side) / 2, oy = (c.h - side) / 2, sc = side / S.L;
        ctx.fillStyle = T.panel; ctx.fillRect(ox, oy, side, side);
        ctx.strokeStyle = T.axis; ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, oy + 0.5, side - 1, side - 1);
        // interaction radius for scale
        const rcx = ox + side - sc * R - 6, rcy = oy + sc * R + 6;
        ctx.globalAlpha = 0.6; ctx.strokeStyle = T.axis; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.arc(rcx, rcy, sc * R, 0, TWO_PI); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = T.text; ctx.font = T.font(10, 700); ctx.textAlign = "center";
        ctx.fillText("r = 1", rcx, rcy + 4); ctx.globalAlpha = 1;
        const len = Math.max(5, Math.min(12, sc * 0.35)), light = T.dark ? 65 : 42;
        ctx.lineWidth = 2;
        for (let i = 0; i < S.N; i++) {
          const X = ox + sc * S.x[i], Y = oy + side - sc * S.y[i], a = S.th[i];
          const dx = len * Math.cos(a), dy = -len * Math.sin(a);
          ctx.strokeStyle = `hsl(${(a * 180) / Math.PI}, 80%, ${light}%)`;
          ctx.beginPath(); ctx.moveTo(X - dx / 2, Y - dy / 2); ctx.lineTo(X + dx / 2, Y + dy / 2); ctx.stroke();
          ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(X + dx / 2, Y + dy / 2, 2, 0, TWO_PI); ctx.fill();
        }
      }
      function drawTrace(c) {
        const n1 = Math.max(HIST, S.n);
        const ax = VIZ.axes(c, { xlim: [n1 - HIST, n1], ylim: [0, 1], xlabel: "time step", ylabel: "va" });
        ax.clip();
        ax.line(S.hist.map((p) => p[0]), S.hist.map((p) => p[1]), { color: ax.T.c[0], width: 2 });
        c.ctx.restore();
      }
      function drawCurve(c) {
        const ax = VIZ.axes(c, { xlim: [0, 5], ylim: [0, 1], xlabel: "η", ylabel: "va" });
        const T = ax.T;
        ax.clip();
        const floor = 1 / Math.sqrt(S.N);
        ax.line([0, 5], [floor, floor], { color: T.axis, width: 1.5, dash: [5, 4] });
        ax.line([S.eta, S.eta], [0, 1], { color: T.accent, width: 2, alpha: 0.6 });
        S.meas.forEach(([e, v]) => ax.dot(e, v, 5, T.c[1], T.panel));
        c.ctx.restore();
      }
      function readout() {
        const rho = S.N / (S.L * S.L);
        let t = `N = ${S.N}, L = ${S.L.toFixed(1)}, density ρ = N/L² = ${rho.toFixed(2)}, η = ${S.eta.toFixed(2)} · va = ${S.va.toFixed(2)} · step ${S.n}`;
        if (S.sweep) t += ` · sweeping η, point ${S.sweep.i + 1} of ${S.sweep.etas.length}`;
        if (S.pendingMeasure) t += ` · measuring after ${2000 - S.since} more steps…`;
        ui.readout.textContent = t;
      }

      reset();
      if (VIZ.reduceMotion) { for (let i = 0; i < 600; i++) step(); draw(); }
      else loop.start();
    },
  });
})();

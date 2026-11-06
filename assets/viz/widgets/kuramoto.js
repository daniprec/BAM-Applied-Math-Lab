/* Kuramoto model: N phase oscillators with mean-field coupling.
   θ_i' = ω_i + K r sin(ψ − θ_i),   r e^{iψ} = (1/N) Σ e^{iθ_j}
   Compare measured r(K) with the N → ∞ self-consistency result (Strogatz 2000). */
(function () {
  "use strict";
  const DIST = {
    normal: { label: "Normal, σ = 1", g: (w) => Math.exp(-w * w / 2) / Math.sqrt(2 * Math.PI), Kc: Math.sqrt(8 / Math.PI),
      sample: (u) => u.normal() },
    lorentz: { label: "Lorentzian, γ = 1", g: (w) => 1 / (Math.PI * (1 + w * w)), Kc: 2,
      sample: (u) => Math.tan(Math.PI * (u() - 0.5)) },
  };

  /* r_∞(K) from 1 = K ∫ cos²θ g(K r sinθ) dθ over [−π/2, π/2] */
  function rInf(K, d) {
    if (K <= d.Kc) return 0;
    const F = (r) => {
      const n = 200, hh = Math.PI / n; let s = 0;
      for (let i = 0; i <= n; i++) {
        const th = -Math.PI / 2 + i * hh, wgt = i === 0 || i === n ? 1 : i % 2 ? 4 : 2;
        s += wgt * Math.cos(th) ** 2 * d.g(K * r * Math.sin(th));
      }
      return (K * s * hh) / 3;
    };
    let lo = 1e-6, hi = 1;
    if (F(hi) >= 1) return 1;
    for (let k = 0; k < 50; k++) { const mid = (lo + hi) / 2; if (F(mid) > 1) lo = mid; else hi = mid; }
    return (lo + hi) / 2;
  }

  VIZ.register("kuramoto", {
    title: "Kuramoto model: when does a crowd of oscillators synchronize?",
    init(ui) {
      const S = { K: 1, N: 300, dist: "normal", seed: 7, th: [], w: [], t: 0, hist: [], meas: [], r: 0, psi: 0, sweep: null, kAt: 0, pending: false };

      const sK = VIZ.slider(ui.controls, { label: "<i>K</i>", min: 0, max: 5, step: 0.05, value: S.K,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.K = v; S.sweep = null; S.kAt = S.t; } });
      VIZ.select(ui.controls, { label: "g(ω)", value: S.dist, options: [["normal", DIST.normal.label], ["lorentz", DIST.lorentz.label]],
        onChange: (v) => { S.dist = v; S.meas = []; resample(); } });
      VIZ.select(ui.controls, { label: "N", value: "300", options: [["50", "50"], ["300", "300"], ["1000", "1000"]],
        onChange: (v) => { S.N = +v; resample(); } });
      const g = VIZ.group(ui.controls);
      VIZ.button(g, "Measure r(K)", measure, { primary: true });
      const sweepBtn = VIZ.button(g, "Sweep K automatically", startSweep);
      VIZ.button(g, "New random sample", () => { S.seed++; resample(); });
      VIZ.button(g, "Clear points", () => { S.meas = []; draw(); });

      const circle = ui.panel({ title: "Phases on the circle", aspect: 1, minHeight: 240,
        label: "Oscillators as dots on the unit circle with the order parameter drawn as an arrow", onDraw: drawCircle });
      const trace = ui.panel({ title: "Order parameter  r(t)", aspect: 1, minHeight: 240,
        label: "Recent history of the order parameter r", onDraw: drawTrace });
      const curve = ui.panel({ title: "r against K: your measurements (dots) and the N → ∞ theory (line)", cls: "viz__panel--wide",
        aspect: 2.6, minHeight: 220, label: "Measured order parameter against coupling, with the theoretical curve", onDraw: drawCurve });

      const DT = 0.01;
      const loop = VIZ.loop(ui.el, (dt) => {
        const speed = S.sweep ? 25 : 8;           // time units per second
        const steps = Math.round((speed * dt) / DT);
        for (let s = 0; s < steps; s++) step();
        if (S.sweep) runSweep();
        if (S.pending && S.t - S.kAt >= 25) { S.pending = false; S.meas.push([S.K, avgR(20)]); }
        draw();
      });

      function order() {
        let c = 0, s = 0;
        for (let i = 0; i < S.N; i++) { c += Math.cos(S.th[i]); s += Math.sin(S.th[i]); }
        S.r = Math.hypot(c, s) / S.N; S.psi = Math.atan2(s, c);
      }
      function step() {
        order();
        const kr = S.K * S.r;
        for (let i = 0; i < S.N; i++) S.th[i] += DT * (S.w[i] + kr * Math.sin(S.psi - S.th[i]));
        S.t += DT;
        if (Math.round(S.t / DT) % 10 === 0) { S.hist.push([S.t, S.r]); if (S.hist.length > 600) S.hist.shift(); }
      }
      function resample() {
        const u = VIZ.rng(S.seed * 7919 + S.N), d = DIST[S.dist];
        S.w = Array.from({ length: S.N }, () => d.sample(u));
        const mean = S.w.reduce((a, b) => a + b, 0) / S.N;   // rotating frame at the sample mean
        S.w = S.w.map((x) => x - mean);
        S.th = Array.from({ length: S.N }, () => 2 * Math.PI * u());
        S.t = 0; S.kAt = 0; S.hist = []; order(); draw();
      }
      function avgR(span) {
        // average only after the transient that follows the last change of K
        const from = Math.max(S.t - span, S.kAt + 5);
        const pts = S.hist.filter(([t]) => t >= from);
        return pts.length ? pts.reduce((a, p) => a + p[1], 0) / pts.length : S.r;
      }
      function measure() {
        if (S.t - S.kAt >= 25) S.meas.push([S.K, avgR(20)]);
        else S.pending = true;                       // wait for the transient to die out
        draw();
      }
      function startSweep() {
        S.meas = []; S.pending = false; S.sweep = { i: 0, Ks: VIZ.linspace(0, 4, 17), t0: null };
        sweepBtn.setAttribute("aria-pressed", "true");
      }
      function runSweep() {
        const sw = S.sweep;
        if (sw.t0 === null) { S.K = sw.Ks[sw.i]; sK.set(S.K); sw.t0 = S.t; S.kAt = S.t; }
        if (S.t - sw.t0 > 40) {                      // 20 units transient + 20 units average
          S.meas.push([S.K, avgR(20)]);
          sw.i++; sw.t0 = null;
          if (sw.i >= sw.Ks.length) { S.sweep = null; sweepBtn.setAttribute("aria-pressed", "false"); }
        }
      }
      function draw() { circle.draw(); trace.draw(); curve.draw(); readout(); }

      function drawCircle(c) {
        const T = VIZ.theme(), ctx = c.ctx, cx = c.w / 2, cy = c.h / 2, R = Math.min(c.w, c.h) * 0.4;
        ctx.fillStyle = T.panel; ctx.fillRect(0, 0, c.w, c.h);
        ctx.strokeStyle = T.axis; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
        const kr = S.K * S.r;
        for (let i = 0; i < S.N; i++) {
          const locked = Math.abs(S.w[i]) <= kr;
          ctx.fillStyle = locked ? T.c[0] : T.c[1];
          ctx.globalAlpha = S.N > 500 ? 0.6 : 0.85;
          ctx.beginPath(); ctx.arc(cx + R * Math.cos(S.th[i]), cy - R * Math.sin(S.th[i]), S.N > 500 ? 2.5 : 3.5, 0, 2 * Math.PI); ctx.fill();
        }
        ctx.globalAlpha = 1;
        const ex = cx + R * S.r * Math.cos(S.psi), ey = cy - R * S.r * Math.sin(S.psi);
        ctx.strokeStyle = T.stable; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.fillStyle = T.stable; ctx.beginPath(); ctx.arc(ex, ey, 4.5, 0, 2 * Math.PI); ctx.fill();
        ctx.font = T.font(11, 600); ctx.fillStyle = T.text; ctx.textAlign = "left";
        ctx.fillText("● locked  |ω| ≤ Kr", 8, c.h - 24);
        ctx.fillStyle = T.c[1]; ctx.fillText("● drifting", 8, c.h - 8);
        ctx.fillStyle = T.c[0]; ctx.fillText("●", 8, c.h - 24);
      }
      function drawTrace(c) {
        const t1 = Math.max(60, S.t), ax = VIZ.axes(c, { xlim: [t1 - 60, t1], ylim: [0, 1], xlabel: "t", ylabel: "r" });
        ax.clip();
        ax.line(S.hist.map((p) => p[0]), S.hist.map((p) => p[1]), { color: ax.T.c[0], width: 2 });
        c.ctx.restore();
      }
      function drawCurve(c) {
        const d = DIST[S.dist], ax = VIZ.axes(c, { xlim: [0, 5], ylim: [0, 1], xlabel: "K", ylabel: "r" });
        const T = ax.T;
        ax.clip();
        const Ks = VIZ.linspace(0, 5, 200);
        ax.line(Ks, Ks.map((K) => rInf(K, d)), { color: T.stable, width: 2 });
        ax.line([d.Kc, d.Kc], [0, 1], { color: T.axis, width: 1, dash: [3, 4] });
        ax.label(`K_c = ${d.Kc.toFixed(2)}`, d.Kc, 0.92, { dx: 6 });
        ax.line([S.K, S.K], [0, 1], { color: T.accent, width: 2, alpha: 0.6 });
        S.meas.forEach(([K, r]) => ax.dot(K, r, 5, T.c[1], T.panel));
        c.ctx.restore();
      }
      function readout() {
        const kr = S.K * S.r, locked = S.w.filter((w) => Math.abs(w) <= kr).length;
        const d = DIST[S.dist];
        ui.readout.textContent = `K = ${S.K.toFixed(2)} (K_c = ${d.Kc.toFixed(2)} for N → ∞) · r = ${S.r.toFixed(2)} · ` +
          `${Math.round((100 * locked) / S.N)}% of oscillators locked` + (S.sweep ? ` · sweeping, step ${S.sweep.i + 1} of 17` : "") + (S.pending ? " · measuring once r has settled…" : "");
      }

      resample();
      if (VIZ.reduceMotion) { for (let i = 0; i < 4000; i++) step(); draw(); }
      else loop.start();
    },
  });
})();

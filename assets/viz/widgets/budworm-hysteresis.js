/* Budworm hysteresis: drag the carrying capacity k and watch the state follow
   the stable branch, jump at a fold, and return along a different path.
   dx/dτ = r x (1 - x/k) - x² / (1 + x²) */
(function () {
  "use strict";
  const KMIN = 3, KMAX = 16, XTOP = 14;
  const f = (x, r, k) => r * x * (1 - x / k) - (x * x) / (1 + x * x);
  const fprime = (x, r, k) => r * (1 - (2 * x) / k) - (2 * x) / Math.pow(1 + x * x, 2);
  const kOfX = (x, r) => { const d = 1 - x / (r * (1 + x * x)); return d > 0 ? x / d : NaN; };
  const foldK = (x) => (2 * x * x * x) / (x * x - 1);
  const foldR = (x) => (2 * x * x * x) / Math.pow(1 + x * x, 2);

  /* k-values of the saddle-node folds for a given r (Strogatz §3.7) */
  function folds(r) {
    const xs = VIZ.roots((x) => foldR(x) - r, 1.0005, 400, 12000);
    return xs.map(foldK).sort((a, b) => a - b);
  }

  /* stable branches as polylines of (k, x), split where stability changes */
  function branches(r) {
    const segs = [];
    let cur = null;
    const xs = VIZ.linspace(0.005, XTOP, 2500);
    xs.forEach((x) => {
      const k = kOfX(x, r);
      const ok = isFinite(k) && k <= KMAX + 1;
      const st = ok ? fprime(x, r, k) < 0 : null;
      if (!ok || !cur || cur.stable !== st) {
        if (cur && cur.k.length > 1) segs.push(cur);
        cur = ok ? { stable: st, k: [], x: [] } : null;
      }
      if (cur) { cur.k.push(k); cur.x.push(x); }
    });
    if (cur && cur.k.length > 1) segs.push(cur);
    return segs;
  }

  VIZ.register("budworm-hysteresis", {
    title: "Hysteresis: change the forest slowly, then change it back",
    init(ui) {
      const S = { r: 0.56, k: 5, x: 0.6, trail: [], sweep: 0, speed: 0.8, segs: [], folds: [] };
      relaxTo(400);

      const sK = VIZ.slider(ui.controls, { label: "<i>k</i> (forest)", min: KMIN, max: KMAX, step: 0.01, value: S.k,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.sweep = 0; setK(v); loop.start(); } });
      const sR = VIZ.slider(ui.controls, { label: "<i>r</i>", min: 0.15, max: 0.75, step: 0.005, value: S.r,
        fmt: (v) => v.toFixed(3), onInput: (v) => { S.r = v; recompute(); loop.start(); } });
      const g = VIZ.group(ui.controls);
      const up = VIZ.button(g, "Sweep k up ▶", () => startSweep(+1), { primary: true });
      const down = VIZ.button(g, "◀ Sweep k down", () => startSweep(-1));
      VIZ.button(g, "Reset", reset);
      VIZ.select(ui.controls, { label: "Speed", value: "0.8",
        options: [["0.8", "slow (quasi-static)"], ["6", "fast"]], onChange: (v) => { S.speed = +v; } });

      const bif = ui.panel({ title: "Bifurcation diagram  x* against k", aspect: 1.25, minHeight: 260,
        label: "S-shaped bifurcation diagram of the budworm model with the current state and its history",
        onDraw: drawBif,
        onPointer: (e, c) => { if (c.ax && c.ax.inside(e.x, e.y)) { S.sweep = 0; setK(clampK(c.ax.ix(e.x))); sK.set(S.k); loop.start(); } } });
      const cusp = ui.panel({ title: "Parameter plane  (k, r)  ·  click to move", aspect: 1.25, minHeight: 260,
        label: "Region of bistability in the k-r plane bounded by two fold curves meeting at a cusp",
        onDraw: drawCusp,
        onPointer: (e, c) => {
          if (!c.ax || !c.ax.inside(e.x, e.y)) return;
          S.sweep = 0;
          S.r = Math.min(0.75, Math.max(0.15, c.ax.iy(e.y))); sR.set(S.r);
          setK(clampK(c.ax.ix(e.x))); sK.set(S.k);
          recompute(); loop.start();
        } });

      const loop = VIZ.loop(ui.el, (dt) => {
        if (S.sweep) {
          const nk = clampK(S.k + S.sweep * S.speed * dt);
          if (nk === S.k) S.sweep = 0;
          setK(nk); sK.set(S.k);
        }
        const moved = relaxTo(Math.round(dt * 2400));  // ~120 time units per second
        draw();
        if (!S.sweep && !moved) loop.stop();
      });

      function clampK(k) { return Math.min(KMAX, Math.max(KMIN, k)); }
      function setK(k) {
        const dir = Math.sign(k - S.k);
        S.k = k;
        if (dir) S.trail.push({ k: S.k, x: S.x, dir });
        if (S.trail.length > 6000) S.trail.splice(0, 1000);
      }
      function relaxTo(steps) {
        const x0 = S.x;
        for (let i = 0; i < steps; i++) S.x = Math.max(1e-6, VIZ.rk4((t, y) => [f(y[0], S.r, S.k)], 0, [S.x], 0.05)[0]);
        if (S.trail.length) { const last = S.trail[S.trail.length - 1]; last.x = S.x; }
        return Math.abs(S.x - x0) > 1e-4;
      }
      function startSweep(dir) { S.sweep = dir; loop.start(); }
      function reset() {
        S.sweep = 0; S.trail = []; S.k = 5; sK.set(5); S.x = 0.6; relaxTo(400); draw();
      }
      function recompute() { S.segs = branches(S.r); S.folds = folds(S.r); draw(); }
      function draw() { bif.draw(); cusp.draw(); readout(); }

      function drawBif(c) {
        const ax = (c.ax = VIZ.axes(c, { xlim: [KMIN, KMAX], ylim: [0, XTOP], xlabel: "k", ylabel: "x" }));
        const T = ax.T;
        ax.clip();
        ax.line([KMIN, KMAX], [0, 0], { color: T.unstable, width: 1.5, dash: [5, 4] });
        S.segs.forEach((s) => ax.line(s.k, s.x, s.stable ? { color: T.stable, width: 2.5 } : { color: T.unstable, width: 1.5, dash: [5, 4] }));
        // history: blue while k increases, red while it decreases (as in the notes)
        for (let i = 1; i < S.trail.length; i++) {
          const a = S.trail[i - 1], b = S.trail[i];
          ax.line([a.k, b.k], [a.x, b.x], { color: b.dir > 0 ? T.c[0] : T.c[1], width: 3, alpha: 0.85 });
        }
        S.folds.forEach((kf) => { if (kf <= KMAX) ax.line([kf, kf], [0, XTOP], { color: T.axis, width: 1, dash: [2, 4] }); });
        ax.dot(S.k, S.x, 7, T.accent, T.stable);
        c.ctx.restore();
        VIZ.legend(c, [
          { label: "stable", color: T.stable }, { label: "unstable", color: T.unstable, dash: [5, 4], width: 1.5 },
          { label: "k increasing", color: T.c[0] }, { label: "k decreasing", color: T.c[1] }],
          ax.pad.l + 10, ax.pad.t + 14);
      }

      function drawCusp(c) {
        const ax = (c.ax = VIZ.axes(c, { xlim: [0, 25], ylim: [0.1, 0.75], xlabel: "k", ylabel: "r" }));
        const T = ax.T, ctx = c.ctx;
        ax.clip();
        const xs = VIZ.linspace(1.0008, 60, 3000);
        const K = xs.map(foldK), R = xs.map(foldR);
        ctx.save();
        ctx.beginPath();
        K.forEach((k, i) => (i ? ctx.lineTo(ax.sx(k), ax.sy(R[i])) : ctx.moveTo(ax.sx(k), ax.sy(R[i]))));
        ctx.closePath();
        ctx.fillStyle = T.accent; ctx.globalAlpha = 0.25; ctx.fill();
        ctx.restore();
        ax.line(K, R, { color: T.stable, width: 2 });
        ax.line([0, 25], [S.r, S.r], { color: T.axis, width: 1, dash: [3, 4] });
        ax.dot(3 * Math.sqrt(3), (3 * Math.sqrt(3)) / 8, 5, T.unstable);
        ax.label("cusp", 3 * Math.sqrt(3), (3 * Math.sqrt(3)) / 8, { dx: -6, dy: -9, align: "right" });
        ax.label("bistable", 17, 0.38, { color: T.text, size: 12, align: "center" });
        ax.label("outbreak only", 14, 0.68, { size: 11, align: "center" });
        ax.label("refuge only", 14, 0.16, { size: 11, align: "center" });
        ax.dot(Math.min(S.k, 25), S.r, 7, T.accent, T.stable);
        ctx.restore();
      }

      function readout() {
        const where = S.x > 2.5 ? "outbreak branch (high)" : "refuge branch (low)";
        let range;
        if (S.folds.length >= 2) range = `bistable for ${S.folds[0].toFixed(2)} < k < ${S.folds[1].toFixed(2)}`;
        else if (S.folds.length === 1) range = `bistable for k > ${S.folds[0].toFixed(2)}`;
        else range = "no bistability for this r (above the cusp)";
        ui.readout.textContent = `k = ${S.k.toFixed(2)}, r = ${S.r.toFixed(3)} · state x = ${S.x.toFixed(2)} on the ${where} · ${range}.`;
      }

      recompute();
    },
  });
})();

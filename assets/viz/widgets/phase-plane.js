/* Phase plane explorer for planar systems.
   ::: {.viz data-viz="phase-plane" data-model="vdp"} :::   Van der Pol (Liénard form)
   ::: {.viz data-viz="phase-plane" data-model="fhn"} :::   FitzHugh-Nagumo (course version)
   Click anywhere in the plane to launch a trajectory. */
(function () {
  "use strict";

  const MODELS = {
    vdp: {
      title: "Van der Pol oscillator: one limit cycle, many starting points",
      xlabel: "x", ylabel: "y", xlim: [-3, 3], ylim: [-3, 3],
      params: [{ key: "mu", label: "<i>μ</i>", min: 0.1, max: 5, step: 0.05, value: 1 }],
      // x' = μ (y - F(x)),  y' = -x/μ,  F(x) = x³/3 - x   (amlab.odes_2d.van_der_pol)
      f: (x, y, p) => [p.mu * (y - (x * x * x / 3 - x)), -x / p.mu],
      nullclines: (p) => [
        { label: "x-nullcline  y = x³/3 − x", fn: (x) => x * x * x / 3 - x },
        { label: "y-nullcline  x = 0", vertical: 0 },
      ],
      fixed: () => [[0, 0]],
      dt: 0.004, speed: 5, tmax: 40, starts: [[0.1, 0], [2.8, 2.6]],
    },
    fhn: {
      title: "FitzHugh–Nagumo: from excitable to oscillating",
      xlabel: "v", ylabel: "w", xlim: [-0.5, 1.2], ylim: [-0.2, 0.9],
      params: [
        { key: "I", label: "<i>I</i><sub>app</sub>", min: 0, max: 1.5, step: 0.005, value: 0 },
        { key: "eps", label: "<i>ε</i>", min: 0.005, max: 0.2, step: 0.005, value: 0.01 },
      ],
      consts: { alpha: 0.1, gamma: 0.5 },
      // ε v' = v(1-v)(v-α) - w + I,  w' = v - γ w   (amlab.odes_2d.fitzhugh_nagumo)
      f: (v, w, p) => [(v * (1 - v) * (v - 0.1) - w + p.I) / p.eps, v - 0.5 * w],
      nullclines: (p) => [
        { label: "v-nullcline  w = f(v) + I", fn: (v) => v * (1 - v) * (v - 0.1) + p.I },
        { label: "w-nullcline  w = v/γ", fn: (v) => v / 0.5 },
      ],
      fixed: (p) => VIZ.roots((v) => v * (1 - v) * (v - 0.1) + p.I - v / 0.5, -2, 2, 4000).map((v) => [v, v / 0.5]),
      dt: 0.0004, speed: 1, tmax: 6, starts: [[0.2, 0]],
    },
  };

  function classify(m, p, x, y) {
    const h = 1e-6, f0 = m.f(x, y, p), fx = m.f(x + h, y, p), fy = m.f(x, y + h, p);
    const a = (fx[0] - f0[0]) / h, b = (fy[0] - f0[0]) / h, c = (fx[1] - f0[1]) / h, d = (fy[1] - f0[1]) / h;
    const tr = a + d, det = a * d - b * c, disc = tr * tr - 4 * det;
    let type;
    if (det < 0) type = "saddle";
    else if (Math.abs(tr) < 1e-9) type = "centre (linear test inconclusive)";
    else type = (tr < 0 ? "stable " : "unstable ") + (disc < 0 ? "spiral" : "node");
    return { tr, det, type, stable: det > 0 && tr < 0 };
  }

  VIZ.register("phase-plane", {
    title: "Phase plane",
    init(ui, el) {
      const m = MODELS[el.dataset.model || "vdp"];
      ui.el.querySelector(".viz__title").textContent = m.title;
      const p = {};
      m.params.forEach((q) => { p[q.key] = q.value; });
      let trajs = [], latest = null;

      m.params.forEach((q) => VIZ.slider(ui.controls, { label: q.label, min: q.min, max: q.max, step: q.step, value: q.value,
        fmt: (v) => (q.step < 0.01 ? v.toFixed(3) : v.toFixed(2)), onInput: (v) => { p[q.key] = v; restartAll(); } }));
      const g = VIZ.group(ui.controls);
      VIZ.button(g, "Replay", restartAll, { primary: true });
      VIZ.button(g, "Clear", () => { trajs = []; latest = null; draw(); });

      const plane = ui.panel({ title: `Phase plane (${m.xlabel}, ${m.ylabel})  ·  click to launch a trajectory`, aspect: 1.25, minHeight: 280,
        label: "Phase plane with vector field, nullclines, fixed points and trajectories",
        onDraw: drawPlane,
        onPointer: (e, c) => {
          if (e.type !== "pointerdown" || !c.ax || !c.ax.inside(e.x, e.y)) return;
          add([c.ax.ix(e.x), c.ax.iy(e.y)]);
        } });
      const series = ui.panel({ title: `Time series  ${m.xlabel}(t)  of the latest trajectory`, aspect: 1.25, minHeight: 280,
        label: "Time series of the first variable for the most recent trajectory", onDraw: drawSeries });

      const loop = VIZ.loop(ui.el, (dt) => {
        let active = false;
        trajs.forEach((tr) => {
          if (tr.t >= m.tmax) return;
          active = true;
          const steps = Math.ceil((m.speed * dt) / m.dt);
          for (let i = 0; i < steps && tr.t < m.tmax; i++) {
            tr.s = VIZ.rk4((t, y) => m.f(y[0], y[1], p), tr.t, tr.s, m.dt);
            tr.t += m.dt;
            if (i % 4 === 0) { tr.xs.push(tr.s[0]); tr.ys.push(tr.s[1]); tr.ts.push(tr.t); }
          }
          if (!isFinite(tr.s[0]) || Math.abs(tr.s[0]) > 1e3) tr.t = m.tmax;
        });
        draw();
        if (!active) loop.stop();
      });

      function newTraj(s0) { return { s0, s: s0.slice(), t: 0, xs: [s0[0]], ys: [s0[1]], ts: [0] }; }
      function finish(tr) {   // reduced motion: compute the whole trajectory at once
        while (tr.t < m.tmax) {
          tr.s = VIZ.rk4((t, y) => m.f(y[0], y[1], p), tr.t, tr.s, m.dt); tr.t += m.dt;
          tr.xs.push(tr.s[0]); tr.ys.push(tr.s[1]); tr.ts.push(tr.t);
        }
      }
      function add(s0) {
        latest = newTraj(s0);
        trajs.push(latest);
        if (trajs.length > 6) trajs.shift();
        if (VIZ.reduceMotion) finish(latest); else loop.start();
        draw();
      }
      function restartAll() {
        trajs = trajs.map((tr) => newTraj(tr.s0));
        latest = trajs[trajs.length - 1] || null;
        if (VIZ.reduceMotion) trajs.forEach(finish); else loop.start();
        draw();
      }
      function draw() { plane.draw(); series.draw(); readout(); }

      function drawPlane(c) {
        const ax = (c.ax = VIZ.axes(c, { xlim: m.xlim, ylim: m.ylim, xlabel: m.xlabel, ylabel: m.ylabel }));
        const T = ax.T, ctx = c.ctx;
        ax.clip();
        // direction field: unit arrows
        const nx = 17, ny = 13;
        ctx.save(); ctx.strokeStyle = T.axis; ctx.globalAlpha = 0.55; ctx.lineWidth = 1;
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
          const x = m.xlim[0] + ((i + 0.5) / nx) * (m.xlim[1] - m.xlim[0]);
          const y = m.ylim[0] + ((j + 0.5) / ny) * (m.ylim[1] - m.ylim[0]);
          const d = m.f(x, y, p);
          let dx = ax.sx(x + d[0]) - ax.sx(x), dy = ax.sy(y + d[1]) - ax.sy(y);
          const L = Math.hypot(dx, dy) || 1, len = Math.min(ax.pw / nx, ax.ph / ny) * 0.38;
          dx *= len / L; dy *= len / L;
          const X = ax.sx(x), Y = ax.sy(y);
          ctx.beginPath(); ctx.moveTo(X - dx, Y - dy); ctx.lineTo(X + dx, Y + dy);
          const a = Math.atan2(dy, dx);
          ctx.lineTo(X + dx - 4 * Math.cos(a - 0.5), Y + dy - 4 * Math.sin(a - 0.5));
          ctx.moveTo(X + dx, Y + dy);
          ctx.lineTo(X + dx - 4 * Math.cos(a + 0.5), Y + dy - 4 * Math.sin(a + 0.5));
          ctx.stroke();
        }
        ctx.restore();
        // nullclines
        const ncl = m.nullclines(p), cols = [T.c[0], T.c[1]];
        ncl.forEach((n, i) => {
          if (n.vertical != null) ax.line([n.vertical, n.vertical], m.ylim, { color: cols[i], width: 2, dash: [7, 4] });
          else { const xs = VIZ.linspace(m.xlim[0], m.xlim[1], 400); ax.line(xs, xs.map(n.fn), { color: cols[i], width: 2, dash: [7, 4] }); }
        });
        // trajectories
        trajs.forEach((tr) => {
          ax.line(tr.xs, tr.ys, { color: T.stable, width: tr === latest ? 2.2 : 1.4, alpha: tr === latest ? 1 : 0.6 });
          ax.dot(tr.s0[0], tr.s0[1], 3.5, T.stable);
          ax.dot(tr.s[0], tr.s[1], 5, T.accent, T.stable);
        });
        // fixed points
        m.fixed(p).forEach(([x, y]) => ax.fixed(x, y, classify(m, p, x, y).stable));
        ctx.restore();
        VIZ.legend(c, ncl.map((n, i) => ({ label: n.label, color: cols[i], dash: [7, 4] })), ax.pad.l + 10, ax.pad.t + 14);
      }

      function drawSeries(c) {
        const ax = VIZ.axes(c, { xlim: [0, m.tmax], ylim: m.xlim, xlabel: "t", ylabel: m.xlabel });
        ax.clip();
        if (latest) ax.line(latest.ts, latest.xs, { color: ax.T.c[0], width: 2 });
        c.ctx.restore();
        if (!latest) ax.label("Click in the phase plane", m.tmax / 2, (m.xlim[0] + m.xlim[1]) / 2, { align: "center", size: 12 });
      }

      function readout() {
        const fps = m.fixed(p).map(([x, y]) => {
          const k = classify(m, p, x, y);
          return `(${x.toFixed(2)}, ${y.toFixed(2)}): ${k.type}, tr J = ${k.tr.toFixed(2)}, det J = ${k.det.toFixed(2)}`;
        });
        ui.readout.textContent = "Fixed point " + fps.join("; ") + ".";
      }

      m.starts.forEach((s) => add(s));
    },
  });
})();

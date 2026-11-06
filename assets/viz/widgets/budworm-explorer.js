/* Spruce budworm explorer: rate plot + trajectories.
   dx/dτ = r x (1 - x/k) - x² / (1 + x²)        (dimensionless, Strogatz §3.7) */
(function () {
  "use strict";
  const XMAX = 12, TMAX = 60, DT = 0.05;

  const f = (x, r, k) => r * x * (1 - x / k) - (x * x) / (1 + x * x);
  const fprime = (x, r, k) => r * (1 - (2 * x) / k) - (2 * x) / Math.pow(1 + x * x, 2);

  function fixedPoints(r, k) {
    const g = (x) => r * (1 - x / k) - x / (1 + x * x); // f(x) = x g(x)
    const pos = VIZ.roots(g, 1e-6, XMAX, 2400);
    return [{ x: 0, stable: false }].concat(pos.map((x) => ({ x, stable: fprime(x, r, k) < 0 })));
  }

  function integrate(x0, r, k) {
    const n = Math.round(TMAX / DT) + 1, ts = new Array(n), xs = new Array(n);
    let x = x0;
    for (let i = 0; i < n; i++) {
      ts[i] = i * DT; xs[i] = x;
      x = VIZ.rk4((t, y) => [f(y[0], r, k)], 0, [x], DT)[0];
    }
    return { ts, xs };
  }

  VIZ.register("budworm-explorer", {
    title: "Spruce budworm: fixed points and basins of attraction",
    init(ui) {
      const S = { r: 0.5, k: 10, x0s: [], trajs: [], fps: [], anim: null };

      const sR = VIZ.slider(ui.controls, { label: "<i>r</i>", min: 0.1, max: 0.9, step: 0.005, value: S.r,
        fmt: (v) => v.toFixed(3), onInput: (v) => { S.r = v; update(); } });
      const sK = VIZ.slider(ui.controls, { label: "<i>k</i>", min: 1, max: 12, step: 0.05, value: S.k,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.k = v; update(); } });
      const g = VIZ.group(ui.controls);
      VIZ.button(g, "Fan of starts", () => { S.x0s = [0.3, 1, 1.8, 2.3, 3.5, 5, 7, 9, 11.5]; update(); });
      VIZ.button(g, "Clear", () => { S.x0s = []; S.anim = null; update(); });

      const rate = ui.panel({
        title: "Rate  dx/dτ = f(x)  ·  click to start a trajectory", aspect: 1.35, minHeight: 240,
        label: "Graph of the budworm growth rate against population, with fixed points marked",
        onDraw: drawRate,
        onPointer: (e, c) => {
          if (e.type !== "pointerdown" || !c.ax || !c.ax.inside(e.x, e.y)) return;
          const x0 = Math.min(XMAX, Math.max(0.02, c.ax.ix(e.x)));
          S.x0s.push(x0);
          if (S.x0s.length > 10) S.x0s.shift();
          compute();
          S.anim = { i: S.trajs.length - 1, t: 0 };
          if (VIZ.reduceMotion) S.anim = null; else loop.start();
          redraw();
        },
      });
      const time = ui.panel({
        title: "Trajectories  x(τ)", aspect: 1.35, minHeight: 240,
        label: "Time series of the budworm population for each chosen initial condition",
        onDraw: drawTime,
      });

      const loop = VIZ.loop(ui.el, (dt) => {
        if (!S.anim) { loop.stop(); return; }
        S.anim.t += dt / 2.2;            // reveal a trajectory in about 2 s
        if (S.anim.t >= 1) { S.anim = null; loop.stop(); }
        redraw();
      });

      function compute() {
        S.fps = fixedPoints(S.r, S.k);
        const stable = S.fps.filter((p) => p.stable).map((p) => p.x);
        S.trajs = S.x0s.map((x0) => {
          const tr = integrate(x0, S.r, S.k);
          const end = tr.xs[tr.xs.length - 1];
          // colour by attractor: lowest stable state (refuge) vs highest (outbreak)
          let which = 0;
          if (stable.length > 1 && Math.abs(end - stable[stable.length - 1]) < Math.abs(end - stable[0])) which = 1;
          if (stable.length === 1) which = stable[0] > 2.5 ? 1 : 0;
          tr.which = which; tr.x0 = x0;
          return tr;
        });
      }
      function update() { S.anim = null; compute(); redraw(); readout(); }
      function redraw() { rate.draw(); time.draw(); }

      function drawRate(c) {
        const xs = VIZ.linspace(0, XMAX, 600), ys = xs.map((x) => f(x, S.r, S.k));
        let lo = Math.min(...ys), hi = Math.max(...ys);
        lo = Math.min(lo, -0.2); hi = Math.max(hi, 0.2);
        const padY = 0.12 * (hi - lo);
        const ax = (c.ax = VIZ.axes(c, { xlim: [0, XMAX], ylim: [lo - padY, hi + padY], xlabel: "x", ylabel: "dx/dτ" }));
        const T = ax.T, ctx = c.ctx;
        ax.clip();
        ax.line([0, XMAX], [0, 0], { color: T.axis, width: 1 });
        ax.line(xs, ys, { color: T.c[0], width: 2.5 });
        // phase-line arrows between fixed points
        const fpx = S.fps.map((p) => p.x).concat([XMAX]);
        for (let i = 0; i < fpx.length - 1; i++) {
          const a = fpx[i], b = fpx[i + 1];
          if (b - a < 0.35) continue;
          const m = 0.5 * (a + b), dir = Math.sign(f(m, S.r, S.k));
          const X = ax.sx(m), Y = ax.sy(0);
          ctx.fillStyle = T.text;
          ctx.beginPath();
          ctx.moveTo(X + 7 * dir, Y); ctx.lineTo(X - 5 * dir, Y - 5); ctx.lineTo(X - 5 * dir, Y + 5);
          ctx.closePath(); ctx.fill();
        }
        S.fps.forEach((p) => ax.fixed(p.x, 0, p.stable));
        // moving state during the reveal animation
        if (S.anim) {
          const tr = S.trajs[S.anim.i];
          const j = Math.min(tr.xs.length - 1, Math.floor(S.anim.t * (tr.xs.length - 1)));
          ax.dot(tr.xs[j], 0, 5, T.c[tr.which ? 1 : 0]);
        }
        ctx.restore();
      }

      function drawTime(c) {
        const ax = VIZ.axes(c, { xlim: [0, TMAX], ylim: [0, XMAX], xlabel: "τ", ylabel: "x" });
        const T = ax.T;
        ax.clip();
        S.fps.forEach((p) => {
          if (p.x <= 0) return;
          ax.line([0, TMAX], [p.x, p.x], { color: p.stable ? T.stable : T.unstable, width: 1, dash: p.stable ? [] : [4, 4], alpha: 0.7 });
        });
        S.trajs.forEach((tr, i) => {
          let n = tr.ts.length;
          if (S.anim && S.anim.i === i) n = Math.max(2, Math.floor(S.anim.t * n));
          ax.line(tr.ts.slice(0, n), tr.xs.slice(0, n), { color: T.c[tr.which ? 1 : 0], width: 2 });
        });
        c.ctx.restore();
        if (!S.trajs.length) {
          ax.label("Click on the rate plot to choose x(0)", TMAX / 2, XMAX / 2, { align: "center", size: 12 });
        } else {
          VIZ.legend(c, [{ label: "→ refuge (low)", color: T.c[0] }, { label: "→ outbreak (high)", color: T.c[1] }],
            ax.pad.l + 10, ax.pad.t + 14);
        }
      }

      function readout() {
        const pos = S.fps.filter((p) => p.x > 0);
        const stable = pos.filter((p) => p.stable).length;
        const list = pos.map((p) => `${p.x.toFixed(2)} (${p.stable ? "stable" : "unstable"})`).join(", ");
        const verdict = stable >= 2 ? "Bistable: two stable states, separated by the unstable threshold."
          : pos.length ? "One stable state: every positive start ends there." : "No positive fixed point.";
        ui.readout.textContent = `Positive fixed points: ${list || "none"}. ${verdict}`;
      }

      update();
    },
  });
})();

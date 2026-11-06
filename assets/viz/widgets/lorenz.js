/* Lorenz system: sensitive dependence on initial conditions.
   x' = s(y − x),  y' = x(r − z) − y,  z' = xy − b z
   Same scheme as the lab (amlab.extra.lorenz_attractor.simulate_lorenz):
   forward Euler, dt = 0.01, 3000 steps, start (0, 1, 1.05); run B adds δ to z. */
(function () {
  "use strict";
  const DT = 0.01, STEPS = 3000, TMAX = STEPS * DT;

  function simulate(s, r, b, z0) {
    const X = new Float64Array(STEPS + 1), Y = new Float64Array(STEPS + 1), Z = new Float64Array(STEPS + 1);
    let x = 0, y = 1, z = z0;
    for (let k = 0; k <= STEPS; k++) {
      X[k] = x; Y[k] = y; Z[k] = z;
      const dx = s * (y - x), dy = x * (r - z) - y, dz = x * y - b * z;
      x += DT * dx; y += DT * dy; z += DT * dz;
    }
    return { X, Y, Z };
  }

  VIZ.register("lorenz", {
    title: "Lorenz system: two almost identical starts",
    init(ui) {
      const S = { s: 10, r: 28, b: 8 / 3, lgd: -5, proj: "xz", A: null, B: null, d: null, k: 0 };

      VIZ.slider(ui.controls, { label: "<i>s</i>", min: 1, max: 20, step: 0.5, value: S.s, fmt: (v) => v.toFixed(1), onInput: (v) => { S.s = v; restart(); } });
      VIZ.slider(ui.controls, { label: "<i>r</i>", min: 0.5, max: 40, step: 0.5, value: S.r, fmt: (v) => v.toFixed(1), onInput: (v) => { S.r = v; restart(); } });
      VIZ.slider(ui.controls, { label: "<i>b</i>", min: 0.5, max: 5, step: 0.01, value: S.b, fmt: (v) => v.toFixed(2), onInput: (v) => { S.b = v; restart(); } });
      VIZ.slider(ui.controls, { label: "initial gap <i>δ</i>", min: -10, max: -1, step: 0.5, value: S.lgd,
        fmt: (v) => "1e" + v.toFixed(1).replace(".0", ""), onInput: (v) => { S.lgd = v; restart(); } });
      VIZ.select(ui.controls, { label: "View", value: S.proj, options: [["xz", "x–z plane"], ["xy", "x–y plane"]],
        onChange: (v) => { S.proj = v; draw(); } });
      VIZ.button(ui.controls, "Replay", restart, { primary: true });

      const traj = ui.panel({ title: "Run A and run B (start differs by δ in z)", aspect: 1.15, minHeight: 260,
        label: "Projection of two Lorenz trajectories that start very close together", onDraw: drawTraj });
      const sep = ui.panel({ title: "Separation |A − B| on a log scale", aspect: 1.15, minHeight: 260,
        label: "Logarithm of the distance between the two trajectories against time, with a reference slope", onDraw: drawSep });

      const loop = VIZ.loop(ui.el, (dt) => {
        S.k = Math.min(STEPS, S.k + Math.round(dt * 500));     // 30 time units in about 6 s
        draw();
        if (S.k >= STEPS) loop.stop();
      });

      function restart() {
        const d = Math.pow(10, S.lgd);
        S.A = simulate(S.s, S.r, S.b, 1.05);
        S.B = simulate(S.s, S.r, S.b, 1.05 + d);
        S.d = new Float64Array(STEPS + 1);
        for (let k = 0; k <= STEPS; k++) S.d[k] = Math.hypot(S.A.X[k] - S.B.X[k], S.A.Y[k] - S.B.Y[k], S.A.Z[k] - S.B.Z[k]);
        S.k = VIZ.reduceMotion ? STEPS : 0;
        draw();
        if (!VIZ.reduceMotion) loop.start();
      }
      function draw() { traj.draw(); sep.draw(); readout(); }

      function drawTraj(c) {
        const xz = S.proj === "xz";
        const ax = VIZ.axes(c, { xlim: [-25, 25], ylim: xz ? [0, 55] : [-30, 30], xlabel: "x", ylabel: xz ? "z" : "y" });
        const T = ax.T, ctx = c.ctx, n = S.k;
        ax.clip();
        [[S.A, T.c[0]], [S.B, T.c[1]]].forEach(([R, col], j) => {
          const V = xz ? R.Z : R.Y;
          ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.1; ctx.globalAlpha = j ? 0.85 : 0.75;
          ctx.beginPath();
          for (let k = 0; k <= n; k++) { const px = ax.sx(R.X[k]), py = ax.sy(V[k]); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
          ctx.stroke(); ctx.restore();
          ax.dot(R.X[n], V[n], 5, col, T.panel);
        });
        ctx.restore();
        VIZ.legend(c, [{ label: "run A", color: T.c[0] }, { label: "run B", color: T.c[1] }], ax.pad.l + 10, ax.pad.t + 14);
      }

      function drawSep(c) {
        const ax = VIZ.axes(c, { xlim: [0, TMAX], ylim: [-12, 2], xlabel: "t", ylabel: "separation (log scale)",
          yticks: [-12, -10, -8, -6, -4, -2, 0, 2], fmtY: (v) => "1e" + v });
        const T = ax.T, n = S.k;
        ax.clip();
        // reference growth e^{0.9 t} from the initial gap
        const d0 = Math.pow(10, S.lgd), ts = VIZ.linspace(0, TMAX, 100);
        ax.line(ts, ts.map((t) => Math.log10(d0) + (0.9 * t) / Math.LN10), { color: T.axis, width: 1.5, dash: [6, 4] });
        const tt = [], yy = [];
        for (let k = 0; k <= n; k += 2) { tt.push(k * DT); yy.push(Math.log10(Math.max(S.d[k], 1e-16))); }
        ax.line(tt, yy, { color: T.c[1], width: 2 });
        c.ctx.restore();
        VIZ.legend(c, [{ label: "|A − B|", color: T.c[1] }, { label: "slope λ ≈ 0.9", color: T.axis, dash: [6, 4], width: 1.5 }],
          ax.pad.l + ax.pw - 120, ax.pad.t + ax.ph - 30);
      }

      function readout() {
        let kh = -1;
        for (let k = 0; k <= STEPS; k++) if (S.d[k] >= 1) { kh = k; break; }
        const horizon = kh < 0 ? "the runs never separate by more than 1 within t = 30" : `the runs are 1 apart by t ≈ ${(kh * DT).toFixed(1)}`;
        const fixedStable = S.r < (S.s * (S.s + S.b + 3)) / (S.s - S.b - 1) && S.s > S.b + 1;
        ui.readout.textContent = `t = ${(S.k * DT).toFixed(1)} · gap now ${S.d[S.k].toExponential(1)} · ${horizon}.` +
          (S.r < 1 ? " r < 1: the origin attracts everything." : fixedStable ? " Here r < r_H = s(s + b + 3)/(s − b − 1), so the fixed points C± are linearly stable." : "");
      }

      restart();
    },
  });
})();

/* One-dimensional bifurcations: the four normal forms of Strogatz, Ch. 3.
   Left: the flow ẋ = f(x; μ) with its fixed points (phase line on the x-axis).
   Right: the bifurcation diagram x* against μ, with the current μ marked. */
(function () {
  "use strict";
  const XL = 2, ML = 1;

  const FORMS = {
    sn: {
      label: "Saddle-node  ẋ = μ + x²",
      f: (x, m) => m + x * x, fp: (x) => 2 * x,
      fixed: (m) => (m < 0 ? [-Math.sqrt(-m), Math.sqrt(-m)] : m === 0 ? [0] : []),
      branches: [
        { x: (m) => -Math.sqrt(-m), dom: [-ML, 0] },
        { x: (m) => Math.sqrt(-m), dom: [-ML, 0] },
      ],
    },
    tc: {
      label: "Transcritical  ẋ = μx − x²",
      f: (x, m) => m * x - x * x, fp: (x, m) => m - 2 * x,
      fixed: (m) => (m === 0 ? [0] : [0, m]),
      branches: [{ x: () => 0, dom: [-ML, ML] }, { x: (m) => m, dom: [-ML, ML] }],
    },
    pf: {
      label: "Supercritical pitchfork  ẋ = μx − x³",
      f: (x, m) => m * x - x * x * x, fp: (x, m) => m - 3 * x * x,
      fixed: (m) => (m > 0 ? [-Math.sqrt(m), 0, Math.sqrt(m)] : [0]),
      branches: [
        { x: () => 0, dom: [-ML, ML] },
        { x: (m) => Math.sqrt(m), dom: [0, ML] },
        { x: (m) => -Math.sqrt(m), dom: [0, ML] },
      ],
    },
    pfs: {
      label: "Subcritical pitchfork  ẋ = μx + x³",
      f: (x, m) => m * x + x * x * x, fp: (x, m) => m + 3 * x * x,
      fixed: (m) => (m < 0 ? [-Math.sqrt(-m), 0, Math.sqrt(-m)] : [0]),
      branches: [
        { x: () => 0, dom: [-ML, ML] },
        { x: (m) => Math.sqrt(-m), dom: [-ML, 0] },
        { x: (m) => -Math.sqrt(-m), dom: [-ML, 0] },
      ],
    },
  };

  VIZ.register("bifurcations-1d", {
    title: "Bifurcations in one dimension",
    init(ui, el) {
      const S = { form: el.dataset.model || "sn", mu: -0.5 };

      VIZ.select(ui.controls, { label: "Normal form", value: S.form,
        options: Object.entries(FORMS).map(([k, v]) => [k, v.label]),
        onChange: (v) => { S.form = v; draw(); } });
      VIZ.slider(ui.controls, { label: "<i>μ</i>", min: -ML, max: ML, step: 0.01, value: S.mu,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.mu = Math.abs(v) < 1e-9 ? 0 : v; draw(); } });

      const flow = ui.panel({ title: "Flow  ẋ = f(x; μ)  and phase line", aspect: 1.3, minHeight: 240,
        label: "Graph of the vector field f(x) with fixed points and flow arrows", onDraw: drawFlow });
      const diag = ui.panel({ title: "Bifurcation diagram  x* against μ", aspect: 1.3, minHeight: 240,
        label: "Fixed points as a function of mu, solid when stable and dashed when unstable", onDraw: drawDiag });

      function kind(x, m) {
        const F = FORMS[S.form], d = F.fp(x, m);
        if (Math.abs(d) < 1e-9) {          // linearization fails: look at the sign of f on each side
          const l = F.f(x - 1e-3, m), r = F.f(x + 1e-3, m);
          if (l > 0 && r < 0) return "stable (weakly)";
          if (l < 0 && r > 0) return "unstable (weakly)";
          return "half";
        }
        return d < 0 ? "stable" : "unstable";
      }
      function draw() { flow.draw(); diag.draw(); readout(); }

      function drawFlow(c) {
        const F = FORMS[S.form], m = S.mu;
        const xs = VIZ.linspace(-XL, XL, 400), ys = xs.map((x) => F.f(x, m));
        const ax = VIZ.axes(c, { xlim: [-XL, XL], ylim: [-2.5, 2.5], xlabel: "x", ylabel: "ẋ" });
        const T = ax.T, ctx = c.ctx;
        ax.clip();
        ax.line([-XL, XL], [0, 0], { color: T.axis, width: 1 });
        ax.line(xs, ys, { color: T.c[0], width: 2.5 });
        // arrows between fixed points
        const fps = F.fixed(m);
        const cuts = [-XL].concat(fps, [XL]);
        for (let i = 0; i < cuts.length - 1; i++) {
          const a = cuts[i], b = cuts[i + 1];
          if (b - a < 0.2) continue;
          const mid = 0.5 * (a + b), dir = Math.sign(F.f(mid, m));
          if (!dir) continue;
          const X = ax.sx(mid), Y = ax.sy(0);
          ctx.fillStyle = T.text;
          ctx.beginPath(); ctx.moveTo(X + 7 * dir, Y); ctx.lineTo(X - 5 * dir, Y - 5); ctx.lineTo(X - 5 * dir, Y + 5); ctx.closePath(); ctx.fill();
        }
        fps.forEach((x) => {
          const k = kind(x, m);
          if (k === "half") {   // half-stable: half filled
            ax.dot(x, 0, 6, T.panel, T.unstable);
            ctx.save(); ctx.beginPath(); ctx.arc(ax.sx(x), ax.sy(0), 6, Math.PI / 2, 1.5 * Math.PI); ctx.fillStyle = T.stable; ctx.fill(); ctx.restore();
          } else ax.fixed(x, 0, k.startsWith("stable"));
        });
        ctx.restore();
      }

      function drawDiag(c) {
        const F = FORMS[S.form];
        const ax = VIZ.axes(c, { xlim: [-ML, ML], ylim: [-1.5, 1.5], xlabel: "μ", ylabel: "x*" });
        const T = ax.T;
        ax.clip();
        F.branches.forEach((br) => {
          const ms = VIZ.linspace(br.dom[0], br.dom[1], 300);
          // split each branch where its stability changes
          let seg = { m: [], x: [], st: null };
          const flush = () => { if (seg.m.length > 1) ax.line(seg.m, seg.x, seg.st ? { color: T.stable, width: 2.5 } : { color: T.unstable, width: 1.8, dash: [6, 4] }); };
          ms.forEach((m) => {
            const x = br.x(m), st = F.fp(x, m) < 0;
            if (seg.st !== null && st !== seg.st) { const lm = seg.m[seg.m.length - 1], lx = seg.x[seg.x.length - 1]; flush(); seg = { m: [lm], x: [lx], st }; }
            seg.st = st; seg.m.push(m); seg.x.push(x);
          });
          flush();
        });
        ax.line([S.mu, S.mu], [-1.5, 1.5], { color: T.accent, width: 2, alpha: 0.7 });
        F.fixed(S.mu).forEach((x) => { if (Math.abs(x) <= 1.5) ax.fixed(S.mu, x, kind(x, S.mu).startsWith("stable")); });
        c.ctx.restore();
        VIZ.legend(c, [{ label: "stable", color: T.stable }, { label: "unstable", color: T.unstable, dash: [6, 4], width: 1.8 }],
          ax.pad.l + 10, ax.pad.t + 14);
      }

      function readout() {
        const F = FORMS[S.form], m = S.mu, fps = F.fixed(m);
        if (!fps.length) { ui.readout.textContent = `μ = ${m.toFixed(2)}: no fixed points. Every trajectory runs off to +∞.`; return; }
        const parts = fps.map((x) => {
          const k = kind(x, m), d = F.fp(x, m);
          return k === "half" ? `x* = ${x.toFixed(2)} (half-stable, f′ = 0)`
            : k.includes("weakly") ? `x* = ${x.toFixed(2)} (${k}: f′ = 0, so the linearization says nothing)`
            : `x* = ${x.toFixed(2)} (${k}, ${k === "stable" ? "decay" : "escape"} time 1/|f′| = ${(1 / Math.abs(d)).toFixed(2)})`;
        });
        ui.readout.textContent = `μ = ${m.toFixed(2)}: ` + parts.join("; ") + ".";
      }

      draw();
    },
  });
})();

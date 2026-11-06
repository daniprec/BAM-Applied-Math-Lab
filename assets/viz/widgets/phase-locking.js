/* Phase locking of a firefly to a periodic stimulus (Strogatz §4.5).
   Θ' = Ω,   θ' = ω + A sin(Θ − θ),   φ = Θ − θ  ⇒  φ' = Ω − ω − A sin φ.
   Integrated with RK4, dt = 0.05, from φ(0) = 0, as amlab.synchronization.phase_models.firefly_phase_difference. */
(function () {
  "use strict";
  const OMEGA = 1, DT = 0.05, SPEED = 4, WINDOW = 120;   // speed in time units per second
  const TWO_PI = 2 * Math.PI;
  const fmtPi = (v) => {
    const k = Math.round((v / Math.PI) * 2);
    return ["0", "π/2", "π", "3π/2", "2π"][k] || VIZ.fmt(v);
  };

  VIZ.register("phase-locking", {
    title: "Phase locking: a firefly and a flashing light",
    init(ui) {
      const S = { W: 1.15, A: 0.2, t: 0, Th: 0, th: 0, hist: [], flashS: 0, flashF: 0, cross: [] };

      VIZ.slider(ui.controls, { label: "Stimulus <i>Ω</i>", min: 0.5, max: 1.5, step: 0.005, value: S.W,
        fmt: (v) => v.toFixed(3), onInput: (v) => { S.W = v; S.cross = []; draw(); } });
      VIZ.slider(ui.controls, { label: "Coupling <i>A</i>", min: 0, max: 0.5, step: 0.005, value: S.A,
        fmt: (v) => v.toFixed(3), onInput: (v) => { S.A = v; S.cross = []; draw(); } });
      VIZ.button(VIZ.group(ui.controls), "Restart from φ = 0", restart, { primary: true });

      const circle = ui.panel({ title: "Stimulus Θ and firefly θ on the circle", aspect: 1, minHeight: 240,
        label: "Two phases moving around a circle; each flashes when it passes the top", onDraw: drawCircle });
      const flow = ui.panel({ title: "Flow on the circle  φ′ = Ω − ω − A sin φ", aspect: 1, minHeight: 240,
        label: "Rate of change of the phase difference with its fixed points", onDraw: drawFlow });
      const trace = ui.panel({ title: "Phase difference φ(t) = Θ − θ (not wrapped)", cls: "viz__panel--wide", aspect: 3, minHeight: 200,
        label: "Phase difference against time", onDraw: drawTrace });

      const rhs = (t, y) => [OMEGA + S.A * Math.sin(S.W * t - y[0])];
      function step() {
        const before = { s: Math.floor(S.Th / TWO_PI), f: Math.floor(S.th / TWO_PI), phi: S.Th - S.th };
        S.th = VIZ.rk4(rhs, S.t, [S.th], DT)[0];
        S.t += DT; S.Th = S.W * S.t;
        if (Math.floor(S.Th / TWO_PI) > before.s) S.flashS = 1;
        if (Math.floor(S.th / TWO_PI) > before.f) S.flashF = 1;
        const phi = S.Th - S.th;
        // time at which φ passes a multiple of 2π (to measure the drift period)
        if (Math.floor(phi / TWO_PI) !== Math.floor(before.phi / TWO_PI)) { S.cross.push(S.t); if (S.cross.length > 3) S.cross.shift(); }
        S.hist.push([S.t, phi]);
        while (S.hist.length && S.hist[0][0] < S.t - WINDOW) S.hist.shift();
      }
      function restart() { S.t = 0; S.Th = 0; S.th = 0; S.hist = [[0, 0]]; S.cross = []; draw(); loop.start(); }

      const loop = VIZ.loop(ui.el, (dt) => {
        const n = Math.max(1, Math.round((SPEED * dt) / DT));
        for (let i = 0; i < n; i++) step();
        S.flashS *= 0.85; S.flashF *= 0.85;
        draw();
      });

      function draw() { circle.draw(); flow.draw(); trace.draw(); readout(); }

      function drawCircle(c) {
        const T = VIZ.theme(), ctx = c.ctx, cx = c.w / 2, cy = c.h / 2, R = Math.min(c.w, c.h) * 0.36;
        ctx.fillStyle = T.panel; ctx.fillRect(0, 0, c.w, c.h);
        ctx.strokeStyle = T.axis; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, TWO_PI); ctx.stroke();
        // flash marker at phase 0 (top of the circle)
        ctx.beginPath(); ctx.moveTo(cx, cy - R - 8); ctx.lineTo(cx, cy - R + 8); ctx.stroke();
        const pos = (a) => [cx + R * Math.sin(a), cy - R * Math.cos(a)];   // phase 0 at the top, clockwise
        // arc for φ
        ctx.strokeStyle = T.accent; ctx.lineWidth = 6; ctx.globalAlpha = 0.45;
        const a0 = (S.th % TWO_PI) - Math.PI / 2, a1 = a0 + ((((S.Th - S.th) % TWO_PI) + TWO_PI) % TWO_PI);
        ctx.beginPath(); ctx.arc(cx, cy, R, a0, a1); ctx.stroke(); ctx.globalAlpha = 1;
        [[S.Th, T.c[0], S.flashS, "stimulus Θ"], [S.th, T.c[1], S.flashF, "firefly θ"]].forEach(([a, col, fl, lab], k) => {
          const [x, y] = pos(a);
          if (fl > 0.05) { ctx.fillStyle = T.c[3]; ctx.globalAlpha = fl * 0.8; ctx.beginPath(); ctx.arc(x, y, 9 + 14 * fl, 0, TWO_PI); ctx.fill(); ctx.globalAlpha = 1; }
          ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 9, 0, TWO_PI); ctx.fill();
          ctx.font = T.font(11, 700); ctx.fillStyle = col; ctx.textAlign = "left";
          ctx.fillText("● " + lab, 10, c.h - 26 + 16 * k);
        });
        ctx.fillStyle = T.text; ctx.font = T.font(11, 600); ctx.textAlign = "right";
        ctx.fillText("flash at the top", c.w - 10, 16);
      }

      function drawFlow(c) {
        const d = S.W - OMEGA, top = Math.max(0.3, Math.abs(d) + S.A) * 1.15;
        const ax = VIZ.axes(c, { xlim: [0, TWO_PI], ylim: [-top, top], xticks: [0, Math.PI / 2, Math.PI, 1.5 * Math.PI, TWO_PI],
          fmtX: fmtPi, xlabel: "φ (mod 2π)", ylabel: "φ′" });
        const T = ax.T, ctx = c.ctx;
        ax.clip();
        ax.line([0, TWO_PI], [0, 0], { color: T.axis, width: 1 });
        const xs = VIZ.linspace(0, TWO_PI, 300);
        ax.line(xs, xs.map((p) => d - S.A * Math.sin(p)), { color: T.c[0], width: 2.5 });
        // fixed points: sin φ* = μ
        if (S.A > 0 && Math.abs(d) <= S.A) {
          const p1 = Math.asin(d / S.A), p2 = Math.PI - p1;
          [p1, p2].forEach((p) => {
            const pp = ((p % TWO_PI) + TWO_PI) % TWO_PI;
            ax.fixed(pp, 0, Math.cos(pp) > 1e-9);      // stable where −A cos φ < 0
          });
        }
        const cur = (((S.Th - S.th) % TWO_PI) + TWO_PI) % TWO_PI;
        ax.dot(cur, d - S.A * Math.sin(cur), 6, T.accent, T.stable);
        ctx.restore();
      }

      function drawTrace(c) {
        const ts = S.hist.map((p) => p[0]), ps = S.hist.map((p) => p[1]);
        let lo = Math.min(-0.5, ...ps), hi = Math.max(Math.PI, ...ps);
        const t1 = Math.max(WINDOW, S.t);
        const ax = VIZ.axes(c, { xlim: [t1 - WINDOW, t1], ylim: [lo, hi + 0.1 * (hi - lo)], xlabel: "t", ylabel: "φ (rad)" });
        const T = ax.T;
        ax.clip();
        const mu = S.A > 0 ? (S.W - OMEGA) / S.A : Infinity;
        if (Math.abs(mu) <= 1) {
          const target = Math.asin(mu);
          ax.line([t1 - WINDOW, t1], [target, target], { color: T.axis, width: 1.5, dash: [5, 4] });
          ax.label("arcsin μ", t1 - WINDOW, target, { dx: 6, dy: -6 });
        }
        ax.line(ts, ps, { color: T.c[0], width: 2.2 });
        c.ctx.restore();
      }

      function readout() {
        const d = S.W - OMEGA, mu = S.A > 0 ? d / S.A : Infinity;
        let txt = `ω = 1, Ω = ${S.W.toFixed(3)}, A = ${S.A.toFixed(3)} · μ = (Ω − ω)/A = ${isFinite(mu) ? mu.toFixed(2) : "∞"} · `;
        if (Math.abs(mu) <= 1) {
          txt += `Locked: φ → arcsin μ = ${Math.asin(mu).toFixed(3)} rad; the firefly flashes at the stimulus frequency.`;
        } else {
          const T = TWO_PI / Math.sqrt(d * d - S.A * S.A);
          const n = S.cross.length;
          const meas = n >= 2 ? (S.cross[n - 1] - S.cross[n - 2]).toFixed(1) : "…";
          txt += `Drifting: T_drift = 2π/√((ω − Ω)² − A²) = ${T.toFixed(1)}; measured ${meas}.`;
        }
        ui.readout.textContent = txt;
      }

      S.hist = [[0, 0]];
      if (VIZ.reduceMotion) { for (let i = 0; i < 2400; i++) step(); draw(); }
      else loop.start();
    },
  });
})();

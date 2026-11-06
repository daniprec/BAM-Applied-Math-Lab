/* Couzin et al. (2002) zones model in 2D, ported from amlab.collective_motion.couzin.couzin_equations:
   repulsion (|r| < r_r) has priority; otherwise orientation (r_r ≤ |r| < r_o) and attraction
   (r_o ≤ |r| < r_a) are summed. Field of view α = 270°, Gaussian angular noise σ = 0.05,
   maximum turning rate 40°/s, speed s = 3, τ = 0.1, r_r = 1; unbounded plane.
   Readouts: polarization p_group and angular momentum m_group (eq-couzin-order). */
(function () {
  "use strict";
  const TAU = 0.1, V0 = 3, RR = 1, SIGMA = 0.05, HALF_FOV = (270 * Math.PI) / 360, MAX_TURN = ((40 * Math.PI) / 180) * TAU;
  const TWO_PI = 2 * Math.PI, HIST = 1500;
  const wrap = (a) => ((((a + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI) - Math.PI;

  VIZ.register("couzin", {
    title: "Couzin model: swarm, torus or parallel group",
    init(ui) {
      const S = { N: 50, dro: 5, dra: 8, x: [], y: [], th: [], n: 0, hist: [], seed: 1, view: 40, p: 0, m: 0 };
      let u = VIZ.rng(1);

      const sO = VIZ.slider(ui.controls, { label: "Δ<i>r</i><sub>o</sub> (orientation)", min: 0, max: 15, step: 0.5, value: S.dro,
        fmt: (v) => v.toFixed(1), onInput: (v) => { S.dro = v; } });
      const sA = VIZ.slider(ui.controls, { label: "Δ<i>r</i><sub>a</sub> (attraction)", min: 0, max: 15, step: 0.5, value: S.dra,
        fmt: (v) => v.toFixed(1), onInput: (v) => { S.dra = v; } });
      VIZ.select(ui.controls, { label: "N", value: "50", options: [["50", "50"], ["100", "100"]], onChange: (v) => { S.N = +v; reset(); } });
      const g = VIZ.group(ui.controls);
      const preset = (o, a) => () => { S.dro = o; S.dra = a; sO.set(o); sA.set(a); };
      VIZ.button(g, "Swarm (Δr_o = 0)", preset(0, 12));
      VIZ.button(g, "Torus (Δr_o = 3)", preset(3, 12));
      VIZ.button(g, "Parallel (Δr_o = 12)", preset(12, 12));
      VIZ.button(g, "Restart from random", () => { S.seed++; reset(); }, { primary: true });

      const plane = ui.panel({ title: "Group, centred on its centroid · zones drawn around one individual", aspect: 1, minHeight: 260,
        label: "Individuals drawn as arrows with the three zones of one individual", onDraw: drawPlane });
      const trace = ui.panel({ title: "Polarization p and angular momentum m", aspect: 1, minHeight: 260,
        label: "Recent history of polarization and angular momentum", onDraw: drawTrace });

      function reset() {
        u = VIZ.rng(S.seed * 7 + S.N);
        S.th = Array.from({ length: S.N }, () => TWO_PI * u());
        S.x = []; S.y = [];
        for (let i = 0; i < S.N; i++) { S.x.push(25 * u()); S.y.push(25 * u()); }   // box_size = 25, as in the lab
        S.n = 0; S.hist = []; measure(); draw();
      }

      function step() {
        const N = S.N, x = S.x, y = S.y, th = S.th, ro = RR + S.dro, ra = ro + S.dra;
        const nt = new Array(N);
        for (let i = 0; i < N; i++) {
          const hi = th[i];
          let repX = 0, repY = 0, nRep = 0, oX = 0, oY = 0, nO = 0, aX = 0, aY = 0, nA = 0, seen = 0;
          for (let j = 0; j < N; j++) {
            if (j === i) continue;
            const dx = x[j] - x[i], dy = y[j] - y[i], d = Math.hypot(dx, dy);
            if (d > ra) continue;
            if (Math.abs(wrap(Math.atan2(dy, dx) - hi)) > HALF_FOV) continue;   // blind sector
            seen++;
            const ux = d > 0 ? dx / d : 0, uy = d > 0 ? dy / d : 0;
            if (d < RR) { repX -= ux; repY -= uy; nRep++; }
            else if (d < ro) { oX += Math.cos(th[j]); oY += Math.sin(th[j]); nO++; }
            else if (d < ra) { aX += ux; aY += uy; nA++; }
          }
          let dxv, dyv;
          if (!seen) { dxv = Math.cos(hi); dyv = Math.sin(hi); }
          else if (nRep) { dxv = repX; dyv = repY; }
          else if (nO || nA) { dxv = oX + aX; dyv = oY + aY; }
          else { dxv = Math.cos(hi); dyv = Math.sin(hi); }
          if (Math.abs(dxv) < 1e-8 && Math.abs(dyv) < 1e-8) { dxv = Math.cos(hi); dyv = Math.sin(hi); }
          let desired = Math.atan2(dyv, dxv) + SIGMA * u.normal();
          let dh = wrap(desired - hi);
          if (Math.abs(dh) > MAX_TURN) dh = Math.sign(dh) * MAX_TURN;
          nt[i] = (((hi + dh) % TWO_PI) + TWO_PI) % TWO_PI;
        }
        for (let i = 0; i < N; i++) {
          th[i] = nt[i];
          x[i] += TAU * V0 * Math.cos(nt[i]);
          y[i] += TAU * V0 * Math.sin(nt[i]);
        }
        S.n++;
        measure();
        S.hist.push([S.n, S.p, S.m]);
        if (S.hist.length > HIST) S.hist.shift();
      }
      function measure() {
        const N = S.N;
        let cx = 0, cy = 0, vx = 0, vy = 0;
        for (let i = 0; i < N; i++) { cx += S.x[i]; cy += S.y[i]; vx += Math.cos(S.th[i]); vy += Math.sin(S.th[i]); }
        cx /= N; cy /= N;
        let mz = 0;
        for (let i = 0; i < N; i++) {
          const rx = S.x[i] - cx, ry = S.y[i] - cy, d = Math.hypot(rx, ry) || 1;
          mz += (rx / d) * Math.sin(S.th[i]) - (ry / d) * Math.cos(S.th[i]);
        }
        S.p = Math.hypot(vx, vy) / N; S.m = Math.abs(mz) / N; S.cx = cx; S.cy = cy;
      }

      const loop = VIZ.loop(ui.el, () => {
        const per = S.N > 60 ? 2 : 3;
        for (let k = 0; k < per; k++) step();
        draw();
      });

      function draw() { plane.draw(); trace.draw(); readout(); }

      function drawPlane(c) {
        const T = VIZ.theme(), ctx = c.ctx;
        ctx.fillStyle = T.panel; ctx.fillRect(0, 0, c.w, c.h);
        // zoom to the group's extent (smoothed)
        let ext = 0;
        for (let i = 0; i < S.N; i++) ext = Math.max(ext, Math.hypot(S.x[i] - S.cx, S.y[i] - S.cy));
        const target = Math.min(120, Math.max(20, 2.3 * ext + 6));
        S.view += (target - S.view) * 0.05;
        const sc = Math.min(c.w, c.h) / S.view, X = (px) => c.w / 2 + sc * (px - S.cx), Y = (py) => c.h / 2 - sc * (py - S.cy);
        // zones of individual 0, with its blind sector
        const ro = RR + S.dro, ra = ro + S.dra, h0 = S.th[0];
        const zones = [[ra, T.c[2], "attraction"], [ro, T.c[0], "orientation"], [RR, T.c[1], "repulsion"]];
        zones.forEach(([r, col]) => {
          ctx.beginPath(); ctx.moveTo(X(S.x[0]), Y(S.y[0]));
          ctx.arc(X(S.x[0]), Y(S.y[0]), sc * r, -h0 - HALF_FOV, -h0 + HALF_FOV);
          ctx.closePath(); ctx.fillStyle = col; ctx.globalAlpha = T.dark ? 0.26 : 0.14; ctx.fill(); ctx.globalAlpha = 1;
        });
        const len = Math.max(6, Math.min(11, sc * 1.2));
        for (let i = 0; i < S.N; i++) {
          const px = X(S.x[i]), py = Y(S.y[i]), a = S.th[i], cs = Math.cos(a), sn = -Math.sin(a);
          ctx.fillStyle = i === 0 ? T.accent : T.stable;
          ctx.beginPath();
          ctx.moveTo(px + len * cs, py + len * sn);
          ctx.lineTo(px - 0.5 * len * cs - 0.45 * len * sn, py - 0.5 * len * sn + 0.45 * len * cs);
          ctx.lineTo(px - 0.5 * len * cs + 0.45 * len * sn, py - 0.5 * len * sn - 0.45 * len * cs);
          ctx.closePath(); ctx.fill();
        }
        ctx.font = T.font(11, 600); ctx.textAlign = "left";
        zones.slice().reverse().forEach(([, col, lab], k) => {
          ctx.fillStyle = col; ctx.globalAlpha = 0.6; ctx.fillRect(8, c.h - 50 + 15 * k, 10, 10); ctx.globalAlpha = 1;
          ctx.fillStyle = T.text; ctx.fillText(lab, 24, c.h - 41 + 15 * k);
        });
      }
      function drawTrace(c) {
        const n1 = Math.max(HIST, S.n);
        const ax = VIZ.axes(c, { xlim: [n1 - HIST, n1], ylim: [0, 1], xlabel: "time step (τ = 0.1 s)", ylabel: "" });
        const T = ax.T;
        ax.clip();
        ax.line(S.hist.map((h) => h[0]), S.hist.map((h) => h[1]), { color: T.c[0], width: 2 });
        ax.line(S.hist.map((h) => h[0]), S.hist.map((h) => h[2]), { color: T.c[1], width: 2, dash: [6, 3] });
        c.ctx.restore();
        VIZ.legend(c, [{ label: "polarization p", color: T.c[0] }, { label: "angular momentum m", color: T.c[1], dash: [6, 3] }],
          ax.pad.l + 10, ax.pad.t + 14);
      }
      function readout() {
        const last = S.hist.slice(-300), k = Math.max(1, last.length);
        const p = last.reduce((a, h) => a + h[1], 0) / k, m = last.reduce((a, h) => a + h[2], 0) / k;
        let state;
        if (S.n < 300) state = "settling…";
        else if (p >= 0.95) state = "highly parallel group";
        else if (p >= 0.5) state = "dynamic parallel group";
        else if (m >= 0.45) state = "torus (milling)";
        else if (m < 0.35 && p < 0.3) state = "swarm";
        else state = "in between";
        ui.readout.textContent = `r_r = 1, r_o = ${(RR + S.dro).toFixed(1)}, r_a = ${(RR + S.dro + S.dra).toFixed(1)} · ` +
          `mean over the last 300 steps: p = ${p.toFixed(2)}, m = ${m.toFixed(2)} → ${state}`;
      }

      reset();
      if (VIZ.reduceMotion) { for (let i = 0; i < 1500; i++) step(); draw(); }
      else loop.start();
    },
  });
})();

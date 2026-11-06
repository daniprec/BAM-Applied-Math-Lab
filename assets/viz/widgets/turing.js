/* Turing instability in the Gierer-Meinhardt model (course scaling, Murray II §2.3):
     u_t = u_xx + γ (a − b u + u²/v),     v_t = d v_xx + γ (u² − v),   b = γ = 1
   Panel 1: dispersion relation Re σ(k) with the unstable band and the Neumann modes k_n = nπ/L.
   Panel 2–3: explicit Euler simulation from 1% noise (as amlab.turing.simulate_gm_1d,
   Neumann boundaries by copying the nearest interior value), profile and space-time diagram. */
(function () {
  "use strict";
  const B = 1, GAMMA = 1, DX = 0.5, KMAX = 1.5, SPEED = 60, ROW_DT = 2, ROWS = 260;

  function jac(a) {                       // amlab.turing.gierer_meinhardt_jacobian
    const us = (a + 1) / B, vs = us * us;
    return { fu: -B + (2 * us) / vs, fv: -(us * us) / (vs * vs), gu: 2 * us, gv: -1, us, vs };
  }
  function sigma(k2, J, d) {              // amlab.turing.dispersion_relation (larger real part)
    const tr = GAMMA * (J.fu + J.gv) - k2 * (1 + d);
    const h = d * k2 * k2 - GAMMA * (d * J.fu + J.gv) * k2 + GAMMA * GAMMA * (J.fu * J.gv - J.fv * J.gu);
    const disc = tr * tr - 4 * h;
    return disc >= 0 ? 0.5 * (tr + Math.sqrt(disc)) : 0.5 * tr;
  }
  function dCrit(J) {                     // amlab.turing.critical_diffusion_ratio
    const qa = J.fu * J.fu, qb = 2 * (2 * J.fv * J.gu - J.fu * J.gv), qc = J.gv * J.gv;
    const disc = qb * qb - 4 * qa * qc;
    if (disc < 0 || qa === 0) return NaN;
    const root = (-qb + Math.sqrt(disc)) / (2 * qa);
    return root * J.fu + J.gv > 0 ? root : NaN;
  }
  function band(J, d) {                   // amlab.turing.unstable_band, returned as k (not k²)
    const det = J.fu * J.gv - J.fv * J.gu, s = d * J.fu + J.gv, disc = s * s - 4 * d * det;
    if (s <= 0 || disc <= 0) return null;
    return [Math.sqrt((GAMMA * (s - Math.sqrt(disc))) / (2 * d)), Math.sqrt((GAMMA * (s + Math.sqrt(disc))) / (2 * d))];
  }

  VIZ.register("turing", {
    title: "Turing instability: which wavelengths grow?",
    init(ui) {
      const S = { a: 0.4, d: 30, L: 40, seed: 3, t: 0, rows: [], paused: false };

      VIZ.slider(ui.controls, { label: "<i>d</i> = D<sub>v</sub>/D<sub>u</sub>", min: 5, max: 80, step: 0.5, value: S.d,
        fmt: (v) => v.toFixed(1), onInput: (v) => { S.d = v; draw(); loop.start(); } });
      VIZ.slider(ui.controls, { label: "<i>a</i>", min: 0.05, max: 0.7, step: 0.01, value: S.a,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.a = v; reset(); } });
      VIZ.slider(ui.controls, { label: "<i>L</i>", min: 5, max: 100, step: 1, value: S.L,
        fmt: (v) => v.toFixed(0), onInput: (v) => { S.L = v; reset(); } });
      const g = VIZ.group(ui.controls);
      VIZ.button(g, "Restart from noise", () => { S.seed++; reset(); }, { primary: true });
      const pauseBtn = VIZ.button(g, "Pause", () => {
        S.paused = !S.paused; pauseBtn.textContent = S.paused ? "Resume" : "Pause";
        pauseBtn.setAttribute("aria-pressed", String(S.paused));
        if (!S.paused) loop.start();
      });

      const disp = ui.panel({ title: "Dispersion relation  Re σ(k)  ·  dots: Neumann modes k = nπ/L", aspect: 1.35, minHeight: 240,
        label: "Growth rate against wavenumber with the unstable band shaded and the admissible modes marked", onDraw: drawDisp });
      const prof = ui.panel({ title: "Inhibitor profile  v(x, t)", aspect: 1.35, minHeight: 240,
        label: "Current inhibitor concentration along the domain", onDraw: drawProfile });
      const kymo = ui.panel({ title: "Space-time diagram of v (time runs downward; blue above, red below the uniform state)",
        cls: "viz__panel--wide", aspect: 3.2, minHeight: 170, label: "Space-time diagram of the inhibitor", onDraw: drawKymo });

      const loop = VIZ.loop(ui.el, (dtSec) => {
        if (S.paused) { loop.stop(); return; }
        advance(SPEED * dtSec);
        draw();
      });

      function dt() { return Math.min(0.01, (0.4 * DX * DX) / Math.max(S.d, 1)); }
      function reset() {
        const J = jac(S.a), n = Math.max(4, Math.round(S.L / DX)), u = VIZ.rng(S.seed * 9973 + 1);
        S.N = n; S.u = new Float64Array(n); S.v = new Float64Array(n);
        S.lu = new Float64Array(n); S.lv = new Float64Array(n);
        for (let i = 0; i < n; i++) {           // steady state × (1 + 1% Gaussian noise)
          S.u[i] = J.us * (1 + 0.01 * u.normal());
          S.v[i] = J.vs * (1 + 0.01 * u.normal());
        }
        S.t = 0; S.nextRow = 0; S.rows = [];
        if (VIZ.reduceMotion) advance(600);
        draw();
        if (!VIZ.reduceMotion && !S.paused) loop.start();
      }
      function advance(T) {
        const h = dt(), steps = Math.min(40000, Math.ceil(T / h)), n = S.N, u = S.u, v = S.v, lu = S.lu, lv = S.lv;
        const a = S.a, d = S.d, idx2 = 1 / (DX * DX);
        for (let s = 0; s < steps; s++) {
          for (let i = 0; i < n; i++) {        // 3-point Laplacian with periodic wrap, as np.roll
            const l = i === 0 ? n - 1 : i - 1, r = i === n - 1 ? 0 : i + 1;
            lu[i] = (u[l] + u[r] - 2 * u[i]) * idx2;
            lv[i] = (v[l] + v[r] - 2 * v[i]) * idx2;
          }
          for (let i = 0; i < n; i++) {
            const ui = u[i], vi = v[i];
            u[i] = ui + h * (lu[i] + GAMMA * (a - B * ui + (ui * ui) / vi));
            v[i] = vi + h * (d * lv[i] + GAMMA * (ui * ui - vi));
          }
          u[0] = u[1]; u[n - 1] = u[n - 2];    // Neumann: copy the nearest interior value
          v[0] = v[1]; v[n - 1] = v[n - 2];
          S.t += h;
          if (S.t >= S.nextRow) {
            S.rows.push(Float64Array.from(v));
            if (S.rows.length > ROWS) S.rows.shift();
            S.nextRow += ROW_DT;
          }
        }
        if (!isFinite(u[1])) { S.paused = true; }
      }
      function draw() { disp.draw(); prof.draw(); kymo.draw(); readout(); }

      function drawDisp(c) {
        const J = jac(S.a), ks = VIZ.linspace(0, KMAX, 400), sg = ks.map((k) => sigma(k * k, J, S.d));
        const top = Math.max(0.04, 1.25 * Math.max(...sg));
        const ax = VIZ.axes(c, { xlim: [0, KMAX], ylim: [-0.15, top], xlabel: "wavenumber k", ylabel: "Re σ" });
        const T = ax.T, ctx = c.ctx;
        ax.clip();
        const bd = band(J, S.d);
        if (bd && dCrit(J) <= S.d) {
          ctx.save(); ctx.fillStyle = T.accent; ctx.globalAlpha = 0.22;
          ctx.fillRect(ax.sx(bd[0]), ax.pad.t, ax.sx(bd[1]) - ax.sx(bd[0]), ax.ph); ctx.restore();
        }
        ax.line([0, KMAX], [0, 0], { color: T.axis, width: 1 });
        ax.line(ks, sg, { color: T.c[0], width: 2.5 });
        let best = null;
        for (let n = 0; (n * Math.PI) / S.L <= KMAX; n++) {
          const k = (n * Math.PI) / S.L, s = sigma(k * k, J, S.d);
          if (s > 0 && (!best || s > best.s)) best = { n, k, s };
          if (s > -0.15) ax.dot(k, s, 4, s > 0 ? T.c[1] : T.panel, s > 0 ? null : T.axis);
        }
        if (best) ax.label(`n = ${best.n}`, best.k, best.s, { dy: -9, align: "center", color: T.c[1] });
        ctx.restore();
        S.best = best; S.band = bd;
      }
      function drawProfile(c) {
        const J = jac(S.a), xs = Array.from({ length: S.N }, (_, i) => i * DX);
        const ax = VIZ.axes(c, { xlim: [0, S.L], ylim: [0.7 * J.vs, 1.3 * J.vs], xlabel: "x", ylabel: "v" });
        ax.clip();
        ax.line([0, S.L], [J.vs, J.vs], { color: ax.T.axis, width: 1, dash: [4, 4] });
        ax.line(xs, Array.from(S.v), { color: ax.T.c[0], width: 2.5 });
        c.ctx.restore();
        ax.label(`t = ${S.t.toFixed(0)}`, S.L, 1.3 * J.vs, { align: "right", dx: -6, dy: 14 });
      }
      function drawKymo(c) {
        const T = VIZ.theme(), ctx = c.ctx, J = jac(S.a);
        const pad = { l: 46, r: 12, t: 6, b: 22 }, pw = c.w - pad.l - pad.r, ph = c.h - pad.t - pad.b;
        ctx.fillStyle = T.panel; ctx.fillRect(pad.l, pad.t, pw, ph);
        const rows = S.rows, n = S.N;
        if (rows.length) {
          const img = ctx.createImageData(n, ROWS), hi = rgb(T.c[0]), lo = rgb(T.c[1]), mid = rgb(T.panel);
          for (let r = 0; r < ROWS; r++) {
            const row = rows[r];
            for (let i = 0; i < n; i++) {
              const o = 4 * (r * n + i);
              let col = mid;
              if (row) {
                const z = Math.max(-1, Math.min(1, (row[i] / J.vs - 1) / 0.25));
                col = mix(mid, z > 0 ? hi : lo, Math.abs(z));
              }
              img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
            }
          }
          const off = document.createElement("canvas"); off.width = n; off.height = ROWS;
          off.getContext("2d").putImageData(img, 0, 0);
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(off, pad.l, pad.t, pw, ph);
        }
        ctx.strokeStyle = T.axis; ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, pw - 1, ph - 1);
        ctx.font = T.font(11); ctx.fillStyle = T.text; ctx.textAlign = "center";
        ctx.fillText("x = 0", pad.l + 16, c.h - 6); ctx.fillText(`x = ${S.L}`, pad.l + pw - 20, c.h - 6);
        ctx.textAlign = "right";
        ctx.fillText("time", pad.l - 6, pad.t + 12); ctx.fillText("↓", pad.l - 6, pad.t + 26);
      }
      function readout() {
        const J = jac(S.a), dc = dCrit(J);
        let amp = 0;
        if (S.v) { let lo = Infinity, hi = -Infinity; for (let i = 0; i < S.N; i++) { lo = Math.min(lo, S.v[i]); hi = Math.max(hi, S.v[i]); } amp = hi - lo; }
        let txt = isFinite(dc) ? `d_c = ${dc.toFixed(2)} for a = ${S.a.toFixed(2)}. ` : `No Turing instability for a = ${S.a.toFixed(2)}. `;
        if (isFinite(dc) && S.d > dc && S.band) {
          txt += `Unstable band ${S.band[0].toFixed(3)} < k < ${S.band[1].toFixed(3)}; `;
          txt += S.best ? `fastest Neumann mode n = ${S.best.n} (σ = ${S.best.s.toFixed(3)}). ` : "no Neumann mode falls inside it, so the uniform state survives. ";
        } else if (isFinite(dc)) txt += "d is below d_c: every mode decays. ";
        txt += `Pattern amplitude max v − min v = ${amp.toFixed(3)} at t = ${S.t.toFixed(0)}.`;
        ui.readout.textContent = txt;
      }

      reset();
    },
  });

  function rgb(css) {
    const m = css.trim().match(/^#([0-9a-f]{6})$/i);
    if (m) return [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
    const n = css.match(/\d+(\.\d+)?/g) || [0, 0, 0];
    return n.slice(0, 3).map(Number);
  }
  function mix(a, b, t) { return [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)); }
})();

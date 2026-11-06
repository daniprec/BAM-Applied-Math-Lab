/* SIR and SIS epidemics (fractions of the population, time in days).
   SIR:  S' = −βSI,        I' = βSI − γI,  R' = γI
   SIS:  I' = βI(1 − I) − γI
   Defaults match the lab (sir-simulation.qmd): β = 0.3, γ = 0.1, I(0) = 0.01, 160 days.
   A fraction p is immune (vaccinated) at t = 0: S(0) = 1 − p − I(0), R(0) = p. */
(function () {
  "use strict";
  const DT = 0.05;

  function simulate(m, beta, gamma, i0, p, tmax) {
    const n = Math.round(tmax / DT);
    const t = new Float64Array(n + 1), S = new Float64Array(n + 1), I = new Float64Array(n + 1), R = new Float64Array(n + 1);
    let y = m === "sir" ? [Math.max(0, 1 - p - i0), i0, p] : [1 - i0, i0, 0];
    const f = m === "sir"
      ? (tt, u) => [-beta * u[0] * u[1], beta * u[0] * u[1] - gamma * u[1], gamma * u[1]]
      : (tt, u) => [-beta * u[0] * u[1] + gamma * u[1], beta * u[0] * u[1] - gamma * u[1], 0];
    for (let k = 0; k <= n; k++) {
      t[k] = k * DT; S[k] = y[0]; I[k] = y[1]; R[k] = y[2];
      y = VIZ.rk4(f, 0, y, DT);
    }
    return { t, S, I, R };
  }

  VIZ.register("sir", {
    title: "Epidemics: SIR and SIS",
    init(ui, el) {
      const S = { m: el.dataset.model || "sir", beta: 0.3, gamma: 0.1, lgI0: -2, p: 0, sim: null, tmax: 160 };

      VIZ.select(ui.controls, { label: "Model", value: S.m, options: [["sir", "SIR (immunity)"], ["sis", "SIS (no immunity)"]],
        onChange: (v) => { S.m = v; sP.disable(v !== "sir"); update(); } });
      VIZ.slider(ui.controls, { label: "<i>β</i>", min: 0.02, max: 1, step: 0.01, value: S.beta,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.beta = v; update(); } });
      VIZ.slider(ui.controls, { label: "<i>γ</i>", min: 0.02, max: 0.5, step: 0.01, value: S.gamma,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.gamma = v; update(); } });
      VIZ.slider(ui.controls, { label: "<i>I</i>(0)", min: -6, max: -1, step: 0.1, value: S.lgI0,
        fmt: (v) => (Math.pow(10, v) >= 0.001 ? Math.pow(10, v).toFixed(3) : "1e" + v.toFixed(1)), onInput: (v) => { S.lgI0 = v; update(); } });
      const sP = VIZ.slider(ui.controls, { label: "vaccinated <i>p</i>", min: 0, max: 0.95, step: 0.01, value: S.p,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.p = v; update(); } });

      const ts = ui.panel({ title: "Fractions of the population over time", aspect: 1.35, minHeight: 240,
        label: "Time series of susceptible, infectious and recovered fractions", onDraw: drawTime });
      const ph = ui.panel({ title: "Phase view", aspect: 1.35, minHeight: 240,
        label: "Phase view of the epidemic", onDraw: drawPhase });

      function update() {
        const i0 = Math.pow(10, S.lgI0);
        // choose a window: the lab's 160 days, extended if the epidemic is slow
        const base = S.m === "sir" ? 160 : 200;          // the windows used in the labs
        let sim = simulate(S.m, S.beta, S.gamma, i0, S.p, base);
        const tpk = sim.t[argmax(sim.I)];
        const slowSIR = S.m === "sir" && tpk > 0.7 * base;
        const slowSIS = S.m === "sis" && Math.abs(sim.I[sim.I.length - 1] - sim.I[sim.I.length - 200]) > 1e-3;
        S.tmax = base;
        if (slowSIR || slowSIS) {
          S.tmax = Math.min(800, Math.ceil((slowSIR ? 2.2 * tpk : 3 * base) / 40) * 40);
          sim = simulate(S.m, S.beta, S.gamma, i0, S.p, S.tmax);
        }
        S.sim = sim;
        ph.fig.querySelector(".viz__panel-title").textContent =
          S.m === "sir" ? "Phase plane (S, I): the peak happens at S = 1/R₀" : "Rate plot dI/dt against I";
        ts.draw(); ph.draw(); readout();
      }
      function argmax(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; }
      function every(a, step) { const out = []; for (let i = 0; i < a.length; i += step) out.push(a[i]); return out; }

      function drawTime(c) {
        const sim = S.sim, ax = VIZ.axes(c, { xlim: [0, S.tmax], ylim: [0, 1], xlabel: "t (days)", ylabel: "fraction" });
        const T = ax.T, st = Math.max(1, Math.round(sim.t.length / 600)), t = every(sim.t, st);
        ax.clip();
        ax.line(t, every(sim.S, st), { color: T.c[0], width: 2.2 });
        ax.line(t, every(sim.I, st), { color: T.c[1], width: 2.6 });
        if (S.m === "sir") ax.line(t, every(sim.R, st), { color: T.c[2], width: 2.2, dash: [6, 3] });
        const k = argmax(sim.I);
        if (S.m === "sir" && k > 0 && k < sim.I.length - 1) ax.dot(sim.t[k], sim.I[k], 5, T.c[1], T.panel);
        c.ctx.restore();
        VIZ.legend(c, [{ label: "S", color: T.c[0] }, { label: "I", color: T.c[1] }].concat(S.m === "sir" ? [{ label: "R", color: T.c[2], dash: [6, 3] }] : []),
          ax.pad.l + ax.pw - 60, ax.pad.t + ax.ph * 0.45);
      }

      function drawPhase(c) {
        const T = VIZ.theme(), R0 = S.beta / S.gamma;
        if (S.m === "sir") {
          const ax = VIZ.axes(c, { xlim: [0, 1], ylim: [0, Math.max(0.1, Math.min(1, 1.2 * Math.max(...S.sim.I)))], xlabel: "S", ylabel: "I" });
          ax.clip();
          if (1 / R0 <= 1) ax.line([1 / R0, 1 / R0], ax.ylim, { color: T.axis, width: 1.5, dash: [4, 4] });
          const st = Math.max(1, Math.round(S.sim.t.length / 600));
          ax.line(every(S.sim.S, st), every(S.sim.I, st), { color: T.c[1], width: 2.5 });
          ax.dot(S.sim.S[0], S.sim.I[0], 4, T.stable);
          c.ctx.restore();
          if (1 / R0 <= 1) ax.label("S = 1/R₀", 1 / R0, ax.ylim[1], { dx: 5, dy: 14 });
        } else {
          const Is = VIZ.linspace(0, 1, 300), f = (I) => S.beta * I * (1 - I) - S.gamma * I;
          const ys = Is.map(f), lo = Math.min(-0.05, ...ys), hi = Math.max(0.05, ...ys);
          const ax = VIZ.axes(c, { xlim: [0, 1], ylim: [lo * 1.15, hi * 1.15], xlabel: "I", ylabel: "dI/dt" });
          ax.clip();
          ax.line([0, 1], [0, 0], { color: T.axis, width: 1 });
          ax.line(Is, ys, { color: T.c[1], width: 2.5 });
          const Istar = 1 - 1 / R0;
          ax.fixed(0, 0, R0 < 1);
          if (Istar > 0) ax.fixed(Istar, 0, true);
          c.ctx.restore();
        }
      }

      function readout() {
        const R0 = S.beta / S.gamma, sim = S.sim, i0 = Math.pow(10, S.lgI0);
        if (S.m === "sir") {
          const Reff = R0 * sim.S[0], k = argmax(sim.I);
          const infected = sim.R[sim.R.length - 1] + sim.I[sim.I.length - 1] - S.p;
          const pc = Math.max(0, 1 - 1 / R0);
          let txt = `R₀ = ${R0.toFixed(2)}, R_eff(0) = R₀S(0) = ${Reff.toFixed(2)}. `;
          txt += Reff > 1 && k > 0
            ? `Peak I = ${sim.I[k].toFixed(3)} on day ${Math.round(sim.t[k])}. `
            : "No outbreak: I decreases from the start. ";
          txt += `Ever infected: ${(100 * (infected)).toFixed(1)}% of the population. Herd-immunity threshold p_c = 1 − 1/R₀ = ${pc.toFixed(2)}.`;
          ui.readout.textContent = txt;
        } else {
          const Istar = 1 - 1 / R0;
          ui.readout.textContent = `R₀ = ${R0.toFixed(2)}. ` + (R0 > 1
            ? `Endemic equilibrium I* = 1 − 1/R₀ = ${Istar.toFixed(3)}; I(${S.tmax}) = ${sim.I[sim.I.length - 1].toFixed(3)}. Perturbations decay like exp(−(β−γ)t), time scale ${(1 / (S.beta - S.gamma)).toFixed(0)} days.`
            : `R₀ ≤ 1: the infection dies out (I → 0).`);
        }
      }

      sP.disable(S.m !== "sir");
      update();
    },
  });
})();

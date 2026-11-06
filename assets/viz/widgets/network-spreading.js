/* Epidemics on networks: discrete-time SIR / SIS with vaccination.
   Update rule exactly as amlab.networks_spreading.run_sir_fast / run_sis_fast:
   synchronous steps; a susceptible node with m infected neighbours is infected with
   probability 1 - (1 - beta)^m; an infected node recovers with probability mu.
   Vaccinated nodes start in R (immune); 5 initial infected among the others.
   Networks as in the lab: Barabási–Albert (m = 3) or Erdős–Rényi with the same mean degree. */
(function () {
  "use strict";

  /* ---------- networks (NetworkX algorithms) ---------- */
  const empty = (n) => Array.from({ length: n }, () => new Set());
  const addEdge = (A, u, v) => { A[u].add(v); A[v].add(u); };
  function barabasiAlbert(n, m, u) {
    const A = empty(n);
    for (let i = 1; i <= m; i++) addEdge(A, 0, i);
    const rep = [];
    for (let i = 0; i <= m; i++) for (let d = 0; d < A[i].size; d++) rep.push(i);
    for (let s = m + 1; s < n; s++) {
      const t = new Set();
      while (t.size < m) t.add(rep[Math.floor(u() * rep.length)]);
      t.forEach((x) => { addEdge(A, s, x); rep.push(x, s); });
    }
    return A;
  }
  function erdosRenyi(n, p, u) {
    const A = empty(n);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (u() < p) addEdge(A, i, j);
    return A;
  }
  function force(A, u) {
    const n = A.length, P = Array.from({ length: n }, () => [u() * 2 - 1, u() * 2 - 1]);
    const k = Math.sqrt(4 / n), iters = n > 300 ? 90 : 200;
    let temp = 0.2;
    for (let it = 0; it < iters; it++) {
      const D = Array.from({ length: n }, () => [0, 0]);
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        let dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1];
        const f = (k * k) / Math.max(1e-6, dx * dx + dy * dy);
        dx *= f; dy *= f; D[i][0] += dx; D[i][1] += dy; D[j][0] -= dx; D[j][1] -= dy;
      }
      for (let i = 0; i < n; i++) A[i].forEach((j) => {
        if (j < i) return;
        const dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], f = Math.hypot(dx, dy) / k;
        D[i][0] -= dx * f; D[i][1] -= dy * f; D[j][0] += dx * f; D[j][1] += dy * f;
      });
      for (let i = 0; i < n; i++) {
        D[i][0] -= 0.02 * P[i][0]; D[i][1] -= 0.02 * P[i][1];
        const L = Math.hypot(D[i][0], D[i][1]) || 1, s = Math.min(L, temp) / L;
        P[i][0] += D[i][0] * s; P[i][1] += D[i][1] * s;
      }
      temp *= n > 300 ? 0.97 : 0.985;
    }
    return P;
  }

  /* ---------- immunization rules (amlab.networks_spreading) ---------- */
  function choose(A, g, strategy, u) {
    const n = A.length, k = Math.round(g * n);
    if (strategy === "targeted") {
      return new Set([...A.keys()].sort((a, b) => A[b].size - A[a].size).slice(0, k));
    }
    const chosen = new Set();
    while (chosen.size < k) {
      const v = Math.floor(u() * n);
      if (strategy === "random") chosen.add(v);
      else { const nb = [...A[v]]; if (nb.length) chosen.add(nb[Math.floor(u() * nb.length)]); }
    }
    return chosen;
  }

  /* ---------- dynamics: 0 = S, 1 = I, 2 = R ---------- */
  function init(A, immune, u) {
    const n = A.length, st = new Uint8Array(n);
    immune.forEach((v) => { st[v] = 2; });
    const cand = []; for (let i = 0; i < n; i++) if (!st[i]) cand.push(i);
    for (let c = 0; c < Math.min(5, cand.length); c++) {
      const j = c + Math.floor(u() * (cand.length - c));
      [cand[c], cand[j]] = [cand[j], cand[c]];
      st[cand[c]] = 1;
    }
    return st;
  }
  function step(A, st, beta, mu, sis, u) {
    const n = A.length, nxt = st.slice();
    for (let i = 0; i < n; i++) {
      if (st[i] === 0) {
        let m = 0; A[i].forEach((j) => { if (st[j] === 1) m++; });
        if (m && u() < 1 - Math.pow(1 - beta, m)) nxt[i] = 1;
      } else if (st[i] === 1 && u() < mu) nxt[i] = sis ? 0 : 2;
    }
    return nxt;
  }

  VIZ.register("network-spreading", {
    title: "Epidemics on networks: hubs, thresholds and vaccination",
    init(ui) {
      const S = { net: "ba", n: 200, model: "sir", beta: 0.05, mu: 0.1, strategy: "random", g: 0, seed: 7,
        A: null, pos: null, st: null, immune: null, t: 0, hist: [], ever: 0, done: false, acc: 0, sweep: null };

      const top = VIZ.group(ui.controls);
      VIZ.select(top, { label: "Network", value: "ba", options: [["ba", "Barabási–Albert, m = 3"], ["er", "Erdős–Rényi, <k> = 6"]],
        onChange: (v) => { S.net = v; S.sweep = null; build(); } });
      VIZ.select(top, { label: "N", value: "200", options: [["200", "200"], ["500", "500"]], onChange: (v) => { S.n = +v; S.sweep = null; build(); } });
      VIZ.select(top, { label: "Model", value: "sir", options: [["sir", "SIR"], ["sis", "SIS"]], onChange: (v) => { S.model = v; S.sweep = null; restart(); } });
      VIZ.slider(ui.controls, { label: "<i>β</i>", min: 0.005, max: 0.3, step: 0.005, value: S.beta, fmt: (v) => v.toFixed(3), onInput: (v) => { S.beta = v; restart(); } });
      VIZ.slider(ui.controls, { label: "<i>μ</i>", min: 0.02, max: 0.5, step: 0.01, value: S.mu, fmt: (v) => v.toFixed(2), onInput: (v) => { S.mu = v; restart(); } });
      const vac = VIZ.group(ui.controls);
      VIZ.select(vac, { label: "Vaccinate", value: "random", options: [["random", "at random"], ["acquaintance", "acquaintances"], ["targeted", "hubs first"]],
        onChange: (v) => { S.strategy = v; restart(); } });
      VIZ.slider(ui.controls, { label: "coverage <i>g</i>", min: 0, max: 0.4, step: 0.02, value: 0, fmt: (v) => v.toFixed(2), onInput: (v) => { S.g = v; restart(); } });
      const btns = VIZ.group(ui.controls);
      VIZ.button(btns, "Run again", () => { S.seed++; restart(); }, { primary: true });
      const sweepBtn = VIZ.button(btns, "Compare strategies over g", startSweep);

      // two columns on wide screens (the wide panel below must not reserve a third, empty column)
      ui.panels.style.gridTemplateColumns = "repeat(auto-fit, minmax(min(100%, 320px), 1fr))";
      const net = ui.panel({ title: "Network  (S grey · I red · R blue · vaccinated green ring)", aspect: 1.1, minHeight: 280,
        label: "Network with each node coloured by its state", onDraw: drawNet });
      const curve = ui.panel({ title: "Fractions S, I, R over time", aspect: 1.1, minHeight: 280,
        label: "Time series of the fractions of susceptible, infected and recovered nodes", onDraw: drawCurve });
      const sweepP = ui.panel({ title: "Final epidemic size (SIR) or long-run prevalence (SIS) against coverage g", cls: "viz__panel--wide",
        aspect: 2.8, minHeight: 200, label: "Outcome against vaccination coverage for the three strategies", onDraw: drawSweep });

      const loop = VIZ.loop(ui.el, (dt) => {
        if (S.done) { loop.stop(); return; }
        S.acc += dt * 10;                                       // 10 steps per second
        while (S.acc >= 1 && !S.done) { S.acc -= 1; advance(); }
        net.draw(); curve.draw(); readout();
      });

      let u = VIZ.rng(1);
      function build() {
        const g = VIZ.rng(S.seed * 31 + S.n);
        S.A = S.net === "ba" ? barabasiAlbert(S.n, 3, g) : erdosRenyi(S.n, 6 / (S.n - 1), g);
        S.pos = force(S.A, VIZ.rng(5));
        const k = S.A.map((s) => s.size), k1 = k.reduce((a, b) => a + b, 0) / S.n, k2 = k.reduce((a, b) => a + b * b, 0) / S.n;
        S.k1 = k1; S.k2 = k2;
        restart();
      }
      function counts() { const c = [0, 0, 0]; S.st.forEach((x) => c[x]++); return c.map((x) => x / S.n); }
      function restart() {
        u = VIZ.rng(S.seed * 7919 + 1);
        S.immune = choose(S.A, S.g, S.strategy, u);
        S.st = init(S.A, S.immune, u);
        S.t = 0; S.acc = 0; S.done = false; S.hist = [counts()];
        S.ever = new Uint8Array(S.n); S.st.forEach((x, i) => { if (x === 1) S.ever[i] = 1; });
        if (VIZ.reduceMotion) { while (!S.done) advance(); }
        net.draw(); curve.draw(); sweepP.draw(); readout();
        if (!VIZ.reduceMotion) loop.start();
      }
      function advance() {
        S.st = step(S.A, S.st, S.beta, S.mu, S.model === "sis", u);
        S.st.forEach((x, i) => { if (x === 1) S.ever[i] = 1; });
        S.t++; S.hist.push(counts());
        const I = S.hist[S.hist.length - 1][1];
        if (I === 0 || (S.model === "sis" && S.t >= 200) || S.t >= 400) S.done = true;
      }

      /* Average outcome over RUNS runs per (strategy, g), computed one point at a time */
      const RUNS = 16;
      function startSweep() {
        const gs = VIZ.linspace(0, 0.3, 7), strategies = ["random", "acquaintance", "targeted"];
        S.sweep = { gs, res: { random: [], acquaintance: [], targeted: [] }, todo: [] };
        strategies.forEach((s) => gs.forEach((g) => S.sweep.todo.push([s, g])));
        sweepBtn.setAttribute("aria-pressed", "true");
        const sw = S.sweep;
        const next = () => {
          if (S.sweep !== sw) return;
          const job = sw.todo.shift();
          if (!job) { sweepBtn.setAttribute("aria-pressed", "false"); readout(); return; }
          const [s, g] = job, r = VIZ.rng(1000 + sw.todo.length);
          let tot = 0;
          for (let run = 0; run < RUNS; run++) {
            let st = init(S.A, choose(S.A, g, s, r), r);
            const ever = st.map((x) => (x === 1 ? 1 : 0));
            let prev = 0;
            for (let t = 0; t < 400; t++) {
              st = step(S.A, st, S.beta, S.mu, S.model === "sis", r);
              let I = 0;
              st.forEach((x, i) => { if (x === 1) { ever[i] = 1; I++; } });
              if (S.model === "sis" && t >= 150) prev += I / S.n / 50;
              if (I === 0 || (S.model === "sis" && t >= 199)) break;
            }
            tot += S.model === "sis" ? prev : ever.reduce((a, b) => a + b, 0) / S.n;
          }
          sw.res[s].push([g, tot / RUNS]);
          sweepP.draw(); readout();
          setTimeout(next, 0);
        };
        next();
      }

      function drawNet(c) {
        if (!S.A) return;
        const T = VIZ.theme(), ctx = c.ctx, pad = 14;
        ctx.fillStyle = T.panel; ctx.fillRect(0, 0, c.w, c.h);
        const xs = S.pos.map((p) => p[0]), ys = S.pos.map((p) => p[1]);
        const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
        const sc = Math.min((c.w - 2 * pad) / (x1 - x0), (c.h - 2 * pad) / (y1 - y0));
        const ox = (c.w - sc * (x1 - x0)) / 2, oy = (c.h - sc * (y1 - y0)) / 2;
        const X = (i) => ox + sc * (S.pos[i][0] - x0), Y = (i) => oy + sc * (y1 - S.pos[i][1]);
        ctx.strokeStyle = T.axis; ctx.globalAlpha = S.n > 300 ? 0.18 : 0.3; ctx.lineWidth = 0.7;
        ctx.beginPath();
        S.A.forEach((nb, i) => nb.forEach((j) => { if (j > i) { ctx.moveTo(X(i), Y(i)); ctx.lineTo(X(j), Y(j)); } }));
        ctx.stroke(); ctx.globalAlpha = 1;
        const kmax = Math.max(...S.A.map((s) => s.size)), base = S.n > 300 ? 1.5 : 2.2;
        S.A.forEach((nb, i) => {
          const r = base + 5 * Math.sqrt(nb.size / kmax);
          ctx.beginPath(); ctx.arc(X(i), Y(i), r, 0, 2 * Math.PI);
          if (S.immune.has(i)) { ctx.strokeStyle = T.c[2]; ctx.lineWidth = 2; ctx.stroke(); return; }
          ctx.fillStyle = S.st[i] === 1 ? T.c[1] : S.st[i] === 2 ? T.c[0] : T.axis;
          ctx.globalAlpha = S.st[i] === 0 ? 0.55 : 1; ctx.fill(); ctx.globalAlpha = 1;
        });
      }

      function drawCurve(c) {
        const tmax = Math.max(50, S.hist.length - 1);
        const ax = VIZ.axes(c, { xlim: [0, tmax], ylim: [0, 1], xlabel: "time step t", ylabel: "fraction of nodes" });
        const T = ax.T, ts = S.hist.map((_, i) => i);
        ax.clip();
        ax.line(ts, S.hist.map((h) => h[0]), { color: T.axis, width: 2 });
        ax.line(ts, S.hist.map((h) => h[1]), { color: T.c[1], width: 2.5 });
        if (S.model === "sir") ax.line(ts, S.hist.map((h) => h[2]), { color: T.c[0], width: 2 });
        c.ctx.restore();
        VIZ.legend(c, [{ label: "S", color: T.axis }, { label: "I", color: T.c[1] }].concat(S.model === "sir" ? [{ label: "R (incl. vaccinated)", color: T.c[0] }] : []),
          ax.pad.l + ax.pw - 150, ax.pad.t + 14);
      }

      function drawSweep(c) {
        const ax = VIZ.axes(c, { xlim: [0, 0.3], ylim: [0, 1], xlabel: "vaccination coverage g", ylabel: S.model === "sir" ? "final size" : "prevalence" });
        const T = ax.T, cols = { random: T.c[1], acquaintance: T.c[2], targeted: T.c[0] };
        ax.clip();
        if (S.sweep) Object.keys(S.sweep.res).forEach((s) => {
          const pts = S.sweep.res[s];
          ax.line(pts.map((p) => p[0]), pts.map((p) => p[1]), { color: cols[s], width: 2 });
          pts.forEach(([g, v]) => ax.dot(g, v, 4, cols[s]));
        });
        c.ctx.restore();
        if (!S.sweep) ax.label("Press “Compare strategies over g” (16 runs per point)", 0.15, 0.5, { align: "center", size: 12 });
        else VIZ.legend(c, [{ label: "random", color: cols.random }, { label: "acquaintance", color: cols.acquaintance }, { label: "hubs first", color: cols.targeted }],
          ax.pad.l + ax.pw - 130, ax.pad.t + 14);
      }

      function readout() {
        const h = S.hist[S.hist.length - 1], lam = S.beta / S.mu;
        const pct = (x) => `${Math.round(100 * x)}%`;
        let txt = `t = ${S.t} · S ${pct(h[0])}, I ${pct(h[1])}` + (S.model === "sir" ? `, R ${pct(h[2])}` : "");
        if (S.done && S.model === "sir") {
          const ever = S.ever.reduce((a, b) => a + b, 0) / S.n;
          txt += ` · final epidemic size ${pct(ever)} of all nodes`;
        }
        txt += ` · λ = β/μ = ${lam.toFixed(2)}, <k>/<k²> = ${(S.k1 / S.k2).toFixed(3)} on this graph`;
        if (S.immune.size) txt += ` · ${S.immune.size} vaccinated`;
        if (S.sweep && S.sweep.todo.length) txt += ` · comparing strategies…`;
        ui.readout.textContent = txt + ".";
      }

      build();
    },
  });
})();

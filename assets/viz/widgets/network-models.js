/* Network growth models: Erdős–Rényi G(N, p), Watts–Strogatz (k, p) and Barabási–Albert (m).
   The generators follow the NetworkX algorithms used in the lab
   (nx.erdos_renyi_graph, nx.watts_strogatz_graph, nx.barabasi_albert_graph);
   default parameters are those of comparing-models.qmd. */
(function () {
  "use strict";

  /* ---------- generators (adjacency as arrays of Sets) ---------- */
  const empty = (n) => Array.from({ length: n }, () => new Set());
  const addEdge = (A, u, v) => { A[u].add(v); A[v].add(u); };

  function erdosRenyi(n, p, u) {
    const A = empty(n);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (u() < p) addEdge(A, i, j);
    return A;
  }
  function wattsStrogatz(n, k, p, u) {
    const A = empty(n), h = Math.floor(k / 2);
    for (let i = 0; i < n; i++) for (let j = 1; j <= h; j++) addEdge(A, i, (i + j) % n);
    // rewire each lattice edge (i, i+j) with probability p, as NetworkX does
    for (let j = 1; j <= h; j++) for (let i = 0; i < n; i++) {
      if (u() >= p) continue;
      const v = (i + j) % n;
      if (!A[i].has(v) || A[i].size >= n - 1) continue;
      let w;
      do { w = Math.floor(u() * n); } while (w === i || A[i].has(w));
      A[i].delete(v); A[v].delete(i); addEdge(A, i, w);
    }
    return A;
  }
  function barabasiAlbert(n, m, u) {
    const A = empty(n);
    for (let i = 1; i <= m; i++) addEdge(A, 0, i);          // initial star on m + 1 nodes
    const repeated = [];
    for (let i = 0; i <= m; i++) for (let d = 0; d < A[i].size; d++) repeated.push(i);
    for (let src = m + 1; src < n; src++) {
      const targets = new Set();
      while (targets.size < m) targets.add(repeated[Math.floor(u() * repeated.length)]);
      targets.forEach((t) => { addEdge(A, src, t); repeated.push(t, src); });
    }
    return A;
  }

  /* ---------- statistics ---------- */
  function components(A) {
    const n = A.length, comp = new Array(n).fill(-1), sizes = [];
    for (let s = 0; s < n; s++) {
      if (comp[s] >= 0) continue;
      const q = [s]; comp[s] = sizes.length; let c = 0;
      while (q.length) { const x = q.pop(); c++; A[x].forEach((y) => { if (comp[y] < 0) { comp[y] = comp[s]; q.push(y); } }); }
      sizes.push(c);
    }
    return { comp, sizes };
  }
  function stats(A) {
    const n = A.length, deg = A.map((s) => s.size);
    const E = deg.reduce((a, b) => a + b, 0) / 2;
    // average clustering (nodes with degree < 2 count as 0, as in nx.average_clustering)
    let C = 0;
    for (let i = 0; i < n; i++) {
      const nb = [...A[i]], d = nb.length;
      if (d < 2) continue;
      let t = 0;
      for (let a = 0; a < d; a++) for (let b = a + 1; b < d; b++) if (A[nb[a]].has(nb[b])) t++;
      C += (2 * t) / (d * (d - 1));
    }
    C /= n;
    // average shortest path length of the largest component
    const { comp, sizes } = components(A);
    const big = sizes.indexOf(Math.max(...sizes)), nodes = [];
    for (let i = 0; i < n; i++) if (comp[i] === big) nodes.push(i);
    let tot = 0;
    const dist = new Int32Array(n);
    nodes.forEach((s) => {
      dist.fill(-1); dist[s] = 0; const q = [s]; let h = 0;
      while (h < q.length) { const x = q[h++]; A[x].forEach((y) => { if (dist[y] < 0) { dist[y] = dist[x] + 1; tot += dist[y]; q.push(y); } }); }
    });
    const L = nodes.length > 1 ? tot / (nodes.length * (nodes.length - 1)) : NaN;
    return { n, E, deg, kmean: (2 * E) / n, C, L, ncomp: sizes.length, giant: Math.max(...sizes) / n };
  }

  /* ---------- layouts ---------- */
  function circular(n) { return Array.from({ length: n }, (_, i) => [Math.cos((2 * Math.PI * i) / n), Math.sin((2 * Math.PI * i) / n)]); }
  function force(A, u) {      // Fruchterman–Reingold, computed once
    const n = A.length, P = Array.from({ length: n }, () => [u() * 2 - 1, u() * 2 - 1]);
    const k = Math.sqrt(4 / n);
    let temp = 0.2;
    for (let it = 0; it < 220; it++) {
      const D = Array.from({ length: n }, () => [0, 0]);
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        let dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1];
        const d2 = Math.max(1e-6, dx * dx + dy * dy), f = (k * k) / d2;
        dx *= f; dy *= f; D[i][0] += dx; D[i][1] += dy; D[j][0] -= dx; D[j][1] -= dy;
      }
      for (let i = 0; i < n; i++) A[i].forEach((j) => {
        if (j < i) return;
        const dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], d = Math.hypot(dx, dy), f = d / k;
        D[i][0] -= dx * f; D[i][1] -= dy * f; D[j][0] += dx * f; D[j][1] += dy * f;
      });
      for (let i = 0; i < n; i++) {
        D[i][0] -= 0.02 * P[i][0]; D[i][1] -= 0.02 * P[i][1];   // weak gravity keeps components on screen
        const L = Math.hypot(D[i][0], D[i][1]) || 1, s = Math.min(L, temp) / L;
        P[i][0] += D[i][0] * s; P[i][1] += D[i][1] * s;
      }
      temp *= 0.985;
    }
    return P;
  }

  const logFact = (k) => { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; };

  VIZ.register("network-models", {
    title: "Random, small-world and scale-free graphs side by side",
    init(ui) {
      const DEF = { er: { n: 80, p: 0.06 }, ws: { n: 80, k: 6, p: 0.08 }, ba: { n: 200, m: 2 } };
      const S = { model: "er", P: JSON.parse(JSON.stringify(DEF)), seed: 7, log: false, A: null, pos: null, st: null };

      VIZ.select(ui.controls, { label: "Model", value: "er",
        options: [["er", "Erdős–Rényi G(N, p)"], ["ws", "Watts–Strogatz"], ["ba", "Barabási–Albert"]],
        onChange: (v) => { S.model = v; buildSliders(); regen(); } });
      const sliderBox = VIZ.group(ui.controls, "viz__group--sliders");
      const g = VIZ.group(ui.controls);
      VIZ.button(g, "New sample", () => { S.seed++; regen(); }, { primary: true });
      const logBtn = VIZ.button(g, "Log–log axes", () => { S.log = !S.log; logBtn.setAttribute("aria-pressed", String(S.log)); hist.draw(); });
      logBtn.setAttribute("aria-pressed", "false");

      function buildSliders() {
        sliderBox.textContent = "";
        const P = S.P[S.model], on = (key) => (v) => { P[key] = v; regen(); };
        VIZ.slider(sliderBox, { label: "<i>N</i>", min: 30, max: 300, step: 10, value: P.n, fmt: (v) => String(v), onInput: on("n") });
        if (S.model === "er") VIZ.slider(sliderBox, { label: "<i>p</i>", min: 0, max: 0.15, step: 0.002, value: P.p, fmt: (v) => v.toFixed(3), onInput: on("p") });
        if (S.model === "ws") {
          VIZ.slider(sliderBox, { label: "<i>k</i>", min: 2, max: 10, step: 2, value: P.k, fmt: (v) => String(v), onInput: on("k") });
          VIZ.slider(sliderBox, { label: "rewire <i>p</i>", min: 0, max: 1, step: 0.01, value: P.p, fmt: (v) => v.toFixed(2), onInput: on("p") });
        }
        if (S.model === "ba") VIZ.slider(sliderBox, { label: "<i>m</i>", min: 1, max: 5, step: 1, value: P.m, fmt: (v) => String(v), onInput: on("m") });
      }

      const net = ui.panel({ title: "Network", aspect: 1.1, minHeight: 280,
        label: "Drawing of the generated network; node size grows with degree", onDraw: drawNet });
      const hist = ui.panel({ title: "Degree distribution  p_k  (bars or dots) and theory (line)", aspect: 1.1, minHeight: 280,
        label: "Degree distribution of the generated network with the theoretical curve", onDraw: drawHist });

      let timer = 0;
      function regen() {             // debounce slider drags: the layout is the expensive part
        clearTimeout(timer);
        timer = setTimeout(() => {
          const u = VIZ.rng(S.seed * 7919 + 13), P = S.P[S.model];
          S.A = S.model === "er" ? erdosRenyi(P.n, P.p, u) : S.model === "ws" ? wattsStrogatz(P.n, P.k, P.p, u) : barabasiAlbert(P.n, P.m, u);
          S.pos = S.model === "ws" ? circular(P.n) : force(S.A, VIZ.rng(S.seed + 1));
          S.st = stats(S.A);
          net.draw(); hist.draw(); readout();
        }, 60);
      }

      function drawNet(c) {
        if (!S.A) return;
        const T = VIZ.theme(), ctx = c.ctx, pad = 16;
        ctx.fillStyle = T.panel; ctx.fillRect(0, 0, c.w, c.h);
        const xs = S.pos.map((p) => p[0]), ys = S.pos.map((p) => p[1]);
        const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
        const sc = Math.min((c.w - 2 * pad) / (x1 - x0 || 1), (c.h - 2 * pad) / (y1 - y0 || 1));
        const ox = (c.w - sc * (x1 - x0)) / 2, oy = (c.h - sc * (y1 - y0)) / 2;
        const X = (i) => ox + sc * (S.pos[i][0] - x0), Y = (i) => oy + sc * (y1 - S.pos[i][1]);
        ctx.strokeStyle = T.axis; ctx.globalAlpha = 0.45; ctx.lineWidth = 0.8;
        ctx.beginPath();
        S.A.forEach((nb, i) => nb.forEach((j) => { if (j > i) { ctx.moveTo(X(i), Y(i)); ctx.lineTo(X(j), Y(j)); } }));
        ctx.stroke(); ctx.globalAlpha = 1;
        const kmax = Math.max(...S.st.deg);
        S.A.forEach((nb, i) => {
          const r = 2 + 6 * Math.sqrt(nb.size / Math.max(kmax, 1));
          ctx.beginPath(); ctx.arc(X(i), Y(i), r, 0, 2 * Math.PI);
          ctx.fillStyle = nb.size === kmax && kmax > 3 * S.st.kmean ? T.c[1] : T.c[0]; ctx.fill();
        });
      }

      function drawHist(c) {
        if (!S.st) return;
        const { deg, n, kmean } = S.st, kmax = Math.max(...deg, 1), P = S.P[S.model];
        const counts = new Array(kmax + 1).fill(0);
        deg.forEach((k) => counts[k]++);
        const pk = counts.map((x) => x / n);
        // theory: Poisson for ER, p_k = 2m(m+1)/(k(k+1)(k+2)) for BA; none for WS
        const theory = (k) => S.model === "er" ? Math.exp(-kmean + k * Math.log(Math.max(kmean, 1e-9)) - logFact(k))
          : S.model === "ba" ? (k >= P.m ? (2 * P.m * (P.m + 1)) / (k * (k + 1) * (k + 2)) : 0) : NaN;
        let ax;
        const T = VIZ.theme();
        if (!S.log) {
          ax = VIZ.axes(c, { xlim: [0, kmax + 1], ylim: [0, Math.max(...pk) * 1.15 || 1], xlabel: "degree k", ylabel: "p_k" });
          ax.clip();
          const bw = Math.max(2, (ax.sx(1) - ax.sx(0)) * 0.75);
          pk.forEach((p, k) => { if (p > 0) { c.ctx.fillStyle = T.c[0]; c.ctx.fillRect(ax.sx(k) - bw / 2, ax.sy(p), bw, ax.sy(0) - ax.sy(p)); } });
          if (S.model !== "ws") {
            const ks = VIZ.linspace(0, kmax + 1, kmax + 2).map(Math.round);
            ax.line(ks, ks.map(theory), { color: T.c[1], width: 2 });
          }
        } else {
          const lk = (k) => Math.log10(k), lp = (p) => Math.log10(p);
          ax = VIZ.axes(c, { xlim: [0, Math.log10(Math.max(kmax, 10)) + 0.1], ylim: [-3.5, 0], xlabel: "log₁₀ k", ylabel: "log₁₀ p_k" });
          ax.clip();
          pk.forEach((p, k) => { if (p > 0 && k > 0) ax.dot(lk(k), lp(p), 4, T.c[0]); });
          if (S.model !== "ws") {
            const ks = VIZ.linspace(1, Math.max(kmax, 10), 200);
            const kk = S.model === "er" ? ks.map(Math.round) : ks;
            ax.line(kk.map(lk), kk.map((k) => { const t = theory(k); return t > 0 ? lp(t) : NaN; }), { color: T.c[1], width: 2 });
          }
        }
        c.ctx.restore();
        VIZ.legend(c, [{ label: "this graph", color: T.c[0] }].concat(S.model === "ws" ? [] :
          [{ label: S.model === "er" ? "Poisson, mean <k>" : "2m(m+1)/(k(k+1)(k+2))", color: T.c[1] }]),
          ax.pad.l + ax.pw - 170, ax.pad.t + 14);
      }

      function readout() {
        const s = S.st;
        ui.readout.textContent = `N = ${s.n}, E = ${s.E}, <k> = ${s.kmean.toFixed(2)}, max degree ${Math.max(...s.deg)} · ` +
          `clustering C = ${s.C.toFixed(3)} · path length L = ${s.L.toFixed(2)} (largest component) · ` +
          `${s.ncomp} component${s.ncomp > 1 ? "s" : ""}, largest holds ${Math.round(100 * s.giant)}% of nodes.`;
      }

      buildSliders();
      regen();
    },
  });
})();

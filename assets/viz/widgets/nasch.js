/* Nagel-Schreckenberg traffic on a ring (as amlab.cellular_automata.traffic_models):
   1) v ← min(v + 1, vmax)  2) v ← min(v, gap)  3) with probability p, v ← max(v − 1, 0)  4) x ← x + v.
   Flow q = Σv / L. "Measure" reproduces mean_flow(): ring of 400 cells, random start at rest,
   200 burn-in steps, average over the next 400. */
(function () {
  "use strict";
  const L = 200, HIST = 200;

  function initial(n, length, rng) {          // ring_initial(mode="random"): random cells, all at rest
    const cells = Array.from({ length }, (_, i) => i);
    for (let i = 0; i < n; i++) { const j = i + Math.floor(rng() * (length - i)); [cells[i], cells[j]] = [cells[j], cells[i]]; }
    const x = Int32Array.from(cells.slice(0, n).sort((a, b) => a - b));
    return { x, v: new Int32Array(n) };
  }
  function step(st, length, vmax, p, rng) {     // nasch_step(): parallel update, cars never overtake
    const { x, v } = st, n = x.length;
    if (!n) return;
    const gap = new Int32Array(n);
    for (let i = 0; i < n; i++) gap[i] = n === 1 ? length - 1 : (((x[(i + 1) % n] - x[i] - 1) % length) + length) % length;
    for (let i = 0; i < n; i++) {
      let s = Math.min(v[i] + 1, vmax);
      s = Math.min(s, gap[i]);
      if (rng() < p) s = Math.max(s - 1, 0);
      v[i] = s;
    }
    for (let i = 0; i < n; i++) x[i] = (x[i] + v[i]) % length;
  }
  function meanFlow(rho, vmax, p, seed) {
    const length = 400, n = Math.max(1, Math.round(rho * length)), rng = VIZ.rng(seed);
    const st = initial(n, length, rng);
    let total = 0;
    for (let t = 0; t < 600; t++) {
      step(st, length, vmax, p, rng);
      if (t >= 200) for (let i = 0; i < n; i++) total += st.v[i];
    }
    return total / (length * 400);
  }

  VIZ.register("nasch", {
    title: "Traffic on a ring road: where do jams come from?",
    init(ui) {
      const S = { rho: 0.2, vmax: 5, p: 0.3, seed: 1, speed: 15, hist: [], meas: [], acc: 0, qHist: [] };

      VIZ.slider(ui.controls, { label: "density ρ", min: 0.02, max: 0.9, step: 0.01, value: S.rho,
        fmt: (x) => x.toFixed(2), onInput: (x) => { S.rho = x; reset(); } });
      VIZ.slider(ui.controls, { label: "<i>v</i><sub>max</sub>", min: 1, max: 8, step: 1, value: S.vmax,
        fmt: (x) => String(x), onInput: (x) => { S.vmax = x; S.meas = []; draw(); } });
      VIZ.slider(ui.controls, { label: "braking <i>p</i>", min: 0, max: 0.9, step: 0.05, value: S.p,
        fmt: (x) => x.toFixed(2), onInput: (x) => { S.p = x; draw(); } });
      VIZ.select(ui.controls, { label: "Speed", value: "15", options: [["4", "slow"], ["15", "normal"], ["60", "fast"]],
        onChange: (x) => { S.speed = +x; } });
      const g = VIZ.group(ui.controls);
      VIZ.button(g, "Measure q at this ρ", () => {
        S.meas.push({ rho: S.rho, q: meanFlow(S.rho, S.vmax, S.p, 1000 + S.meas.length), p: S.p }); draw();
      }, { primary: true });
      VIZ.button(g, "Sweep all densities", () => {
        for (let r = 0.02; r <= 0.9 + 1e-9; r += 0.04) S.meas.push({ rho: r, q: meanFlow(r, S.vmax, S.p, Math.round(r * 1e4)), p: S.p });
        draw();
      });
      VIZ.button(g, "Clear points", () => { S.meas = []; draw(); });
      VIZ.button(g, "Restart", () => { S.seed++; reset(); });

      const road = ui.panel({ title: "The ring road, unrolled (colour = speed; red cars are stopped)", cls: "viz__panel--wide",
        height: 54, label: "Cars on the road coloured by speed", onDraw: drawRoad });
      const st = ui.panel({ title: "Space-time diagram (time runs downward)", aspect: 1.15, minHeight: 250,
        label: "Space-time diagram: cars as dots, jams as dark bands moving backwards", onDraw: drawST });
      const fd = ui.panel({ title: "Fundamental diagram  q(ρ)", aspect: 1.15, minHeight: 250,
        label: "Flow against density: measurements and the deterministic p = 0 curve", onDraw: drawFD });

      const loop = VIZ.loop(ui.el, (dt) => {
        S.acc += dt * S.speed;
        let k = 0;
        while (S.acc >= 1 && k < 20) { tick(); S.acc -= 1; k++; }
        if (k) draw();
      });

      function reset() {
        S.rng = VIZ.rng(S.seed * 31 + 7);
        S.state = initial(Math.max(1, Math.round(S.rho * L)), L, S.rng);
        S.hist = []; S.qHist = []; S.t = 0;
        record();
        if (VIZ.reduceMotion) for (let i = 0; i < 150; i++) tick();
        draw();
      }
      function tick() { step(S.state, L, S.vmax, S.p, S.rng); S.t++; record(); }
      function record() {
        const row = new Int8Array(L).fill(-1);
        let sum = 0;
        for (let i = 0; i < S.state.x.length; i++) { row[S.state.x[i]] = S.state.v[i]; sum += S.state.v[i]; }
        S.hist.push(row); if (S.hist.length > HIST) S.hist.shift();
        S.qHist.push(sum / L); if (S.qHist.length > 100) S.qHist.shift();
      }
      function draw() { road.draw(); st.draw(); fd.draw(); readout(); }

      function speedColour(T, v) {
        if (v === 0) return T.c[1];
        return mix(rgbStr(T.c[0]), rgbStr(T.stable), Math.min(1, v / Math.max(1, S.vmax)));
      }
      function drawRoad(c) {
        const T = VIZ.theme(), ctx = c.ctx, w = c.w / L, y = 14, hgt = c.h - 26;
        ctx.fillStyle = T.panel; ctx.fillRect(0, y - 4, c.w, hgt + 8);
        const { x, v } = S.state;
        for (let i = 0; i < x.length; i++) {
          ctx.fillStyle = speedColour(T, v[i]);
          ctx.fillRect(x[i] * w + 0.5, y, Math.max(1, w - 1), hgt);
        }
        ctx.font = T.font(10, 600); ctx.fillStyle = T.text; ctx.textAlign = "left";
        ctx.fillText("driving direction →", 2, 10);
        ctx.textAlign = "right"; ctx.fillText(`${x.length} cars, ${L} cells`, c.w - 2, 10);
      }
      function drawST(c) {
        const T = VIZ.theme(), ctx = c.ctx, img = ctx.createImageData(L, HIST);
        const bg = rgbArr(T.panel), car = rgbArr(T.stable), stop = rgbArr(T.c[1]);
        for (let r = 0; r < HIST; r++) {
          const row = S.hist[r];
          for (let i = 0; i < L; i++) {
            const o = 4 * (r * L + i), s = row ? row[i] : -1, col = s < 0 ? bg : s === 0 ? stop : car;
            img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
          }
        }
        const oc = document.createElement("canvas"); oc.width = L; oc.height = HIST;
        oc.getContext("2d").putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(oc, 0, 0, c.w, c.h);
        ctx.strokeStyle = T.axis; ctx.strokeRect(0.5, 0.5, c.w - 1, c.h - 1);
      }
      function drawFD(c) {
        const ax = VIZ.axes(c, { xlim: [0, 1], ylim: [0, 1], xlabel: "density ρ", ylabel: "flow q" });
        const T = ax.T;
        ax.clip();
        const rs = VIZ.linspace(0, 1, 300);
        ax.line(rs, rs.map((r) => Math.min(r * S.vmax, 1 - r)), { color: T.stable, width: 1.5, dash: [6, 4] });
        ax.line([S.rho, S.rho], [0, 1], { color: T.accent, width: 2, alpha: 0.6 });
        S.meas.forEach((m) => ax.dot(m.rho, m.q, 4.5, Math.abs(m.p - S.p) < 1e-9 ? T.c[1] : T.axis, T.panel));
        const qNow = S.qHist.reduce((a, b) => a + b, 0) / Math.max(1, S.qHist.length);
        ax.dot(S.rho, qNow, 6, null, T.c[0]);
        c.ctx.restore();
        const lx = ax.pad.l + ax.pw - 196, ly = ax.pad.t + 4;
        c.ctx.save(); c.ctx.globalAlpha = 0.9; c.ctx.fillStyle = T.panel; c.ctx.fillRect(lx - 4, ly, 192, 50); c.ctx.restore();
        VIZ.legend(c, [{ label: `p = 0 exact: min(${S.vmax}ρ, 1 − ρ)`, color: T.stable, dash: [6, 4] },
          { label: "measured (current p)", color: T.c[1] }, { label: "live ring, last 100 steps", color: T.c[0] }],
          lx + 2, ly + 12);
      }
      function readout() {
        const { v } = S.state, n = v.length;
        let sum = 0, stopped = 0;
        for (let i = 0; i < n; i++) { sum += v[i]; if (v[i] === 0) stopped++; }
        const q = S.qHist.reduce((a, b) => a + b, 0) / Math.max(1, S.qHist.length);
        const crit = 1 / (S.vmax + 1);
        ui.readout.textContent = `ρ = ${(n / L).toFixed(2)} (${n} cars) · mean speed now ${(sum / n).toFixed(2)} cells/step · ` +
          `${Math.round((100 * stopped) / n)}% stopped · flow over the last 100 steps q = ${q.toFixed(3)} · ` +
          `for p = 0: q = ${Math.min(S.vmax * S.rho, 1 - S.rho).toFixed(3)}, maximum at ρ = 1/(v_max+1) = ${crit.toFixed(3)}.`;
      }

      reset();
      if (!VIZ.reduceMotion) loop.start();
    },
  });

  function rgbArr(css) {
    const m = css.trim().match(/^#([0-9a-f]{6})$/i);
    if (m) return [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
    const n = css.match(/\d+(\.\d+)?/g) || [0, 0, 0];
    return n.slice(0, 3).map(Number);
  }
  function rgbStr(css) { return rgbArr(css); }
  function mix(a, b, t) { return `rgb(${[0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)).join(",")})`; }
})();

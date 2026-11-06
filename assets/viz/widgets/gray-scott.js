/* Gray-Scott pattern zoo (as amlab.pdes_2d.gray_scott):
     u_t = d1 Δu − u v² + f (1 − u),   v_t = d2 Δv + u v² − (f + k) v
   d1 = 0.1, d2 = 0.05, dx = 1, dt = 2, explicit Euler, periodic 5-point Laplacian.
   Click or drag to add v (square of radius 3 set to 0.5 × (1 + 10% noise)). */
(function () {
  "use strict";
  const N = 128, D1 = 0.1, D2 = 0.05, DT = 2, STEPS_PER_FRAME = 10, BRUSH = 3;
  // Types A–E: table on "Gray-Scott Parameters"; the last three are classic extra examples.
  const PRESETS = [
    ["A", "A · labyrinth", 0.040, 0.060], ["B", "B · chaotic waves", 0.014, 0.047],
    ["C", "C · stable spots", 0.062, 0.065], ["D", "D · ring", 0.078, 0.061],
    ["E", "E · expanding front", 0.082, 0.059], ["worms", "worms", 0.058, 0.065],
    ["mitosis", "mitosis (dividing spots)", 0.0367, 0.0649], ["coral", "coral", 0.0545, 0.062],
  ];

  VIZ.register("gray-scott", {
    title: "Gray–Scott: paint a perturbation and watch a pattern grow",
    init(ui) {
      const S = { f: 0.040, k: 0.060, t: 0, seed: 1, paused: false, erase: false,
        u: new Float32Array(N * N), v: new Float32Array(N * N), u2: new Float32Array(N * N), v2: new Float32Array(N * N) };
      const rnd = VIZ.rng(12345);

      const sF = VIZ.slider(ui.controls, { label: "<i>f</i> (feed)", min: 0.01, max: 0.09, step: 0.0005, value: S.f,
        fmt: (x) => x.toFixed(4), onInput: (x) => { S.f = x; preset.set(""); mapP.draw(); readout(); } });
      const sK = VIZ.slider(ui.controls, { label: "<i>k</i> (kill)", min: 0.03, max: 0.07, step: 0.0005, value: S.k,
        fmt: (x) => x.toFixed(4), onInput: (x) => { S.k = x; preset.set(""); mapP.draw(); readout(); } });
      const preset = VIZ.select(ui.controls, { label: "Preset", value: "A",
        options: [["", "custom"]].concat(PRESETS.map((p) => [p[0], p[1]])),
        onChange: (id) => { const p = PRESETS.find((q) => q[0] === id); if (p) setFK(p[2], p[3], true); } });
      const g = VIZ.group(ui.controls);
      const playBtn = VIZ.button(g, "Pause", togglePause, { primary: true });
      VIZ.button(g, "Restart (central square)", () => { seedSquare(); }, {});
      VIZ.button(g, "Clear", () => { clear(); draw(); }, {});
      const eraseBtn = VIZ.button(g, "Brush: paint", () => {
        S.erase = !S.erase; eraseBtn.textContent = S.erase ? "Brush: erase" : "Brush: paint";
        eraseBtn.setAttribute("aria-pressed", String(S.erase));
      });

      const field = ui.panel({ title: "v(x, y)  ·  click or drag to paint", aspect: 1, minHeight: 260,
        label: "Concentration of v on a 128 by 128 periodic grid; click or drag to add v", onDraw: drawField,
        onPointer: (e, c) => paint(e, c) });
      const mapP = ui.panel({ title: "Parameter plane (f, k)  ·  click to choose", aspect: 1, minHeight: 260,
        label: "Feed and kill parameter plane with the preset patterns marked", onDraw: drawMap,
        onPointer: (e, c) => {
          if (e.type !== "pointerdown" || !c.ax || !c.ax.inside(e.x, e.y)) return;
          setFK(Math.min(0.09, Math.max(0.01, c.ax.ix(e.x))), Math.min(0.07, Math.max(0.03, c.ax.iy(e.y))), false);
          preset.set("");
        } });

      const loop = VIZ.loop(ui.el, () => { for (let s = 0; s < STEPS_PER_FRAME; s++) step(); draw(); });

      function setFK(f, k, restart) {
        S.f = f; S.k = k; sF.set(f); sK.set(k);
        if (restart) seedSquare(); else { mapP.draw(); readout(); }
      }
      function clear() { S.u.fill(1); S.v.fill(0); S.t = 0; }
      function seedSquare() {      // as on "Gray-Scott Parameters": square set to 0.5, times (1 + 10% noise)
        clear();
        const r = 8, c = N >> 1, u = VIZ.rng(++S.seed);
        for (let y = c - r; y < c + r; y++) for (let x = c - r; x < c + r; x++) {
          S.u[y * N + x] = 0.5 * (1 + 0.1 * u.normal());
          S.v[y * N + x] = 0.5 * (1 + 0.1 * u.normal());
        }
        if (VIZ.reduceMotion) { for (let s = 0; s < 1500; s++) step(); }
        draw();
      }
      function step() {
        const u = S.u, v = S.v, un = S.u2, vn = S.v2, f = S.f, fk = S.f + S.k;
        for (let y = 0; y < N; y++) {
          const yu = ((y + N - 1) % N) * N, yd = ((y + 1) % N) * N, yc = y * N;
          for (let x = 0; x < N; x++) {
            const xl = (x + N - 1) % N, xr = (x + 1) % N, i = yc + x;
            const ui = u[i], vi = v[i], uvv = ui * vi * vi;
            const lu = u[yu + x] + u[yd + x] + u[yc + xl] + u[yc + xr] - 4 * ui;
            const lv = v[yu + x] + v[yd + x] + v[yc + xl] + v[yc + xr] - 4 * vi;
            un[i] = ui + DT * (D1 * lu - uvv + f * (1 - ui));
            vn[i] = vi + DT * (D2 * lv + uvv - fk * vi);
          }
        }
        S.u = un; S.v = vn; S.u2 = u; S.v2 = v; S.t += DT;
      }
      function paint(e, c) {
        const side = Math.min(c.w, c.h), ox = (c.w - side) / 2;
        const cx = Math.floor(((e.x - ox) / side) * N), cy = Math.floor((e.y / side) * N);
        if (cx < 0 || cy < 0 || cx >= N || cy >= N) return;
        for (let y = cy - BRUSH; y < cy + BRUSH; y++) for (let x = cx - BRUSH; x < cx + BRUSH; x++) {
          const i = ((y + N) % N) * N + ((x + N) % N);
          if (S.erase) { S.u[i] = 1; S.v[i] = 0; }
          else { S.u[i] = 0.5 * (1 + 0.1 * rnd.normal()); S.v[i] = 0.5 * (1 + 0.1 * rnd.normal()); }
        }
        if (VIZ.reduceMotion && !S.erase) { for (let s = 0; s < 300; s++) step(); }
        draw();
      }
      function togglePause() {
        S.paused = !S.paused;
        playBtn.textContent = S.paused ? "Play" : "Pause";
        if (S.paused) loop.stop(); else loop.start();
      }
      function draw() { field.draw(); readout(); }

      const off = document.createElement("canvas"); off.width = N; off.height = N;
      const offCtx = off.getContext("2d"), img = offCtx.createImageData(N, N);
      function drawField(c) {
        const T = VIZ.theme(), lo = rgb(T.panel), mid = rgb(T.c[0]), hi = rgb(T.stable);
        const v = S.v, data = img.data;
        for (let i = 0; i < N * N; i++) {
          const z = Math.max(0, Math.min(1, v[i] / 0.5));       // colour scale 0 ≤ v ≤ 0.5
          const a = z < 0.5 ? lo : mid, b = z < 0.5 ? mid : hi, s = z < 0.5 ? 2 * z : 2 * z - 1;
          data[4 * i] = a[0] + (b[0] - a[0]) * s; data[4 * i + 1] = a[1] + (b[1] - a[1]) * s;
          data[4 * i + 2] = a[2] + (b[2] - a[2]) * s; data[4 * i + 3] = 255;
        }
        offCtx.putImageData(img, 0, 0);
        const side = Math.min(c.w, c.h), ox = (c.w - side) / 2;
        c.ctx.imageSmoothingEnabled = true;
        c.ctx.drawImage(off, ox, 0, side, side);
        c.ctx.strokeStyle = T.axis; c.ctx.strokeRect(ox + 0.5, 0.5, side - 1, side - 1);
      }
      function drawMap(c) {
        const ax = (c.ax = VIZ.axes(c, { xlim: [0.01, 0.09], ylim: [0.03, 0.07], xlabel: "f (feed)", ylabel: "k (kill)",
          xticks: [0.02, 0.04, 0.06, 0.08], yticks: [0.03, 0.04, 0.05, 0.06, 0.07], fmtX: (x) => x.toFixed(2), fmtY: (x) => x.toFixed(2) }));
        const T = ax.T;
        ax.clip();
        // 4 (f + k)² = f, i.e. k = √f / 2 − f: nontrivial uniform states exist below this curve
        const fs = VIZ.linspace(0.01, 0.09, 200);
        ax.line(fs, fs.map((f) => Math.sqrt(f) / 2 - f), { color: T.stable, width: 2, dash: [6, 4] });
        PRESETS.forEach((p) => {
          ax.dot(p[2], p[3], 5, T.c[0], T.panel);
          const left = p[0] === "worms" || p[0] === "mitosis";
          ax.label(p[0], p[2], p[3], { dx: left ? -7 : 7, dy: -5, size: 10, align: left ? "right" : "left" });
        });
        ax.dot(S.f, S.k, 7, T.accent, T.stable);
        c.ctx.restore();
        ax.label("4(f + k)² = f", 0.0895, 0.0545, { size: 10, color: T.text, align: "right" });
      }
      function readout() {
        let cover = 0;
        for (let i = 0; i < N * N; i++) if (S.v[i] > 0.15) cover++;
        const gap = S.f - 4 * (S.f + S.k) ** 2, exists = gap >= -1e-9;
        ui.readout.textContent = `f = ${S.f.toFixed(4)}, k = ${S.k.toFixed(4)} · t = ${S.t.toFixed(0)} · v > 0.15 on ${(100 * cover / (N * N)).toFixed(0)}% of the grid · ` +
          (Math.abs(gap) < 2e-4 ? "on the curve 4(f+k)² = f, where the nontrivial uniform states are born." :
           exists ? "nontrivial uniform states exist (4(f+k)² ≤ f)." : "only the trivial state (1, 0) exists here: the pattern lives on finite perturbations.");
      }

      seedSquare();
      if (VIZ.reduceMotion) { S.paused = true; playBtn.textContent = "Play"; } else loop.start();
    },
  });

  function rgb(css) {
    const m = css.trim().match(/^#([0-9a-f]{6})$/i);
    if (m) return [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
    const n = css.match(/\d+(\.\d+)?/g) || [0, 0, 0];
    return n.slice(0, 3).map(Number);
  }
})();

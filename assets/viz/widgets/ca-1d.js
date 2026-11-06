/* Elementary cellular automata (as amlab.cellular_automata.elementary):
   bit n of the rule number is the new state for the neighbourhood with value
   n = 4·left + 2·centre + right. 200 cells, 150 rows, time runs downward. */
(function () {
  "use strict";
  const W = 200, STEPS = 150;
  // Wolfram classes of the minimal representatives of the 88 equivalence classes
  // (Wolfram 1984; list as in Martínez 2013). Every other representative is class 2.
  const CLASS1 = [0, 8, 32, 40, 128, 136, 160, 168];
  const CLASS3 = [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150];
  const CLASS4 = [41, 54, 106, 110];
  const NAMES = { 1: "homogeneous", 2: "simple, stable or periodic", 3: "chaotic", 4: "complex, localized structures" };

  const table = (rule) => Array.from({ length: 8 }, (_, n) => (rule >> n) & 1);
  function fromTable(t) { return t.reduce((acc, b, n) => acc | (b << n), 0); }
  function reflect(rule) {
    const t = table(rule), out = new Array(8);
    for (let n = 0; n < 8; n++) { const l = (n >> 2) & 1, c = (n >> 1) & 1, r = n & 1; out[n] = t[4 * r + 2 * c + l]; }
    return fromTable(out);
  }
  function complement(rule) {
    const t = table(rule), out = new Array(8);
    for (let n = 0; n < 8; n++) out[n] = 1 - t[7 - n];
    return fromTable(out);
  }
  function equivalents(rule) {
    return Array.from(new Set([rule, reflect(rule), complement(rule), reflect(complement(rule))])).sort((a, b) => a - b);
  }
  function wolframClass(rule) {
    const m = equivalents(rule)[0];
    return CLASS1.includes(m) ? 1 : CLASS3.includes(m) ? 3 : CLASS4.includes(m) ? 4 : 2;
  }
  function simulate(row0, rule, boundary) {
    const t = table(rule), grid = [Uint8Array.from(row0)];
    for (let s = 1; s < STEPS; s++) {
      const prev = grid[s - 1], next = new Uint8Array(W);
      for (let i = 0; i < W; i++) {
        let l, r;
        if (boundary === "periodic") { l = prev[(i + W - 1) % W]; r = prev[(i + 1) % W]; }
        else { l = i === 0 ? 0 : prev[i - 1]; r = i === W - 1 ? 0 : prev[i + 1]; }   // fixed ghost cells = 0
        next[i] = t[4 * l + 2 * prev[i] + r];
      }
      grid.push(next);
    }
    return grid;
  }

  VIZ.register("ca-1d", {
    title: "Elementary cellular automata: 256 rules, four classes",
    init(ui) {
      const S = { rule: 30, init: "random", density: 0.5, boundary: "periodic", seed: 3, damage: false };

      const sR = VIZ.slider(ui.controls, { label: "Rule", min: 0, max: 255, step: 1, value: S.rule,
        fmt: (v) => String(v), onInput: (v) => setRule(v) });
      const g1 = VIZ.group(ui.controls);
      [30, 90, 110, 184, 32, 108].forEach((r) => VIZ.button(g1, String(r), () => setRule(r)));
      VIZ.select(ui.controls, { label: "Start", value: S.init, options: [["random", "random row"], ["single", "single cell"]],
        onChange: (v) => { S.init = v; sD.disable(v !== "random"); update(); } });
      const sD = VIZ.slider(ui.controls, { label: "density", min: 0.05, max: 0.95, step: 0.05, value: S.density,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.density = v; update(); } });
      VIZ.select(ui.controls, { label: "Boundary", value: S.boundary, options: [["periodic", "periodic"], ["fixed", "fixed (0)"]],
        onChange: (v) => { S.boundary = v; update(); } });
      const g2 = VIZ.group(ui.controls);
      VIZ.button(g2, "New random row", () => { S.seed++; update(); });
      const dmg = VIZ.button(g2, "Flip the centre cell", () => {
        S.damage = !S.damage; dmg.setAttribute("aria-pressed", String(S.damage)); update();
      });

      const tbl = ui.panel({ title: "Rule table  ·  click an output to flip it", cls: "viz__panel--wide", height: 92,
        label: "The eight neighbourhoods and the new state the rule assigns to each", onDraw: drawTable,
        onPointer: (e, c) => {
          if (e.type !== "pointerdown") return;
          const n = 7 - Math.floor((e.x - c.x0) / c.cw);
          if (n < 0 || n > 7 || e.y < c.y0) return;
          const t = table(S.rule); t[n] ^= 1; setRule(fromTable(t));
        } });
      const diag = ui.panel({ title: "Space-time diagram (200 cells, time runs downward)", cls: "viz__panel--wide", aspect: 1.75, minHeight: 230,
        label: "Space-time diagram of the cellular automaton", onDraw: drawDiagram });

      function setRule(r) { S.rule = r; sR.set(r); update(); }
      function row0() {
        const row = new Uint8Array(W);
        if (S.init === "single") row[W >> 1] = 1;
        else { const u = VIZ.rng(S.seed * 7919 + 13); for (let i = 0; i < W; i++) row[i] = u() < S.density ? 1 : 0; }
        return row;
      }
      function update() {
        const r0 = row0();
        S.grid = simulate(r0, S.rule, S.boundary);
        if (S.damage) { const r1 = Uint8Array.from(r0); r1[W >> 1] ^= 1; S.grid2 = simulate(r1, S.rule, S.boundary); }
        else S.grid2 = null;
        tbl.draw(); diag.draw(); readout();
      }

      function drawTable(c) {
        const T = VIZ.theme(), ctx = c.ctx, t = table(S.rule);
        const cw = Math.min(80, (c.w - 8) / 8), x0 = (c.w - 8 * cw) / 2, cell = Math.min(14, cw / 4.2), y0 = 6;
        c.x0 = x0; c.cw = cw; c.y0 = 0;
        for (let k = 0; k < 8; k++) {
          const n = 7 - k, bits = [(n >> 2) & 1, (n >> 1) & 1, n & 1];
          const cx = x0 + k * cw + cw / 2;
          bits.forEach((b, j) => {
            ctx.fillStyle = b ? T.stable : T.panel; ctx.strokeStyle = T.axis;
            ctx.fillRect(cx - 1.5 * cell + j * cell, y0, cell, cell); ctx.strokeRect(cx - 1.5 * cell + j * cell + 0.5, y0 + 0.5, cell - 1, cell - 1);
          });
          ctx.fillStyle = t[n] ? T.stable : T.panel; ctx.strokeStyle = T.accent; ctx.lineWidth = 2;
          ctx.fillRect(cx - cell / 2, y0 + cell + 6, cell, cell); ctx.strokeRect(cx - cell / 2 + 1, y0 + cell + 7, cell - 2, cell - 2);
          ctx.lineWidth = 1;
          const narrow = cw < 64;
          ctx.font = T.font(narrow ? 9 : 11, 700); ctx.fillStyle = T.text; ctx.textAlign = "center";
          ctx.fillText(narrow ? bits.join("") : `${bits.join("")} → ${t[n]}`, cx, y0 + 2 * cell + 24);
        }
        ctx.font = T.font(11, 600); ctx.textAlign = "right"; ctx.fillStyle = T.text;
        ctx.fillText(`${S.rule} = ${t.slice().reverse().join("")}₂`, c.w - 4, c.h - 4);
      }
      function drawDiagram(c) {
        const T = VIZ.theme(), ctx = c.ctx, on = rgb(T.stable), off = rgb(T.panel), hit = rgb(T.c[1]);
        const img = ctx.createImageData(W, STEPS);
        for (let s = 0; s < STEPS; s++) for (let i = 0; i < W; i++) {
          const o = 4 * (s * W + i), v = S.grid[s][i];
          const col = S.grid2 && S.grid2[s][i] !== v ? hit : v ? on : off;
          img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
        }
        const oc = document.createElement("canvas"); oc.width = W; oc.height = STEPS;
        oc.getContext("2d").putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(oc, 0, 0, c.w, c.h);
      }
      function readout() {
        const cls = wolframClass(S.rule), eq = equivalents(S.rule), last = S.grid[STEPS - 1];
        const ones = last.reduce((a, b) => a + b, 0), lambda = table(S.rule).reduce((a, b) => a + b, 0) / 8;
        let txt = `Rule ${S.rule} · equivalent rules ${eq.join(", ")} · Wolfram class ${cls} (${NAMES[cls]}) · Langton λ = ${lambda.toFixed(3)} · ` +
          `density of 1s in the last row = ${(ones / W).toFixed(2)}`;
        if (S.grid2) {
          const ham = S.grid[STEPS - 1].reduce((a, v, i) => a + (v !== S.grid2[STEPS - 1][i] ? 1 : 0), 0);
          txt += ` · after ${STEPS - 1} steps the flipped run differs in ${ham} cells (shown in red)`;
        }
        ui.readout.textContent = txt + ".";
      }

      sD.disable(false);
      update();
    },
  });

  function rgb(css) {
    const m = css.trim().match(/^#([0-9a-f]{6})$/i);
    if (m) return [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
    const n = css.match(/\d+(\.\d+)?/g) || [0, 0, 0];
    return n.slice(0, 3).map(Number);
  }
})();

/* Crowd leaving a room through one door: faster is slower, and a column in front of the door.
   Port of amlab/collective_motion/crowd_exit.py, the physics-based social force model of
   Helbing, Farkas & Vicsek, Nature 407, 487 (2000) (published parameters, time step 0.002 s):
     m dv/dt = m (v0 e - v)/tau + sum [A exp((r_ij - d_ij)/B) + k g(r_ij - d_ij)] n_ij
               + sum kappa g(r_ij - d_ij) (Dv . t_ij) t_ij          (pedestrians, walls and the column)
   The social force acts along wall faces only; at the door jambs just the contact force acts.
   The stage is always drawn on IE Tech black; people are coloured by the contact pressure they feel. */
(function () {
  "use strict";

  const C = { bg: "#000024", floor: "#08083F", panel: "#05053A", wall: "#47BFFF", good: "#6DC201", white: "#FFFFFF",
    muted: "#A9B4D6", bad: "#FF5A63", sea: [0x47, 0xbf, 0xff], pale: [0xe8, 0xf6, 0xff], red: [0xff, 0x5a, 0x63] };
  const PMAX = 2500;                                   // N/m: colour scale saturates here
  const LUT = Array.from({ length: 64 }, (_, i) => {   // calm blue -> bright -> red
    const t = i / 63, a = t < 0.5 ? C.sea : C.pale, b = t < 0.5 ? C.pale : C.red, s = t < 0.5 ? t * 2 : (t - 0.5) * 2;
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * s)).join(",")})`;
  });

  const DEFAULTS = {
    n: 150, roomW: 12, roomH: 12, doorW: 1.0, obstacle: false, obsRadius: 0.5, obsGap: 0.8,
    v0: 3.0, mass: 80, rMin: 0.25, rMax: 0.35, tau: 0.5, A: 2000, B: 0.08, k: 1.2e5, kappa: 2.4e5, dt: 0.002,
  };
  const CUT = 0.5;

  function rng(seed) {
    let s = (seed >>> 0) || 1;
    const u = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) + 0.5) / 4294967296; };
    return u;
  }

  /* ------------------------------------------------------------------ physics */
  function createSim(opts, seed) {
    const P = Object.assign({}, DEFAULTS, opts);
    const u = rng(seed);
    const hw = P.roomW / 2, wd = P.doorW / 2;
    const walls = [[-hw, 0, -hw, P.roomH], [hw, 0, hw, P.roomH], [-hw, P.roomH, hw, P.roomH], [-hw, 0, -wd, 0], [wd, 0, hw, 0]];
    const ocx = 0, ocy = P.obsGap + P.obsRadius;
    let px = [], py = [], rad = [];
    const yLow = 1.2 + (P.obstacle ? P.obsGap + 2 * P.obsRadius : 0);
    for (let tries = 0; px.length < P.n && tries < 200000; tries++) {
      const r = P.rMin + (P.rMax - P.rMin) * u();
      const x = -hw + r + 0.05 + (P.roomW - 2 * r - 0.1) * u(), y = yLow + (P.roomH - r - 0.05 - yLow) * u();
      let ok = true;
      for (let j = 0; j < px.length; j++) if (Math.hypot(x - px[j], y - py[j]) <= r + rad[j] + 0.02) { ok = false; break; }
      if (ok) { px.push(x); py.push(y); rad.push(r); }
    }
    const S = {
      P, t: 0, nTotal: px.length, walls, ocx, ocy, exits: [],
      px, py, rad, vx: px.map(() => 0), vy: px.map(() => 0), press: px.map(() => 0), crossed: px.map(() => 0),
    };
    // uniform grid for neighbour search
    const CS = 1.2, OX = 8, OY = 4, GW = Math.ceil(16 / CS) + 2, GH = Math.ceil(18 / CS) + 2;
    const head = new Int32Array(GW * GH);
    let next = new Int32Array(S.px.length);

    const contact = (fx, fy, i, d, rij, nx, ny, dvx, dvy, social) => {
      // force on body i from a body/wall at distance d; returns body-force magnitude (contact part)
      const g = rij - d, ov = g > 0 ? g : 0;
      const fn = social * P.A * Math.exp(Math.min(g, 0.5) / P.B) + P.k * ov;
      const tx = -ny, ty = nx, ft = P.kappa * ov * (dvx * tx + dvy * ty);
      fx[i] += fn * nx + ft * tx; fy[i] += fn * ny + ft * ty;
      return P.k * ov;
    };

    S.step = function () {
      const n = S.px.length;
      if (n === 0) { S.t += P.dt; return; }
      const { px, py, vx, vy, rad } = S;
      const fx = new Float64Array(n), fy = new Float64Array(n), cp = new Float64Array(n);
      for (let i = 0; i < n; i++) {                                 // (1) wish to move
        let ex = -px[i], ey = -1 - py[i];
        if (py[i] < 0) { ex = 0; ey = -1; }
        const l = Math.max(Math.hypot(ex, ey), 1e-9);
        fx[i] = (P.mass * (P.v0 * ex / l - vx[i])) / P.tau;
        fy[i] = (P.mass * (P.v0 * ey / l - vy[i])) / P.tau;
      }
      head.fill(-1);
      if (next.length < n) next = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const c = Math.floor((px[i] + OX) / CS) + GW * Math.floor((py[i] + OY) / CS);
        next[i] = head[c]; head[c] = i;
      }
      for (let i = 0; i < n; i++) {                                 // (2) pedestrians
        const cx = Math.floor((px[i] + OX) / CS), cy = Math.floor((py[i] + OY) / CS);
        for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
          for (let j = head[cx + a + GW * (cy + b)]; j >= 0; j = next[j]) {
            if (j <= i) continue;
            const dx = px[i] - px[j], dy = py[i] - py[j], d = Math.max(Math.hypot(dx, dy), 1e-9), rij = rad[i] + rad[j];
            if (d > rij + CUT) continue;
            const nx = dx / d, ny = dy / d, dvx = vx[j] - vx[i], dvy = vy[j] - vy[i];
            const g = rij - d, ov = g > 0 ? g : 0;
            const fn = P.A * Math.exp(Math.min(g, 0.5) / P.B) + P.k * ov;
            const tx = -ny, ty = nx, ft = P.kappa * ov * (dvx * tx + dvy * ty);
            const gx = fn * nx + ft * tx, gy = fn * ny + ft * ty;
            fx[i] += gx; fy[i] += gy; fx[j] -= gx; fy[j] -= gy;
            cp[i] += P.k * ov; cp[j] += P.k * ov;
          }
        }
      }
      for (let i = 0; i < n; i++) {                                 // (3) walls
        for (let s = 0; s < 5; s++) {
          const [ax, ay, bx, by] = walls[s], abx = bx - ax, aby = by - ay;
          const tt = Math.min(1, Math.max(0, ((px[i] - ax) * abx + (py[i] - ay) * aby) / (abx * abx + aby * aby)));
          const dx = px[i] - (ax + tt * abx), dy = py[i] - (ay + tt * aby), d = Math.max(Math.hypot(dx, dy), 1e-9);
          if (d >= rad[i] + CUT) continue;
          const face = tt > 0 && tt < 1 ? 1 : 0;                    // social force along faces only
          cp[i] += contact(fx, fy, i, d, rad[i], dx / d, dy / d, -vx[i], -vy[i], face);
        }
        if (P.obstacle) {                                           // (4) the column
          const dx = px[i] - S.ocx, dy = py[i] - S.ocy, d = Math.max(Math.hypot(dx, dy), 1e-9);
          if (d < rad[i] + P.obsRadius + CUT) cp[i] += contact(fx, fy, i, d, rad[i] + P.obsRadius, dx / d, dy / d, -vx[i], -vy[i], 1);
        }
      }
      const vmax = Math.max(3, 1.6 * P.v0);                         // (5) integrate
      for (let i = 0; i < n; i++) {
        vx[i] += (P.dt * fx[i]) / P.mass; vy[i] += (P.dt * fy[i]) / P.mass;
        const sp = Math.hypot(vx[i], vy[i]);
        if (sp > vmax) { vx[i] *= vmax / sp; vy[i] *= vmax / sp; }
        px[i] += P.dt * vx[i]; py[i] += P.dt * vy[i];
        S.press[i] = cp[i] / (2 * Math.PI * rad[i]);
      }
      S.t += P.dt;
      for (let i = n - 1; i >= 0; i--) {                            // (6) bookkeeping
        if (py[i] < 0 && !S.crossed[i]) { S.crossed[i] = 1; S.exits.push(S.t); }
        if (py[i] < -2.5) {
          const l = S.px.length - 1;
          for (const k of ["px", "py", "vx", "vy", "rad", "press", "crossed"]) { S[k][i] = S[k][l]; S[k].pop(); }
        }
      }
    };
    S.done = () => S.exits.length >= S.nTotal;
    S.flow = (w = 10) => { const ww = Math.min(w, S.t); return S.exits.filter((t) => t > S.t - ww).length / Math.max(ww, 1e-9); };
    S.doorPressure = () => {
      let s = 0, c = 0;
      for (let i = 0; i < S.px.length; i++) if (Math.hypot(S.px[i], S.py[i]) < 2) { s += S.press[i]; c++; }
      return c ? s / c : 0;
    };
    S.pause = () => (S.exits.length === 0 || S.done() ? 0 : S.t - S.exits[S.exits.length - 1]);
    return S;
  }

  // lets a Node test (or the console) run the physics without the page
  if (typeof window !== "undefined") window.CrowdPhysics = { createSim, DEFAULTS };
  if (typeof module !== "undefined") module.exports = { createSim, DEFAULTS };
  if (typeof VIZ === "undefined") return;

  /* ------------------------------------------------------------------ widget */
  VIZ.register("crowd-exit", {
    title: "Faster is slower: a crowd at a door",
    init(ui) {
      const S = { n: 150, v0: 3, door: 1.0, col: 0, speed: 4, seed: 1, sim: null, paused: false };

      const sV = VIZ.slider(ui.controls, { label: "Desired speed (m/s)", min: 0.5, max: 8, step: 0.1, value: S.v0,
        fmt: (v) => v.toFixed(1), onInput: (v) => { S.v0 = v; if (S.sim) S.sim.P.v0 = v; } });
      const sD = VIZ.slider(ui.controls, { label: "Door width (m)", min: 0.8, max: 2, step: 0.05, value: S.door,
        fmt: (v) => v.toFixed(2), onInput: (v) => { S.door = v; restart(); } });
      const sC = VIZ.slider(ui.controls, { label: "Column distance (m)", min: 0, max: 2.5, step: 0.1, value: S.col,
        fmt: (v) => (v === 0 ? "none" : v.toFixed(1)), onInput: (v) => { S.col = v; restart(); } });
      VIZ.slider(ui.controls, { label: "Playback speed", min: 1, max: 8, step: 1, value: S.speed,
        fmt: (v) => v + "×", onInput: (v) => { S.speed = v; } });
      VIZ.select(ui.controls, { label: "People", value: "150", options: [["100", "100"], ["150", "150"], ["200", "200"]],
        onChange: (v) => { S.n = +v; restart(); } });
      const g = VIZ.group(ui.controls);
      VIZ.button(g, "Calm walk (1.5 m/s)", () => { S.v0 = 1.5; sV.set(1.5); S.sim.P.v0 = 1.5; });
      VIZ.button(g, "Panic (6 m/s)", () => { S.v0 = 6; sV.set(6); S.sim.P.v0 = 6; });
      VIZ.button(g, "Pause / play", () => { S.paused = !S.paused; });
      VIZ.button(g, "Restart", () => { S.seed++; restart(); }, { primary: true });

      const stage = ui.panel({ title: "The room (people coloured by the pressure they feel)", aspect: 0.92, minHeight: 300,
        label: "A crowd of discs pressing towards a door in the bottom wall", onDraw: drawStage });
      const plot = ui.panel({ title: "People evacuated (red bands: pauses longer than 2 s)", aspect: 1, minHeight: 280,
        label: "Number of people evacuated against time", onDraw: drawPlot });

      function restart() {
        S.sim = createSim({ n: S.n, v0: S.v0, doorW: S.door, obstacle: S.col > 0, obsGap: Math.max(S.col, 0.05) }, S.seed);
        redraw();
      }

      function drawStage(c) {
        const ctx = c.ctx, sim = S.sim, P = sim.P;
        const wx0 = -P.roomW / 2 - 1, wx1 = P.roomW / 2 + 1, wy0 = -3.6, wy1 = P.roomH + 0.8;
        const sc = Math.min(c.w / (wx1 - wx0), c.h / (wy1 - wy0));
        const ox = (c.w - sc * (wx1 - wx0)) / 2, oy = (c.h - sc * (wy1 - wy0)) / 2;
        const X = (x) => ox + sc * (x - wx0), Y = (y) => oy + sc * (wy1 - y);
        ctx.fillStyle = C.bg; ctx.fillRect(0, 0, c.w, c.h);
        ctx.fillStyle = C.panel; ctx.fillRect(X(wx0), Y(0), sc * (wx1 - wx0), sc * 3.6);
        ctx.fillStyle = C.floor; ctx.fillRect(X(-P.roomW / 2), Y(P.roomH), sc * P.roomW, sc * P.roomH);
        ctx.strokeStyle = C.wall; ctx.lineWidth = Math.max(3, sc * 0.18); ctx.lineCap = "round";
        sim.walls.forEach((w) => { ctx.beginPath(); ctx.moveTo(X(w[0]), Y(w[1])); ctx.lineTo(X(w[2]), Y(w[3])); ctx.stroke(); });
        ctx.strokeStyle = C.good; ctx.lineWidth = 4; ctx.setLineDash([6, 5]);
        ctx.beginPath(); ctx.moveTo(X(-P.doorW / 2), Y(0)); ctx.lineTo(X(P.doorW / 2), Y(0)); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.good; ctx.font = "700 12px Montserrat, Arial, sans-serif"; ctx.textAlign = "center";
        ctx.fillText("↓  EXIT  ↓", X(0), Y(-0.9));
        if (P.obstacle) {
          ctx.beginPath(); ctx.arc(X(sim.ocx), Y(sim.ocy), sc * P.obsRadius, 0, 2 * Math.PI);
          ctx.fillStyle = C.good; ctx.fill(); ctx.strokeStyle = C.white; ctx.lineWidth = 3; ctx.stroke();
        }
        for (let i = 0; i < sim.px.length; i++) {
          ctx.globalAlpha = sim.py[i] < 0 ? 0.45 : 1;                // people who made it are dimmed
          ctx.fillStyle = LUT[Math.min(63, Math.floor((64 * sim.press[i]) / PMAX))];
          ctx.beginPath(); ctx.arc(X(sim.px[i]), Y(sim.py[i]), sc * sim.rad[i], 0, 2 * Math.PI); ctx.fill();
          ctx.strokeStyle = "rgba(0,0,36,0.8)"; ctx.lineWidth = 0.8; ctx.stroke();
        }
        ctx.globalAlpha = 1;
        const pause = sim.pause();
        ctx.font = "700 13px Montserrat, Arial, sans-serif"; ctx.textAlign = "left"; ctx.fillStyle = pause > 2 ? C.bad : C.white;
        ctx.fillText(sim.done() ? `✓ EVACUATED in ${sim.exits[sim.exits.length - 1].toFixed(0)} s`
          : pause > 2 ? `▲ ARCH AT THE DOOR: ${pause.toFixed(0)} s without anyone passing` : "▶ EVACUATING", 10, 20);
        // colour scale
        const gx = 10, gy = c.h - 22, gw = Math.min(160, c.w * 0.35);
        for (let i = 0; i < gw; i++) { ctx.fillStyle = LUT[Math.floor((63 * i) / gw)]; ctx.fillRect(gx + i, gy, 1, 8); }
        ctx.font = "500 10px Montserrat, Arial, sans-serif"; ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("calm", gx, gy - 4); ctx.textAlign = "right"; ctx.fillText("crushing", gx + gw, gy - 4);
      }

      function drawPlot(c) {
        const sim = S.sim, T = VIZ.theme();
        const ax = VIZ.axes(c, { xlim: [0, Math.max(60, sim.t * 1.05)], ylim: [0, sim.nTotal], xlabel: "time (s)", ylabel: "people out" });
        const ctx = c.ctx, ex = sim.exits;
        if (!ex.length) return;
        ax.clip();
        ctx.fillStyle = T.dark ? "rgba(255,90,99,0.30)" : "rgba(226,0,15,0.22)";
        const all = ex.concat([sim.t]);
        for (let i = 0; i < all.length - 1; i++) if (all[i + 1] - all[i] > 2) ctx.fillRect(ax.sx(all[i]), ax.pad.t, ax.sx(all[i + 1]) - ax.sx(all[i]), ax.ph);
        const xs = [0], ys = [0];
        ex.forEach((t, i) => { xs.push(t, t); ys.push(i, i + 1); });
        xs.push(sim.t); ys.push(ex.length);
        ax.line(xs, ys, { color: T.c[0], width: 2.5 });
        ctx.restore();
      }

      function redraw() { stage.draw(); plot.draw(); readout(); }
      function readout() {
        const s = S.sim;
        ui.readout.textContent = `t = ${s.t.toFixed(1)} s · evacuated ${s.exits.length} / ${s.nTotal} · flow ${s.flow().toFixed(2)} people/s · ` +
          `pressure near the door ${s.doorPressure().toFixed(0)} N/m · pause at the door ${s.pause().toFixed(1)} s`;
      }

      const loop = VIZ.loop(ui.el, (dt) => {
        if (!S.paused && !S.sim.done()) {
          const steps = Math.min(400, Math.round((S.speed * dt) / S.sim.P.dt));
          for (let k = 0; k < steps; k++) S.sim.step();
        }
        redraw();
      });

      restart();
      if (VIZ.reduceMotion) { for (let i = 0; i < 20000 && !S.sim.done(); i++) S.sim.step(); redraw(); }
      else loop.start();
    },
  });
})();

/* Confusion effect: a predator hunts a Couzin school.
   Port of amlab/collective_motion/couzin_predator.py (same rules, same parameters):
   Couzin zones for the prey (repulsion > orientation + attraction, 270° field of view, noise 0.05,
   40°/s turning), an evasion reflex within 4 body lengths, and a predator with four states
   (search, lock, lost, recover). While locked on a target, every look-alike that newly crosses the
   line of sight (±8°, up to 8 body lengths behind the target) breaks the lock; three broken locks waste
   the pass. A strike that reaches its target succeeds with P = 1 / (1 + c n), n = mean number of
   look-alikes in the line of sight during the approach. An odd prey (green) cannot lose its lock.
   The stage is always drawn on IE Tech black so that moving agents are easy to follow. */
(function () {
  "use strict";
  const TWO_PI = 2 * Math.PI, DEG = Math.PI / 180;
  const wrap = (a) => ((((a + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI) - Math.PI;
  const mod2pi = (a) => ((a % TWO_PI) + TWO_PI) % TWO_PI;

  // IE colours (brand sheet): Tech black stage, Sea blue prey, Tech green odd prey, VIG red (lightened for the dark stage) predator
  const C = { bg: "#000024", floor: "#08083F", dot: "#1B1B5C", prey: "#47BFFF", odd: "#6DC201", pred: "#FF5A63", white: "#FFFFFF", muted: "#A9B4D6" };

  const K = {
    dt: 0.1, vPrey: 3, rRep: 1, dOrient: 3, noise: 0.05, halfFov: 135 * DEG, turn: 40 * DEG,
    fleeR: 4, fleeTurn: 120 * DEG, fleeBoost: 1.15,
    vPred: 5, turnPred: 120 * DEG, sight: 22, halfFovPred: 70 * DEG,
    tol: 8 * DEG, clutter: 8, maxBreaks: 3, relock: 3, strikeR: 0.9, recover: 30, maxAttack: 100, burnIn: 300,
  };
  const TRAIL = 12, LIFE = 10;

  VIZ.register("confusion", {
    title: "Confusion effect: a predator hunts a school",
    init(ui) {
      const S = { N: 60, nOdd: 0, dAttract: 10, c: 1, seed: 3, steps: 0 };
      let u = VIZ.rng(3);
      let x, y, th, odd, hist;
      let pred, state, target, breaks, lostSteps, recSteps, attackSteps, coneSum, coneN, prevCross;
      let stats, probHist, effects, msg, msgAge, cam, half, acc = 0;

      const sCoh = VIZ.slider(ui.controls, { label: "Cohesion Δ<i>r</i><sub>a</sub>", min: 0, max: 14, step: 0.5, value: S.dAttract,
        fmt: (v) => v.toFixed(1), onInput: (v) => { S.dAttract = v; } });
      const sConf = VIZ.slider(ui.controls, { label: "Confusion <i>c</i>", min: 0, max: 2, step: 0.1, value: S.c,
        fmt: (v) => v.toFixed(1), onInput: (v) => { S.c = v; } });
      const sOdd = VIZ.slider(ui.controls, { label: "Odd prey", min: 0, max: 3, step: 1, value: 0,
        fmt: (v) => String(v), onInput: (v) => { S.nOdd = v; reset(); } });
      VIZ.select(ui.controls, { label: "School size", value: "60", options: [["30", "30"], ["60", "60"], ["100", "100"]],
        onChange: (v) => { S.N = +v; reset(); } });
      const g = VIZ.group(ui.controls);
      const preset = (coh) => () => { S.dAttract = coh; sCoh.set(coh); resetStats(); };
      VIZ.button(g, "Dispersed (Δr_a = 0)", preset(0));
      VIZ.button(g, "Bait ball (Δr_a = 10)", preset(10));
      VIZ.button(g, "New hunt", () => { S.seed++; reset(); }, { primary: true });

      const stage = ui.panel({ title: "The hunt (the camera follows the school)", aspect: 1, minHeight: 280,
        label: "A school of fish drawn as arrows, hunted by a red predator", onDraw: drawStage });
      const plot = ui.panel({ title: "Running capture probability", aspect: 1, minHeight: 280,
        label: "Capture probability after each attack", onDraw: drawPlot });

      function resetStats() {
        stats = { attacks: 0, captures: 0, escapes: 0, aborts: 0, timeouts: 0 };
        probHist = [];
      }

      function reset() {
        u = VIZ.rng(S.seed * 101 + S.N);
        x = new Float64Array(S.N); y = new Float64Array(S.N); th = new Float64Array(S.N);
        odd = new Uint8Array(S.N);
        for (let i = 0; i < S.N; i++) { x[i] = 25 * u(); y[i] = 25 * u(); th[i] = TWO_PI * u(); odd[i] = i < S.nOdd ? 1 : 0; }
        resetStats();
        effects = []; msg = ""; msgAge = 0; S.steps = 0;
        state = "search"; target = -1; breaks = 0; lostSteps = recSteps = attackSteps = 0; coneSum = coneN = 0;
        prevCross = new Uint8Array(S.N);
        pred = { x: 1e4, y: 1e4, th: 0 };
        hunting = false;
        for (let k = 0; k < K.burnIn; k++) stepPrey();
        const c = centroid(), a = TWO_PI * u();
        pred = { x: c[0] + 35 * Math.cos(a), y: c[1] + 35 * Math.sin(a), th: mod2pi(a + Math.PI) };
        hunting = true;
        hist = [];
        cam = c.slice(); half = 30;
        redraw();
      }
      let hunting = false;

      const centroid = () => {
        let cx = 0, cy = 0;
        for (let i = 0; i < S.N; i++) { cx += x[i]; cy += y[i]; }
        return [cx / S.N, cy / S.N];
      };

      // ---------------- prey (Couzin zones) ----------------
      function stepPrey() {
        const N = S.N, rO = K.rRep + K.dOrient, rA = rO + S.dAttract, maxTurn = K.turn * K.dt;
        const nt = new Float64Array(N), speed = new Float64Array(N).fill(K.vPrey);
        for (let i = 0; i < N; i++) {
          const hi = th[i];
          let rx = 0, ry = 0, nRep = 0, ox = 0, oy = 0, ax = 0, ay = 0;
          for (let j = 0; j < N; j++) {
            if (j === i) continue;
            const dx = x[j] - x[i], dy = y[j] - y[i], d = Math.hypot(dx, dy);
            if (d > rA || Math.abs(wrap(Math.atan2(dy, dx) - hi)) > K.halfFov) continue;
            const ux = d > 0 ? dx / d : 0, uy = d > 0 ? dy / d : 0;
            if (d < K.rRep) { rx -= ux; ry -= uy; nRep++; }
            else if (d < rO) { ox += Math.cos(th[j]); oy += Math.sin(th[j]); }
            else { ax += ux; ay += uy; }
          }
          let vx = nRep ? rx : ox + ax, vy = nRep ? ry : oy + ay;
          if (Math.hypot(vx, vy) < 1e-8) { vx = Math.cos(hi); vy = Math.sin(hi); }
          const desired = mod2pi(Math.atan2(vy, vx) + K.noise * u.normal());
          let dh = wrap(desired - hi);
          dh = Math.max(-maxTurn, Math.min(maxTurn, dh));
          nt[i] = mod2pi(hi + dh);
        }
        if (hunting && K.fleeR > 0) {            // evasion reflex
          const lim = K.fleeTurn * K.dt;
          for (let i = 0; i < N; i++) {
            const dx = x[i] - pred.x, dy = y[i] - pred.y;
            if (Math.hypot(dx, dy) < K.fleeR) {
              const dh = Math.max(-lim, Math.min(lim, wrap(Math.atan2(dy, dx) - th[i])));
              nt[i] = mod2pi(th[i] + dh);
              speed[i] *= K.fleeBoost;
            }
          }
        }
        for (let i = 0; i < N; i++) {
          th[i] = nt[i];
          x[i] += K.dt * speed[i] * Math.cos(nt[i]);
          y[i] += K.dt * speed[i] * Math.sin(nt[i]);
        }
      }

      // ---------------- predator ----------------
      function turnMove(desired, v) {
        const lim = K.turnPred * K.dt;
        pred.th = mod2pi(pred.th + Math.max(-lim, Math.min(lim, wrap(desired - pred.th))));
        pred.x += K.dt * v * Math.cos(pred.th);
        pred.y += K.dt * v * Math.sin(pred.th);
      }
      function log(kind, ex, ey, text) { effects.push({ kind, x: ex, y: ey, age: 0 }); msg = text; msgAge = 0; }
      function endAttack(outcome) {
        if (outcome === "capture") stats.captures++;
        else if (outcome === "escape") stats.escapes++;
        else if (outcome === "abort") stats.aborts++;
        else stats.timeouts++;
        const done = stats.attacks - 0;
        probHist.push(stats.captures / Math.max(1, done));
        target = -1; state = "recover"; recSteps = K.recover;
      }
      function selectTarget(vis, dist) {
        const N = S.N;
        let best = -1, bestScore = -Infinity;
        for (let i = 0; i < N; i++) if (vis[i] && odd[i]) { if (-dist[i] > bestScore) { bestScore = -dist[i]; best = i; } }
        if (best >= 0) return best;
        for (let i = 0; i < N; i++) {
          if (!vis[i]) continue;
          let nn = Infinity;
          for (let j = 0; j < N; j++) if (j !== i) nn = Math.min(nn, Math.hypot(x[i] - x[j], y[i] - y[j]));
          const score = nn - 0.15 * dist[i];
          if (score > bestScore) { bestScore = score; best = i; }
        }
        return best;
      }
      function stepPredator() {
        const N = S.N, dist = new Float64Array(N), bear = new Float64Array(N), vis = new Uint8Array(N);
        let anyVis = false, nearest = 0, dmin = Infinity;
        for (let i = 0; i < N; i++) {
          const dx = x[i] - pred.x, dy = y[i] - pred.y;
          dist[i] = Math.hypot(dx, dy); bear[i] = Math.atan2(dy, dx);
          if (dist[i] < dmin) { dmin = dist[i]; nearest = i; }
          if (dist[i] < K.sight && Math.abs(wrap(bear[i] - pred.th)) < K.halfFovPred) { vis[i] = 1; anyVis = true; }
        }
        if (state === "search") {
          if (anyVis) {
            target = selectTarget(vis, dist); state = "lock"; breaks = 0; attackSteps = 0; coneSum = coneN = 0;
            prevCross.fill(0); stats.attacks++;
          }
          turnMove(bear[nearest], K.vPred);
        } else if (state === "lock") {
          const t = target; attackSteps++;
          const oddT = !!odd[t];
          if (!oddT) {
            let nNew = 0, nCross = 0;
            for (let i = 0; i < N; i++) {
              const c = i !== t && Math.abs(wrap(bear[i] - bear[t])) < K.tol && dist[i] < dist[t] + K.clutter;
              if (c) { nCross++; if (!prevCross[i]) nNew++; }
              prevCross[i] = c ? 1 : 0;
            }
            coneSum += nCross; coneN++;
            if (nNew > 0) {
              breaks++;
              log("break", x[t], y[t], "A look-alike crossed the line of sight: lock broken");
              if (breaks >= K.maxBreaks) { log("abort", pred.x, pred.y, "Three broken locks: the pass is wasted"); endAttack("abort"); }
              else { state = "lost"; lostSteps = K.relock; }
              turnMove(pred.th, K.vPred); return;
            }
          }
          if (dist[t] < K.strikeR) {
            const p = oddT ? 1 : 1 / (1 + S.c * (coneSum / Math.max(1, coneN)));
            if (u() < p) {
              log("capture", x[t], y[t], "Strike succeeded: prey captured");
              const j = (t + 1 + Math.floor(u() * (N - 1))) % N;           // respawn next to a random member
              x[t] = x[j] + 1.5 * u.normal(); y[t] = y[j] + 1.5 * u.normal(); th[t] = th[j];
              endAttack("capture");
            } else { log("escape", x[t], y[t], "Strike missed: the prey escaped"); endAttack("escape"); }
            turnMove(pred.th, K.vPred); return;
          }
          if (attackSteps > K.maxAttack) { endAttack("timeout"); turnMove(pred.th, K.vPred); return; }
          turnMove(bear[t], K.vPred);
        } else if (state === "lost") {
          if (--lostSteps <= 0) {
            if (anyVis) { target = selectTarget(vis, dist); prevCross.fill(0); state = "lock"; }
            else endAttack("abort");
          }
          turnMove(pred.th, K.vPred);
        } else {
          if (--recSteps <= 0) state = "search";
          turnMove(pred.th, K.vPred);
        }
      }

      function step() {
        if (hunting) stepPredator();
        stepPrey();
        S.steps++;
        hist.push([Float64Array.from(x), Float64Array.from(y)]);
        if (hist.length > TRAIL) hist.shift();
      }

      // ---------------- drawing ----------------
      function drawStage(c) {
        const ctx = c.ctx, N = S.N;
        ctx.fillStyle = C.floor; ctx.fillRect(0, 0, c.w, c.h);
        // camera
        const cen = centroid();
        const rs = [];
        for (let i = 0; i < N; i++) rs.push(Math.hypot(x[i] - cen[0], y[i] - cen[1]));
        rs.sort((a, b) => a - b);
        let hT = Math.max(24, 1.35 * rs[Math.floor(0.92 * (N - 1))] + 9);
        const d = Math.max(Math.abs(pred.x - cen[0]), Math.abs(pred.y - cen[1]));
        if (d < 60) hT = Math.max(hT, 1.12 * d + 4);
        cam[0] += 0.12 * (cen[0] - cam[0]); cam[1] += 0.12 * (cen[1] - cam[1]); half += 0.08 * (hT - half);
        const sc = Math.min(c.w, c.h) / (2 * half);
        const X = (wx) => c.w / 2 + sc * (wx - cam[0]), Y = (wy) => c.h / 2 - sc * (wy - cam[1]);
        const body = Math.max(0.9, half / 17) * sc;
        // motion-cue dots
        const gs = half < 32 ? 5 : 10;
        ctx.fillStyle = C.dot;
        for (let gx = Math.floor((cam[0] - half) / gs) * gs; gx < cam[0] + half + gs; gx += gs)
          for (let gy = Math.floor((cam[1] - half) / gs) * gs; gy < cam[1] + half + gs; gy += gs) ctx.fillRect(X(gx) - 1, Y(gy) - 1, 2, 2);
        // predator field of view
        ctx.fillStyle = "rgba(255,90,99,0.10)";
        ctx.beginPath(); ctx.moveTo(X(pred.x), Y(pred.y));
        ctx.arc(X(pred.x), Y(pred.y), K.sight * sc, -pred.th - K.halfFovPred, -pred.th + K.halfFovPred);
        ctx.closePath(); ctx.fill();
        // trails
        ctx.lineWidth = 1.6; ctx.lineCap = "round";
        for (let k = 1; k < hist.length; k++) {
          ctx.globalAlpha = 0.6 * k / hist.length;
          for (let i = 0; i < N; i++) {
            ctx.strokeStyle = odd[i] ? C.odd : C.prey;
            ctx.beginPath(); ctx.moveTo(X(hist[k - 1][0][i]), Y(hist[k - 1][1][i])); ctx.lineTo(X(hist[k][0][i]), Y(hist[k][1][i])); ctx.stroke();
          }
        }
        ctx.globalAlpha = 1;
        // prey: arrows (an outline, not colour alone, marks the odd ones)
        const arrow = (px, py, a, s, fill, stroke, lw) => {
          const cs = Math.cos(a), sn = -Math.sin(a);
          ctx.beginPath();
          ctx.moveTo(px + s * cs, py + s * sn);
          ctx.lineTo(px - 0.65 * s * cs - 0.45 * s * sn, py - 0.65 * s * sn + 0.45 * s * cs);
          ctx.lineTo(px - 0.65 * s * cs + 0.45 * s * sn, py - 0.65 * s * sn - 0.45 * s * cs);
          ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
          ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke();
        };
        for (let i = 0; i < N; i++) arrow(X(x[i]), Y(y[i]), th[i], body, odd[i] ? C.odd : C.prey, odd[i] ? C.white : C.bg, odd[i] ? 1.8 : 0.6);
        // lock line and ring
        if ((state === "lock" || state === "lost") && target >= 0) {
          ctx.strokeStyle = C.pred; ctx.lineWidth = 1.6; ctx.globalAlpha = state === "lock" ? 1 : 0.3; ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(X(pred.x), Y(pred.y)); ctx.lineTo(X(x[target]), Y(y[target])); ctx.stroke();
          ctx.setLineDash([]); ctx.globalAlpha = 1;
          if (state === "lock") { ctx.strokeStyle = C.white; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(x[target]), Y(y[target]), body * 1.8, 0, TWO_PI); ctx.stroke(); }
        }
        // predator: large arrow with a white outline
        arrow(X(pred.x), Y(pred.y), pred.th, body * 1.9, C.pred, C.white, 1.8);
        // event rings
        effects = effects.filter((e) => e.age <= LIFE);
        effects.forEach((e) => {
          const big = e.kind === "capture" || e.kind === "abort";
          ctx.globalAlpha = 1 - e.age / LIFE;
          ctx.strokeStyle = { break: C.white, capture: C.pred, escape: C.odd, abort: C.muted }[e.kind]; ctx.lineWidth = 2.2;
          ctx.beginPath(); ctx.arc(X(e.x), Y(e.y), body * (1.5 + 0.9 * e.age) * (big ? 1.8 : 1), 0, TWO_PI); ctx.stroke();
          e.age++;
        });
        ctx.globalAlpha = 1;
        // status line (text + symbol, never colour alone)
        const label = { search: "○ SEARCHING", lock: "◎ LOCKED ON TARGET", lost: "✕ LOCK BROKEN", recover: "↻ RECOVERING" }[state];
        ctx.font = "700 13px Montserrat, Arial, sans-serif"; ctx.textAlign = "left"; ctx.fillStyle = C.white;
        ctx.fillText(label + (state === "lock" || state === "lost" ? `   broken locks ${breaks} / ${K.maxBreaks}` : ""), 10, 20);
        ctx.font = "500 11px Montserrat, Arial, sans-serif"; ctx.fillStyle = C.muted;
        if (msgAge < 60) ctx.fillText(msg, 10, 38);
        msgAge++;
      }

      function drawPlot(c) {
        const n = Math.max(10, probHist.length);
        const ax = VIZ.axes(c, { xlim: [1, n], ylim: [0, 1], xlabel: "attack number", ylabel: "P(capture)" });
        const T = ax.T;
        if (probHist.length) {
          ax.clip();
          ax.line(probHist.map((_, i) => i + 1), probHist, { color: T.c[0], width: 2.5 });
          c.ctx.restore();
          const last = probHist[probHist.length - 1];
          ax.dot(probHist.length, last, 4, T.c[0], T.panel);
        }
      }

      function redraw() { stage.draw(); plot.draw(); readout(); }
      function readout() {
        const done = stats.attacks - (state === "lock" || state === "lost" ? 1 : 0);
        const pc = done > 0 ? Math.round((100 * stats.captures) / done) + " %" : "n/a";
        ui.readout.textContent = `attacks ${done} · captured ${stats.captures} · passes wasted ${stats.aborts + stats.timeouts} · ` +
          `strikes that missed ${stats.escapes} · capture probability ${pc} · school ${S.N}, cohesion Δr_a = ${S.dAttract.toFixed(1)}, c = ${S.c.toFixed(1)}`;
      }

      const loop = VIZ.loop(ui.el, (dt) => {
        acc += dt * 30;                        // 30 simulation steps (3 s of hunt) per second
        let k = 0;
        while (acc >= 1 && k < 4) { step(); acc -= 1; k++; }
        acc = Math.min(acc, 2);
        redraw();
      });

      reset();
      if (VIZ.reduceMotion) { for (let i = 0; i < 600; i++) step(); redraw(); }
      else loop.start();
    },
  });
})();

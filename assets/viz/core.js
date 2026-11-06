/* ==========================================================================
   Interactive figures: shared helpers (window.VIZ)
   --------------------------------------------------------------------------
   A widget lives in assets/viz/widgets/NAME.js and registers itself:

     VIZ.register("NAME", { title: "…", init(ui, el) { … } });

   `ui` gives you: ui.controls (row for sliders/buttons), ui.panel(opts)
   (adds a canvas panel), ui.readout (live text), plus helpers below.
   Colours come from the --viz-* CSS variables of the active theme, and every
   widget is redrawn automatically when the reader toggles light/dark mode.
   ========================================================================== */
(function () {
  "use strict";

  const VIZ = { widgets: {} };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  VIZ.reduceMotion = reduceMotion;

  /* ---------------- registry ---------------- */
  VIZ.register = function (name, def) { VIZ.widgets[name] = def; };

  VIZ.mount = function (el) {
    const def = VIZ.widgets[el.dataset.viz];
    if (!def || el.dataset.vizMounted) return;
    el.dataset.vizMounted = "1";
    if (def.bare) {             // decorative figures: no frame, controls or guide
      try { def.init({ el }); } catch (e) { console.error("[viz] " + el.dataset.viz, e); }
      return;
    }
    el.classList.add("viz");

    // Whatever the author wrote inside the div becomes the "Try this" guide.
    const guideNodes = Array.from(el.childNodes);
    el.textContent = "";

    const head = h("div", "viz__head");
    head.append(h("span", "viz__badge", "Interactive"), h("span", "viz__title", def.title || ""));
    const controls = h("div", "viz__controls");
    const panels = h("div", "viz__panels");
    const readout = h("p", "viz__readout");
    readout.setAttribute("aria-live", "polite");
    el.append(head, controls, panels, readout);

    if (guideNodes.some((n) => n.nodeType === 1 || (n.textContent || "").trim())) {
      const guide = h("div", "viz__guide");
      guide.append(h("p", "viz__guide-title", "Try this"));
      guideNodes.forEach((n) => guide.appendChild(n));
      el.append(guide);
    }

    const redrawers = [];
    const ui = {
      el, controls, panels, readout,
      panel(opts) {
        const p = VIZ.panel(panels, opts);
        redrawers.push(() => p.resize());
        return p;
      },
      onRedraw(fn) { redrawers.push(fn); },
      redraw() { redrawers.forEach((f) => f()); },
    };
    try {
      def.init(ui, el);
    } catch (e) {
      console.error("[viz] " + el.dataset.viz, e);
      readout.textContent = "This figure could not start in your browser.";
    }
    VIZ.onTheme(() => ui.redraw());
  };

  /* ---------------- DOM helpers ---------------- */
  function h(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  VIZ.h = h;
  let uid = 0;

  VIZ.slider = function (parent, o) {
    const id = "viz-s" + (++uid);
    const wrap = h("div", "viz__slider");
    const lab = h("label", null);
    lab.htmlFor = id;
    lab.innerHTML = o.label;
    const input = h("input");
    Object.assign(input, { type: "range", id, min: o.min, max: o.max, step: o.step, value: o.value });
    const out = h("output", "viz__value");
    out.htmlFor = id;
    const fmt = o.fmt || ((v) => String(v));
    const sync = () => { out.textContent = fmt(+input.value); };
    input.addEventListener("input", () => { sync(); o.onInput && o.onInput(+input.value); });
    sync();
    wrap.append(lab, input, out);
    parent.appendChild(wrap);
    return {
      input,
      get: () => +input.value,
      set(v, fire) { input.value = v; sync(); if (fire && o.onInput) o.onInput(+input.value); },
      disable(b) { input.disabled = b; wrap.classList.toggle("is-disabled", b); },
    };
  };

  VIZ.button = function (parent, label, onClick, opts) {
    const b = h("button", "viz__btn" + (opts && opts.primary ? " viz__btn--primary" : ""), label);
    b.type = "button";
    b.addEventListener("click", onClick);
    parent.appendChild(b);
    return b;
  };

  VIZ.select = function (parent, o) {
    const id = "viz-sel" + (++uid);
    const wrap = h("div", "viz__select");
    const lab = h("label", null, o.label);
    lab.htmlFor = id;
    const sel = h("select");
    sel.id = id;
    o.options.forEach(([value, text]) => {
      const op = h("option", null, text);
      op.value = value;
      sel.appendChild(op);
    });
    sel.value = o.value;
    sel.addEventListener("change", () => o.onChange && o.onChange(sel.value));
    wrap.append(lab, sel);
    parent.appendChild(wrap);
    return { get: () => sel.value, set: (v) => { sel.value = v; } };
  };

  VIZ.group = function (parent, cls) {
    const g = h("div", "viz__group" + (cls ? " " + cls : ""));
    parent.appendChild(g);
    return g;
  };

  /* ---------------- theme ---------------- */
  VIZ.theme = function () {
    const css = getComputedStyle(document.body);
    const col = (n) => css.getPropertyValue(n).trim();
    return {
      c: [col("--viz-c1"), col("--viz-c2"), col("--viz-c3"), col("--viz-c4")],
      grid: col("--viz-grid"), axis: col("--viz-axis"), text: col("--viz-text"),
      bg: col("--viz-bg"), panel: col("--viz-panel"),
      stable: col("--viz-stable"), unstable: col("--viz-unstable"), accent: col("--viz-accent"),
      dark: document.body.classList.contains("quarto-dark"),
      font: (size = 12, weight = 500) => `${weight} ${size}px Montserrat, Arial, sans-serif`,
    };
  };

  const themeListeners = [];
  VIZ.onTheme = function (fn) { themeListeners.push(fn); };
  new MutationObserver(() => requestAnimationFrame(() => themeListeners.forEach((f) => f())))
    .observe(document.body, { attributes: true, attributeFilter: ["class"] });

  /* ---------------- canvas panels ---------------- */
  /* opts: { title, aspect (w/h), height, minHeight, onDraw(c), onPointer(evt, c) } */
  VIZ.panel = function (parent, opts) {
    const fig = h("div", "viz__panel" + (opts.cls ? " " + opts.cls : ""));
    if (opts.title) fig.appendChild(h("p", "viz__panel-title", opts.title));
    const canvas = h("canvas");
    if (opts.label) { canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", opts.label); }
    fig.appendChild(canvas);
    parent.appendChild(fig);

    const c = { canvas, fig, ctx: canvas.getContext("2d"), w: 0, h: 0, frame: null };
    c.resize = function () {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth || fig.clientWidth;
      const hgt = opts.height || Math.max(opts.minHeight || 180, Math.round(w / (opts.aspect || 1.6)));
      canvas.style.height = hgt + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(hgt * dpr);
      c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.w = w; c.h = hgt;
      c.draw();
    };
    c.draw = function () {
      if (!c.w) return;
      c.ctx.clearRect(0, 0, c.w, c.h);
      opts.onDraw && opts.onDraw(c);
    };
    if (opts.onPointer) {
      canvas.classList.add("is-clickable");
      const handler = (e) => {
        const r = canvas.getBoundingClientRect();
        opts.onPointer({ type: e.type, x: e.clientX - r.left, y: e.clientY - r.top, buttons: e.buttons }, c);
      };
      canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); handler(e); });
      canvas.addEventListener("pointermove", (e) => { if (e.buttons) handler(e); });
    }
    new ResizeObserver(() => c.resize()).observe(fig);
    return c;
  };

  /* ---------------- axes ----------------
     opts: { xlim, ylim, xticks, yticks, xlabel, ylabel, pad, fmtX, fmtY, square } */
  VIZ.axes = function (c, opts) {
    const { ctx, w, h: H } = c;
    const T = VIZ.theme();
    const pad = Object.assign({ l: 46, r: 12, t: 10, b: opts.xlabel ? 40 : 24 }, opts.pad || {});
    let [x0, x1] = opts.xlim, [y0, y1] = opts.ylim;
    const pw = w - pad.l - pad.r, ph = H - pad.t - pad.b;
    const sx = (x) => pad.l + ((x - x0) / (x1 - x0)) * pw;
    const sy = (y) => pad.t + ((y1 - y) / (y1 - y0)) * ph;
    const ix = (px) => x0 + ((px - pad.l) / pw) * (x1 - x0);
    const iy = (py) => y1 - ((py - pad.t) / ph) * (y1 - y0);

    ctx.save();
    ctx.fillStyle = T.panel;
    ctx.fillRect(pad.l, pad.t, pw, ph);
    ctx.lineWidth = 1;
    ctx.font = T.font(11);
    ctx.fillStyle = T.text;
    const xt = opts.xticks || VIZ.ticks(x0, x1, Math.max(3, Math.round(pw / 70)));
    const yt = opts.yticks || VIZ.ticks(y0, y1, Math.max(3, Math.round(ph / 45)));
    const fx = opts.fmtX || VIZ.fmt, fy = opts.fmtY || VIZ.fmt;
    ctx.textAlign = "center";
    xt.forEach((v) => {
      ctx.strokeStyle = T.grid;
      ctx.beginPath(); ctx.moveTo(sx(v) + 0.5, pad.t); ctx.lineTo(sx(v) + 0.5, pad.t + ph); ctx.stroke();
      ctx.fillText(fx(v), sx(v), pad.t + ph + 15);
    });
    ctx.textAlign = "right";
    yt.forEach((v) => {
      ctx.strokeStyle = T.grid;
      ctx.beginPath(); ctx.moveTo(pad.l, sy(v) + 0.5); ctx.lineTo(pad.l + pw, sy(v) + 0.5); ctx.stroke();
      ctx.fillText(fy(v), pad.l - 6, sy(v) + 4);
    });
    ctx.strokeStyle = T.axis;
    ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, pw - 1, ph - 1);
    ctx.font = T.font(12, 600);
    if (opts.xlabel) { ctx.textAlign = "center"; ctx.fillText(opts.xlabel, pad.l + pw / 2, H - 6); }
    if (opts.ylabel) {
      ctx.translate(13, pad.t + ph / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center"; ctx.fillText(opts.ylabel, 0, 0);
    }
    ctx.restore();

    const ax = { sx, sy, ix, iy, pad, pw, ph, xlim: [x0, x1], ylim: [y0, y1], T };
    ax.clip = function () {
      ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, pw, ph); ctx.clip();
    };
    ax.inside = (px, py) => px >= pad.l && px <= pad.l + pw && py >= pad.t && py <= pad.t + ph;
    ax.line = function (xs, ys, style) {
      ctx.save();
      ctx.strokeStyle = style.color; ctx.lineWidth = style.width || 2;
      ctx.globalAlpha = style.alpha == null ? 1 : style.alpha;
      if (style.dash) ctx.setLineDash(style.dash);
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < xs.length; i++) {
        if (!isFinite(xs[i]) || !isFinite(ys[i])) { pen = false; continue; }
        const X = sx(xs[i]), Y = sy(ys[i]);
        if (pen) ctx.lineTo(X, Y); else { ctx.moveTo(X, Y); pen = true; }
      }
      ctx.stroke(); ctx.restore();
    };
    ax.dot = function (x, y, r, fill, stroke) {
      ctx.save(); ctx.beginPath(); ctx.arc(sx(x), sy(y), r, 0, 2 * Math.PI);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
      ctx.restore();
    };
    /* fixed point marker: stable = filled, unstable = hollow (+ ring) */
    ax.fixed = function (x, y, stable) {
      ax.dot(x, y, 6, stable ? T.stable : T.panel, stable ? T.panel : T.unstable);
    };
    ax.label = function (text, x, y, o) {
      o = o || {};
      ctx.save(); ctx.font = T.font(o.size || 11, o.weight || 600);
      ctx.fillStyle = o.color || T.text; ctx.textAlign = o.align || "left";
      ctx.fillText(text, sx(x) + (o.dx || 0), sy(y) + (o.dy || 0)); ctx.restore();
    };
    return ax;
  };

  VIZ.legend = function (c, items, x, y) {
    const { ctx } = c; const T = VIZ.theme();
    ctx.save(); ctx.font = T.font(11, 600); ctx.textAlign = "left";
    items.forEach((it, i) => {
      const yy = y + i * 16;
      ctx.strokeStyle = it.color; ctx.lineWidth = it.width || 2.5;
      if (it.dash) ctx.setLineDash(it.dash); else ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(x, yy); ctx.lineTo(x + 18, yy); ctx.stroke();
      ctx.fillStyle = T.text; ctx.fillText(it.label, x + 24, yy + 4);
    });
    ctx.restore();
  };

  /* ---------------- numbers ---------------- */
  VIZ.ticks = function (a, b, n) {
    const span = b - a; if (span <= 0) return [a];
    const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9 * span; v += step) out.push(+v.toFixed(10));
    return out;
  };
  VIZ.fmt = function (v) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 100 || Number.isInteger(v)) return String(Math.round(v * 100) / 100);
    return v.toFixed(a < 1 ? 2 : 1).replace(/\.?0+$/, "");
  };
  VIZ.linspace = function (a, b, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = a + ((b - a) * i) / (n - 1);
    return out;
  };
  /* All roots of a scalar function on [a, b]: sign changes on a fine grid + bisection */
  VIZ.roots = function (f, a, b, n) {
    n = n || 2000;
    const out = [];
    let xp = a, fp = f(a);
    for (let i = 1; i <= n; i++) {
      const x = a + ((b - a) * i) / n, fx = f(x);
      if (fp === 0) out.push(xp);
      else if (fp * fx < 0) {
        let lo = xp, hi = x, flo = fp;
        for (let k = 0; k < 60; k++) {
          const mid = 0.5 * (lo + hi), fm = f(mid);
          if (flo * fm <= 0) hi = mid; else { lo = mid; flo = fm; }
        }
        out.push(0.5 * (lo + hi));
      }
      xp = x; fp = fx;
    }
    return out;
  };
  /* One classical Runge-Kutta step for y' = f(t, y), y an array */
  VIZ.rk4 = function (f, t, y, dt) {
    const n = y.length, tmp = new Array(n);
    const k1 = f(t, y);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + 0.5 * dt * k1[i];
    const k2 = f(t + 0.5 * dt, tmp);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + 0.5 * dt * k2[i];
    const k3 = f(t + 0.5 * dt, tmp);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + dt * k3[i];
    const k4 = f(t + dt, tmp);
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = y[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    return out;
  };
  /* Seeded RNG so a "new sample" is reproducible within a session */
  VIZ.rng = function (seed) {
    let s = seed >>> 0 || 1;
    const u = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) + 0.5) / 4294967296; };
    u.normal = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
    return u;
  };

  /* ---------------- animation loop ----------------
     Runs fn(dtSeconds) each frame while the figure is on screen and the tab
     is visible. With reduced motion, `step` still works (call it yourself). */
  VIZ.loop = function (el, fn) {
    let running = false, visible = true, raf = 0, last = 0;
    const tick = (now) => {
      raf = 0;
      if (!running || !visible || document.hidden) return;
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
      last = now;
      fn(dt);
      raf = requestAnimationFrame(tick);
    };
    const kick = () => { if (running && visible && !raf) { last = 0; raf = requestAnimationFrame(tick); } };
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; kick(); }).observe(el);
    document.addEventListener("visibilitychange", kick);
    return {
      start() { running = true; kick(); },
      stop() { running = false; },
      get running() { return running; },
    };
  };

  window.VIZ = VIZ;
})();

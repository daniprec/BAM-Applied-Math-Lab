/* Cover art: a ring of Kuramoto oscillators that slowly falls into sync and
   back out (K varies in time). Decorative only; respects reduced motion. */
(function () {
  "use strict";
  VIZ.register("hero-sync", {
    bare: true,
    init({ el }) {
      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-hidden", "true");
      canvas.style.display = "block";
      el.appendChild(canvas);
      const ctx = canvas.getContext("2d");
      const u = VIZ.rng(42), N = 140;
      const w = Array.from({ length: N }, () => 0.55 * u.normal());
      const th = Array.from({ length: N }, () => 2 * Math.PI * u());
      let t = 0, W = 0, H = 0;

      function resize() {
        const dpr = window.devicePixelRatio || 1;
        W = el.clientWidth; H = W;            // square; never read the height back (avoids a resize loop)
        canvas.width = W * dpr; canvas.height = H * dpr;
        canvas.style.width = W + "px"; canvas.style.height = H + "px";
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw();
      }
      function step(dt) {
        const K = 1.6 + 1.5 * Math.sin((2 * Math.PI * t) / 26);
        let c = 0, s = 0;
        for (let i = 0; i < N; i++) { c += Math.cos(th[i]); s += Math.sin(th[i]); }
        const r = Math.hypot(c, s) / N, psi = Math.atan2(s, c);
        for (let i = 0; i < N; i++) th[i] += dt * (w[i] + K * r * Math.sin(psi - th[i]));
        t += dt;
      }
      function draw() {
        const T = VIZ.theme();
        ctx.clearRect(0, 0, W, H);
        const cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.36;
        ctx.strokeStyle = T.accent; ctx.globalAlpha = 0.35; ctx.lineWidth = 1.5;
        [1, 0.72, 0.44].forEach((f) => { ctx.beginPath(); ctx.arc(cx, cy, R * f, 0, 2 * Math.PI); ctx.stroke(); });
        let c = 0, s = 0;
        for (let i = 0; i < N; i++) {
          c += Math.cos(th[i]); s += Math.sin(th[i]);
          const ring = [1, 0.72, 0.44][i % 3];
          ctx.globalAlpha = 0.9;
          ctx.fillStyle = i % 3 === 0 ? T.c[0] : i % 3 === 1 ? T.accent : T.stable;
          ctx.beginPath(); ctx.arc(cx + R * ring * Math.cos(th[i]), cy - R * ring * Math.sin(th[i]), 4, 0, 2 * Math.PI); ctx.fill();
        }
        const r = Math.hypot(c, s) / N, psi = Math.atan2(s, c);
        ctx.globalAlpha = 1; ctx.strokeStyle = T.stable; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * r * Math.cos(psi), cy - R * r * Math.sin(psi)); ctx.stroke();
        ctx.fillStyle = T.stable; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI); ctx.fill();
      }
      new ResizeObserver(resize).observe(el);
      VIZ.onTheme(draw);
      if (VIZ.reduceMotion) { for (let i = 0; i < 3000; i++) step(0.01); t = 6; draw(); return; }
      const loop = VIZ.loop(el, (dt) => { for (let i = 0; i < 4; i++) step(dt); draw(); });
      loop.start();
    },
  });
})();

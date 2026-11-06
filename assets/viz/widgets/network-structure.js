/* Network structure: Zachary's karate club.
   Data generated with networkx 3.6.1 (nx.karate_club_graph(); layout nx.spring_layout(G, seed=3);
   communities: greedy_modularity_communities; centralities: degree, betweenness, closeness,
   eigenvector; clustering: nx.clustering). Values match what students compute in the lab. */
(function () {
  "use strict";
  const K = {"edges":[[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,17],[0,19],[0,21],[0,31],[1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],[3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],[14,32],[14,33],[15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],[23,25],[23,27],[23,29],[23,32],[23,33],[24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]],"pos":[[0.1866,0.3184],[0.2349,0.0919],[0.0378,0.0642],[0.2442,0.2468],[0.1257,0.6209],[0.1802,0.7622],[0.2742,0.7353],[0.2934,0.1601],[-0.0023,-0.1168],[-0.3673,0.0383],[0.0272,0.7106],[0.5005,0.5618],[0.4065,0.4759],[0.1209,0.0309],[-0.4523,-0.4971],[-0.029,-0.5333],[0.2674,1.0],[0.5992,0.2455],[0.0558,-0.6779],[0.3115,-0.0535],[-0.3194,-0.7033],[0.4874,0.286],[-0.3489,-0.567],[-0.3479,-0.3584],[-0.596,-0.1007],[-0.4685,-0.201],[-0.1594,-0.745],[-0.3357,-0.176],[-0.2022,0.0225],[-0.2185,-0.5747],[0.0711,-0.2639],[-0.2514,-0.1091],[-0.1778,-0.3821],[-0.1481,-0.3117]],"comm":[2,1,1,1,2,2,2,1,0,1,2,2,1,1,0,0,2,1,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0],"degree":[16,9,10,6,3,4,4,4,5,2,3,1,2,5,2,2,2,2,2,3,2,2,2,5,3,3,2,4,3,4,4,6,12,17],"betweenness":[0.4376,0.0539,0.1437,0.0119,0.0006,0.03,0.03,0.0,0.0559,0.0008,0.0006,0.0,0.0,0.0459,0.0,0.0,0.0,0.0,0.0,0.0325,0.0,0.0,0.0,0.0176,0.0022,0.0038,0.0,0.0223,0.0018,0.0029,0.0144,0.1383,0.1452,0.3041],"closeness":[0.569,0.4853,0.5593,0.4648,0.3793,0.3837,0.3837,0.44,0.5156,0.4342,0.3793,0.3667,0.3708,0.5156,0.3708,0.3708,0.2845,0.375,0.3708,0.5,0.3708,0.375,0.3708,0.3929,0.375,0.375,0.3626,0.4583,0.4521,0.3837,0.4583,0.541,0.5156,0.55],"eigenvector":[0.3555,0.266,0.3172,0.2112,0.076,0.0795,0.0795,0.171,0.2274,0.1027,0.076,0.0529,0.0843,0.2265,0.1014,0.1014,0.0236,0.0924,0.1014,0.1479,0.1014,0.0924,0.1014,0.1501,0.0571,0.0592,0.0756,0.1335,0.1311,0.135,0.1748,0.191,0.3087,0.3734],"clustering":[0.15,0.3333,0.2444,0.6667,0.6667,0.5,0.5,1.0,0.5,0,0.6667,0,1.0,0.6,1.0,1.0,1.0,1.0,1.0,0.3333,1.0,1.0,1.0,0.4,0.3333,0.3333,1.0,0.1667,0.3333,0.6667,0.5,0.2,0.197,0.1103]};
  const N = K.pos.length;
  const adj = Array.from({ length: N }, () => []);
  K.edges.forEach(([a, b]) => { adj[a].push(b); adj[b].push(a); });
  const MEASURES = {
    degree: { label: "degree k", fmt: (v) => String(v) },
    betweenness: { label: "betweenness", fmt: (v) => v.toFixed(3) },
    closeness: { label: "closeness", fmt: (v) => v.toFixed(3) },
    eigenvector: { label: "eigenvector", fmt: (v) => v.toFixed(3) },
  };

  function bfs(src) {
    const dist = new Array(N).fill(-1); dist[src] = 0;
    const q = [src];
    while (q.length) { const u = q.shift(); adj[u].forEach((v) => { if (dist[v] < 0) { dist[v] = dist[u] + 1; q.push(v); } }); }
    return dist;
  }
  const rankOf = (arr, i) => 1 + arr.filter((v) => v > arr[i]).length;

  VIZ.register("network-structure", {
    title: "Zachary's karate club: degree, clustering and centrality",
    init(ui) {
      const S = { measure: "degree", sel: null, dist: null, hover: null };
      VIZ.select(ui.controls, { label: "Node size", value: S.measure,
        options: Object.keys(MEASURES).map((k) => [k, MEASURES[k].label]),
        onChange: (v) => { S.measure = v; draw(); } });
      const g = VIZ.group(ui.controls);
      VIZ.button(g, "Clear selection", () => { S.sel = null; S.dist = null; draw(); });

      let nodeXY = [];
      const net = ui.panel({ title: "Network  ·  click a node", aspect: 1.15, minHeight: 280,
        label: "Karate club network; node size shows the chosen centrality, colour shows the community",
        onDraw: drawNet,
        onPointer: (e) => {
          if (e.type !== "pointerdown") return;
          let best = -1, bd = 1e9;
          nodeXY.forEach(([x, y, r], i) => { const d = Math.hypot(e.x - x, e.y - y) - r; if (d < bd) { bd = d; best = i; } });
          if (bd < 8) { S.sel = best; S.dist = bfs(best); } else { S.sel = null; S.dist = null; }
          draw();
        } });
      const hist = ui.panel({ title: "Degree distribution  p_k", aspect: 1.15, minHeight: 280,
        label: "Bar chart of the number of nodes with each degree", onDraw: drawHist });

      function draw() { net.draw(); hist.draw(); readout(); }

      function drawNet(c) {
        const T = VIZ.theme(), ctx = c.ctx, pad = 26;
        ctx.fillStyle = T.panel; ctx.fillRect(0, 0, c.w, c.h);
        const xs = K.pos.map((p) => p[0]), ys = K.pos.map((p) => p[1]);
        const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
        const sc = Math.min((c.w - 2 * pad) / (x1 - x0), (c.h - 2 * pad) / (y1 - y0));
        const ox = (c.w - sc * (x1 - x0)) / 2, oy = (c.h - sc * (y1 - y0)) / 2;
        const vals = K[S.measure], vmax = Math.max(...vals);
        nodeXY = K.pos.map(([x, y], i) => [ox + sc * (x - x0), oy + sc * (y1 - y), 5 + 13 * Math.sqrt(vals[i] / vmax)]);
        // edges; with a selection, edges on shortest paths from it are emphasised
        K.edges.forEach(([a, b]) => {
          const onPath = S.dist && Math.abs(S.dist[a] - S.dist[b]) === 1;
          ctx.strokeStyle = onPath ? T.accent : T.axis;
          ctx.globalAlpha = S.dist ? (onPath ? 0.9 : 0.15) : 0.45;
          ctx.lineWidth = onPath ? 2 : 1;
          ctx.beginPath(); ctx.moveTo(nodeXY[a][0], nodeXY[a][1]); ctx.lineTo(nodeXY[b][0], nodeXY[b][1]); ctx.stroke();
        });
        ctx.globalAlpha = 1;
        const cols = [T.c[0], T.c[1], T.c[2], T.c[3]];
        nodeXY.forEach(([x, y, r], i) => {
          ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
          ctx.fillStyle = cols[K.comm[i] % 4]; ctx.fill();
          ctx.lineWidth = i === S.sel ? 3.5 : 1.2; ctx.strokeStyle = i === S.sel ? T.stable : T.panel; ctx.stroke();
          if (S.dist || r > 10) {
            ctx.fillStyle = T.text; ctx.font = T.font(10, 700); ctx.textAlign = "center";
            const lab = S.dist ? String(S.dist[i]) : String(i);
            ctx.fillText(lab, x, y - r - 3);
          }
        });
        ctx.font = T.font(11, 600); ctx.textAlign = "left"; ctx.fillStyle = T.text;
        ctx.fillText(S.dist ? "labels: distance from the selected node" : "labels: node id (largest nodes)", 8, c.h - 8);
      }

      function drawHist(c) {
        const counts = {};
        K.degree.forEach((k) => { counts[k] = (counts[k] || 0) + 1; });
        const kmax = Math.max(...K.degree) + 1;
        const ax = VIZ.axes(c, { xlim: [0, kmax], ylim: [0, 0.4], xlabel: "degree k", ylabel: "p_k" });
        const T = ax.T, ctx = c.ctx;
        ax.clip();
        const bw = Math.max(3, (ax.sx(1) - ax.sx(0)) * 0.7);
        Object.keys(counts).forEach((k) => {
          const p = counts[k] / N, hit = S.sel !== null && K.degree[S.sel] === +k;
          ctx.fillStyle = hit ? (T.dark ? T.c[3] : T.accent) : T.c[0];
          ctx.fillRect(ax.sx(+k) - bw / 2, ax.sy(p), bw, ax.sy(0) - ax.sy(p));
        });
        const mean = K.degree.reduce((a, b) => a + b, 0) / N;
        ax.line([mean, mean], [0, 0.4], { color: T.unstable, width: 1.5, dash: [4, 4] });
        ax.label(`<k> = ${mean.toFixed(2)}`, mean, 0.37, { dx: 5 });
        ctx.restore();
      }

      function readout() {
        if (S.sel === null) {
          ui.readout.textContent = `N = ${N}, E = ${K.edges.length}. Click a node to see its degree, clustering and distances. ` +
            `Average clustering 0.571, average path length 2.408.`;
          return;
        }
        const i = S.sel, m = S.measure, ecc = Math.max(...S.dist);
        ui.readout.textContent = `Node ${i}: degree ${K.degree[i]}, clustering ${K.clustering[i].toFixed(3)}, ` +
          `${MEASURES[m].label} ${MEASURES[m].fmt(K[m][i])} (rank ${rankOf(K[m], i)} of ${N}); farthest node at distance ${ecc}.`;
      }

      draw();
    },
  });
})();

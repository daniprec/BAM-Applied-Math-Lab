// Parity test: runs the JavaScript model of the squid-ink widget for several settings and prints
// mean time to capture (t_max = 250). Compare with tests/squid_parity.py (tolerance about 5 t.u.).
//
//   node tests/squid_parity.js [seeds]        (default 40)
//
// The random numbers differ between the two languages, so the comparison is statistical.
"use strict";
const fs = require("fs");
const path = require("path");

// the same seeded generator as VIZ.rng in assets/viz/core.js
function rngFactory(seed) {
  let s = seed >>> 0 || 1;
  const u = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) + 0.5) / 4294967296; };
  u.normal = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
  return u;
}
global.window = { VIZ: { rng: rngFactory } };
require(path.join(__dirname, "..", "assets", "viz", "widgets", "squid-ink.js"));
const { makeModel, defaults } = window.VIZ.squidInk;

const T_MAX = 250;
function hunt(over, seed) {
  const p = Object.assign({}, defaults, over);
  const m = makeModel(p, rngFactory(seed * 7919 + 101));
  const steps = Math.round(T_MAX / p.dt);
  for (let i = 0; i < steps && !m.captured; i++) m.step();
  return { time: m.captured ? m.captureTime : T_MAX, caught: m.captured };
}

// keep in sync with SETTINGS in squid_parity.py
const SETTINGS = [
  ["no ink", { ink: false }],
  ["ink, searching fish", {}],
  ["ink, searching fish, jet x2", { jetBoost: 2.0 }],
  ["ink, searching fish, jet x2, turn 90 deg", { jetBoost: 2.0, escapeAngle: 90 }],
];

const nSeeds = +(process.argv[2] || 40);
console.log(`JavaScript model, ${nSeeds} seeds, t_max = ${T_MAX}`);
for (const [label, over] of SETTINGS) {
  const t = [];
  let uncaught = 0;
  for (let s = 0; s < nSeeds; s++) { const r = hunt(over, s); t.push(r.time); if (!r.caught) uncaught++; }
  const mean = t.reduce((a, b) => a + b, 0) / t.length;
  const sd = Math.sqrt(t.reduce((a, b) => a + (b - mean) ** 2, 0) / (t.length - 1));
  console.log(`${label.padEnd(44)} mean ${mean.toFixed(1).padStart(6)} +- ${(1.96 * sd / Math.sqrt(t.length)).toFixed(1).padStart(4)}  uncaught ${uncaught}/${nSeeds}`);
}

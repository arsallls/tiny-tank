// Canvas aquarium driven by model.js.
//
// Two clocks, deliberately separate: the model emits one behavior token every
// TICK_MS, the canvas draws at 60fps and interpolates toward whatever the
// current token implies. Drawing on the token clock would look like a
// slideshow. The transformer chooses behavior; it does not do sprite physics.

import { FishModel } from "./model.js";

const TICK_MS = 400;                      // 2.5 tokens/sec - slow enough to read
const $ = (s) => document.querySelector(s);

// An empty context has no circadian information and the model settles toward
// the sleep attractor: 44% of a visitor's first minute asleep, and 6 visits in
// 20 slept through it entirely. This 32-token awake prefix cuts that to 25%
// (the fish's own natural rate) with a much tighter spread - 0 bad visits in 20.
// Longer prefixes score a lower mean but WORSE variance: 192 tokens of
// hand-written behavior is off-distribution and the model answers erratically.
const AWAKE = ["DRIFT_R", "TURN", "DRIFT_L", "CIRCLE", "BUBBLE", "DRIFT_R", "TURN",
               "SCAN", "DRIFT_L", "DRIFT_R", "CIRCLE", "TURN", "DRIFT_U", "BUBBLE",
               "DRIFT_R", "TURN"];
const SEED = [...AWAKE, ...AWAKE];

const MOOD = {
  HUNGRY: ["hungry", "#e8a33d"], CONTENT: ["content", "#7fb069"],
  STARTLED: ["startled", "#d9533f"], BORED: ["bored", "#8a8578"],
  SLEEPY: ["sleepy", "#6b8cae"],
};

// A new token every 400ms is too fast to read, so the headline shows a COARSE
// activity (many tokens map to the same phrase) and holds each one for at least
// HOLD_MS. Tokens not listed here - sounds, moods, stimuli - leave it unchanged.
const HOLD_MS = 1500;
const ACTIVITY = {
  DRIFT_L: "drifting", DRIFT_R: "drifting", DRIFT_U: "drifting", DRIFT_D: "drifting",
  TURN: "drifting", HOVER: "hanging still", CIRCLE: "circling",
  DART_L: "darting", DART_R: "darting", DART_U: "darting", DART_D: "darting",
  SINK: "settling to the bottom", SURFACE: "heading for the surface",
  SCAN: "searching for food", HIDE: "hiding",
  ORIENT: "it has seen the food", CHOMP: "eating", NIBBLE: "eating", GULP: "eating",
};
const STIMULI = new Set(["<FOOD>", "<TAP>", "<HAND>", "<LIGHT_ON>", "<LIGHT_OFF>"]);
const EATING = new Set(["ORIENT", "CHOMP", "NIBBLE", "GULP"]);
// five vocabulary tokens are angle-bracketed (<FOOD>, <TAP>, ...) and the trace
// is built with innerHTML, so the browser was parsing them as tags and dropping
// them - the "you" badges showed with nothing beside them
const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const clock = (tick) => {
  const s = Math.round((tick * TICK_MS) / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

// The model holds no state variables, so these are read out of its own next-token
// distribution. Validated against the simulator's hidden variables over 2500
// ticks: drowsy r=+0.91, fear r=+0.88, hunger r=+0.84 WHILE AWAKE. Divisors are
// the observed 95th percentiles, so a full bar means "as strong as this gets".
const GAUGE = {
  hunger: [["SCAN", "SURFACE", "HUNGRY"], 0.48],
  fear: [["HIDE", "STARTLED", "SPLASH", "DART_D"], 0.36],
  drowsy: [["SLEEPY", "HOVER", "SINK", "DRIFT_D"], 0.96],
};
// A sleeping fish shows no hunger behaviour at all, so the readout collapses to
// zero while it naps (r falls 0.84 -> 0.31). Rather than display a confidently
// wrong 0%, the bar goes blank and says so.
const HUNGER_BLIND = 0.60;

// motion is calibrated by TRAVEL DISTANCE, not raw impulse: with per-frame drag
// d, one impulse carries v0/(1-d) of the tank. The old constants were written as
// if applied per tick but ran per frame, so a DART crossed 92% of the tank.
const DRAG = 0.955;                       // per frame at 60fps
const REACH = 1 / (1 - DRAG);             // 22.2
const DART = 0.16 / REACH;
const DRIFT = 0.05 / REACH;

// static decor, laid out once - positions as fractions of tank size
const ROCKS = [
  { x: 0.11, y: 0.94, w: 0.16, h: 0.075, tone: 0 },
  { x: 0.235, y: 0.955, w: 0.10, h: 0.05, tone: 1 },
  { x: 0.85, y: 0.945, w: 0.14, h: 0.065, tone: 0 },
  { x: 0.47, y: 0.965, w: 0.075, h: 0.035, tone: 1 },
  { x: 0.945, y: 0.955, w: 0.06, h: 0.03, tone: 0 },
  { x: 0.025, y: 0.95, w: 0.05, h: 0.028, tone: 1 },
];
const PLANTS = [
  { x: 0.09, h: 0.40, blades: 5, phase: 0.4 },
  { x: 0.635, h: 0.24, blades: 3, phase: 2.1 },
  { x: 0.935, h: 0.32, blades: 4, phase: 3.4 },
  { x: 0.30, h: 0.14, blades: 3, phase: 1.1 },
  { x: 0.78, h: 0.18, blades: 3, phase: 4.0 },
];
const DRIFTWOOD = { x1: 0.30, y1: 0.965, x2: 0.58, y2: 0.90, w: 10 };
const CORALS = [
  // hsl lightness is a PERCENT and clamps at 100 - the imported values (120,
  // 150) both rendered as pure white sticks. Read as 0-255 they are 47% / 59%.
  { x: 0.50, y: 0.965, h: 0.11, branches: 4, hue: [196, 96, 47] },
  { x: 0.905, y: 0.955, h: 0.085, branches: 3, hue: [214, 84, 59] },
];
const ORNAMENT = { x: 0.68, y: 0.965, w: 0.10, h: 0.16 };

class Tank {
  constructor(model, canvas) {
    this.m = model;
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.id = Object.fromEntries(model.cfg.vocab.map((t, i) => [t, i]));
    this.fish = { x: 0.5, y: 0.5, vx: 0.002, vy: 0, face: 1, wag: 0, mouth: 0 };
    this.pellets = []; this.bubbles = []; this.trace = []; this.pending = [];
    this.thoughts = [];
    this.activity = "waking up"; this.wantActivity = null; this.activityAt = 0;
    this.mood = null;
    this.gauges = { hunger: 0, fear: 0, drowsy: 0 };
    this.lights = true;
    this.ticks = 0;
    this.circle = 0;
    this.audio = null;
    for (const t of SEED) this.logits = this.m.step(this.id[t]);
  }

  inject(tok) {
    this.pending.push(tok);
    if (!this.audio) this.audio = new (window.AudioContext || webkitAudioContext)();
  }

  blip(type) {
    if (!this.audio) return;
    const a = this.audio, t = a.currentTime, o = a.createOscillator(), g = a.createGain();
    const spec = { BUBBLE: [380, 900, 0.16, 0.07], GULP: [220, 110, 0.13, 0.12],
                   SPLASH: [900, 240, 0.18, 0.09] }[type];
    o.type = type === "SPLASH" ? "triangle" : "sine";
    o.frequency.setValueAtTime(spec[0], t);
    o.frequency.exponentialRampToValueAtTime(spec[1], t + spec[3]);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(spec[2], t + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t + spec[3]);
    o.connect(g).connect(a.destination);
    o.start(t); o.stop(t + spec[3] + 0.02);
  }

  readGauges() {
    let mx = -Infinity;
    for (const v of this.logits) if (v > mx) mx = v;
    let z = 0; const p = new Float32Array(this.logits.length);
    for (let i = 0; i < p.length; i++) { p[i] = Math.exp(this.logits[i] - mx); z += p[i]; }
    for (const [name, [toks, div]] of Object.entries(GAUGE)) {
      let s = 0;
      for (const t of toks) s += p[this.id[t]] / z;
      const v = Math.max(0, Math.min(1, s / div));
      this.gauges[name] += 0.10 * (v - this.gauges[name]);      // EMA, alpha .10
    }
  }

  nearestPellet() {
    const f = this.fish;
    let best = null, bd = Infinity;
    for (const p of this.pellets) {
      const d = Math.hypot(p.x - f.x, p.y - f.y);
      if (d < bd) { bd = d; best = p; }
    }
    return [best, bd];
  }

  tick() {
    const fromUser = this.pending.length > 0;
    const tok = fromUser
      ? this.id[this.pending.shift()]
      : this.m.sample(this.logits, 1.0);
    this.logits = this.m.step(tok);
    const name = this.m.cfg.vocab[tok];
    this.ticks++;
    this.trace.push({ tick: this.ticks, tok: name, user: fromUser });
    if (this.trace.length > 90) this.trace.shift();
    this.act(name);
    this.readGauges();
    this.paintHud(name);
  }

  // hold each headline for HOLD_MS; a change arriving inside the hold is kept
  // as `wantActivity` and applied when the hold expires
  setActivity(tok, now) {
    const a = ACTIVITY[tok];
    if (a) this.wantActivity = a;
    if (this.wantActivity && this.wantActivity !== this.activity &&
        now - this.activityAt >= HOLD_MS) {
      this.activity = this.wantActivity;
      this.activityAt = now;
      return true;
    }
    return false;
  }

  act(t) {
    const f = this.fish;
    if (MOOD[t] && t !== this.mood) {
      this.mood = t;
      this.thoughts.unshift({ tick: this.ticks, mood: t });
      if (this.thoughts.length > 7) this.thoughts.pop();
    }
    switch (t) {
      case "DART_L": f.vx = -DART; f.face = -1; break;
      case "DART_R": f.vx = DART; f.face = 1; break;
      case "DART_U": f.vy = -DART; break;
      case "DART_D": f.vy = DART; break;
      case "DRIFT_L": f.vx = -DRIFT; f.face = -1; break;
      case "DRIFT_R": f.vx = DRIFT; f.face = 1; break;
      case "DRIFT_U": f.vy = -DRIFT; break;
      case "DRIFT_D": f.vy = DRIFT; break;
      case "HOVER": f.vx *= 0.3; f.vy *= 0.3; break;
      case "SINK": f.vy = DRIFT * 0.8; f.vx *= 0.5; break;
      case "TURN": f.face *= -1; f.vx = -f.vx * 0.4; break;
      case "CIRCLE": this.circle = 34; break;
      case "SURFACE": f.vy = -DRIFT * 1.6; break;
      case "HIDE": f.vy = DRIFT * 1.4; f.vx = DRIFT * 0.7 * f.face; break;
      case "SCAN": f.vx *= 0.4; f.face *= -1; break;
      case "ORIENT": {
        const [p] = this.nearestPellet();
        // gain 0.03 x reach 22.2 closes ~2/3 of the gap per ORIENT; at 0.012 it
        // only closed a quarter and the fish never arrived before the pellet sank
        if (p) { f.face = p.x > f.x ? 1 : -1; f.vx = (p.x - f.x) * 0.03; f.vy = (p.y - f.y) * 0.03; }
        break;
      }
      case "CHOMP": case "NIBBLE": {
        f.mouth = 1;
        const [p, d] = this.nearestPellet();
        if (!p) break;
        if (d < 0.14) {
          this.pellets.splice(this.pellets.indexOf(p), 1);
        } else {
          // it decided to eat but the food is across the tank, so read that as
          // a lunge. Consuming regardless of distance (the old shortcut) let it
          // swallow pellets it never swam to.
          f.face = p.x > f.x ? 1 : -1;
          f.vx = (p.x - f.x) / d * DART;
          f.vy = (p.y - f.y) / d * DART;
        }
        break;
      }
      case "BUBBLE": case "GULP": case "SPLASH":
        this.blip(t);
        for (let i = 0; i < (t === "SPLASH" ? 6 : 2); i++)
          this.bubbles.push({ x: f.x + f.face * 0.03, y: f.y, r: 1.5 + Math.random() * 3,
                              v: 0.0018 + Math.random() * 0.0022, w: Math.random() * 6.28 });
        break;
      case "<FOOD>":
        // inside the frame, not above it: at 0.036/s a pellet spawned at
        // y=-0.15 took 4s to even become visible, and the fish had usually
        // "eaten" it by then
        for (let i = 0; i < 3; i++)
          this.pellets.push({ x: 0.3 + Math.random() * 0.4, y: 0.03 + i * 0.05 });
        break;
      case "<LIGHT_ON>": this.lights = true; break;
      case "<LIGHT_OFF>": this.lights = false; break;
    }
  }

  physics(dt) {
    const f = this.fish, k = dt * 60;
    if (this.circle > 0) {
      this.circle--; const a = this.circle * 0.19;
      f.vx = Math.cos(a) * DRIFT * 2.4; f.vy = Math.sin(a) * DRIFT * 2.4;
      f.face = f.vx > 0 ? 1 : -1;
    }
    f.x += f.vx * k; f.y += f.vy * k;
    const d = Math.pow(DRAG, k);
    f.vx *= d; f.vy = f.vy * d + 0.000015 * k;         // drag + a touch of sink
    if (f.x < 0.06) { f.x = 0.06; f.vx = Math.abs(f.vx) * 0.5; f.face = 1; }
    if (f.x > 0.94) { f.x = 0.94; f.vx = -Math.abs(f.vx) * 0.5; f.face = -1; }
    if (f.y < 0.08) { f.y = 0.08; f.vy = Math.abs(f.vy) * 0.5; }
    if (f.y > 0.92) { f.y = 0.92; f.vy = -Math.abs(f.vy) * 0.5; }
    f.wag += (0.035 + Math.hypot(f.vx, f.vy) * 25) * k;
    f.mouth *= Math.pow(0.9, k);
    for (const p of this.pellets) p.y += 0.0006 * k;
    this.pellets = this.pellets.filter((p) => p.y < 0.98);
    for (const b of this.bubbles) { b.y -= b.v * k; b.w += 0.08 * k; }
    this.bubbles = this.bubbles.filter((b) => b.y > 0.02);
  }

  drawSubstrate(c, W, H, dim) {
    const gy = H * 0.90;
    const g = c.createLinearGradient(0, gy, 0, H);
    g.addColorStop(0, "rgba(50,38,26,0)");
    g.addColorStop(1, `rgba(36,27,18,${0.92 * dim})`);
    c.fillStyle = g; c.fillRect(0, gy, W, H - gy);
    for (let i = 0; i < 30; i++) {
      const px = ((i * 53.7) % 100) / 100 * W + Math.sin(i * 7.1) * 6;
      const py = H * (0.925 + ((i * 31) % 7) / 7 * 0.06);
      const r = 1.6 + (i * 17) % 5;
      c.fillStyle = i % 3 === 0 ? `rgba(150,130,100,${0.55 * dim})` : `rgba(88,70,50,${0.6 * dim})`;
      c.beginPath(); c.ellipse(px, py, r, r * 0.68, 0, 0, 6.29); c.fill();
    }
  }

  drawRocks(c, W, H, dim) {
    for (const r of ROCKS) {
      const x = r.x * W, y = r.y * H, w = r.w * W, h = r.h * H;
      const g = c.createRadialGradient(x - w * 0.3, y - h * 0.55, 1, x, y, w);
      if (r.tone === 0) { g.addColorStop(0, `rgba(122,113,100,${dim})`); g.addColorStop(1, `rgba(52,48,42,${dim})`); }
      else { g.addColorStop(0, `rgba(102,97,88,${dim})`); g.addColorStop(1, `rgba(46,44,38,${dim})`); }
      c.fillStyle = g;
      c.beginPath(); c.ellipse(x, y, w, h, 0, 0, 6.29); c.fill();
    }
    const d = DRIFTWOOD, x1 = d.x1 * W, y1 = d.y1 * H, x2 = d.x2 * W, y2 = d.y2 * H;
    c.strokeStyle = `rgba(74,52,34,${dim})`; c.lineWidth = d.w; c.lineCap = "round";
    c.beginPath(); c.moveTo(x1, y1); c.quadraticCurveTo((x1 + x2) / 2, y1 - 10, x2, y2); c.stroke();
    c.strokeStyle = `rgba(104,76,50,${0.6 * dim})`; c.lineWidth = 2;
    c.beginPath(); c.moveTo(x1, y1 - 2); c.quadraticCurveTo((x1 + x2) / 2, y1 - 12, x2, y2 - 2); c.stroke();
  }

  drawPlants(c, W, H, now, dim) {
    for (const p of PLANTS) {
      const bx = p.x * W, by = H * 0.955;
      for (let b = 0; b < p.blades; b++) {
        const bh = p.h * H * (0.7 + 0.3 * Math.sin(b * 1.7 + 1));
        const sway = Math.sin(now / 1500 + p.phase + b * 0.55) * bh * 0.2;
        const off = (b - p.blades / 2) * 5.5;
        c.strokeStyle = `rgba(${58 + b * 5},${104 + b * 9},${56 + b * 5},${0.78 * dim})`;
        c.lineWidth = 3.4 - b * 0.28;
        c.lineCap = "round";
        c.beginPath();
        c.moveTo(bx + off, by);
        c.quadraticCurveTo(bx + off + sway * 0.55, by - bh * 0.55, bx + off + sway, by - bh);
        c.stroke();
      }
    }
  }

  drawCoral(c, W, H, now, dim) {
    for (const cr of CORALS) {
      const bx = cr.x * W, by = cr.y * H, [h1, s1, l1] = cr.hue;
      for (let i = 0; i < cr.branches; i++) {
        const bh = cr.h * H * (0.6 + 0.4 * Math.sin(i * 2.3));
        const sway = Math.sin(now / 1700 + i * 1.3) * bh * 0.12;
        const ang = (i - cr.branches / 2) * 0.32;
        c.strokeStyle = `hsla(${h1},${s1}%,${l1}%,${0.85 * dim})`;
        c.lineWidth = 5 - i * 0.4; c.lineCap = "round";
        c.beginPath();
        c.moveTo(bx, by);
        c.quadraticCurveTo(bx + Math.sin(ang) * bh * 0.4 + sway * 0.5, by - bh * 0.6,
                            bx + Math.sin(ang) * bh * 0.9 + sway, by - bh);
        c.stroke();
      }
    }
  }

  drawOrnament(c, W, H, dim) {
    const o = ORNAMENT, x = o.x * W, y = o.y * H, w = o.w * W, h = o.h * H;
    const g = c.createLinearGradient(x - w / 2, y - h, x + w / 2, y);
    g.addColorStop(0, `rgba(196,186,160,${dim})`); g.addColorStop(1, `rgba(140,130,108,${dim})`);
    c.fillStyle = g; c.lineJoin = "round";
    // simple sunken archway: two pillars + a rounded lintel
    c.beginPath();
    c.moveTo(x - w / 2, y); c.lineTo(x - w / 2, y - h * 0.55);
    c.arc(x, y - h * 0.55, w / 2, Math.PI, 0);
    c.lineTo(x + w / 2, y); c.lineTo(x + w * 0.32, y);
    c.lineTo(x + w * 0.32, y - h * 0.45);
    c.arc(x, y - h * 0.45, w * 0.32, 0, Math.PI, true);
    c.lineTo(x - w / 2, y);
    c.closePath(); c.fill();
    c.strokeStyle = `rgba(90,84,68,${0.5 * dim})`; c.lineWidth = 1.5; c.stroke();
  }

  draw(now) {
    const c = this.ctx, W = this.cv.clientWidth, H = this.cv.clientHeight;
    c.clearRect(0, 0, W, H);
    const dim = this.lights ? 1 : 0.34;

    c.save();
    for (let i = 0; i < 3; i++) {
      const p = (now / 14000 + i * 0.37) % 1.4 - 0.2;
      const g = c.createLinearGradient(p * W, 0, p * W + W * 0.16, H);
      g.addColorStop(0, `rgba(180,224,214,${0.075 * dim})`);
      g.addColorStop(1, "rgba(180,224,214,0)");
      c.fillStyle = g;
      c.beginPath(); c.moveTo(p * W, 0); c.lineTo(p * W + W * 0.11, 0);
      c.lineTo(p * W + W * 0.3, H); c.lineTo(p * W + W * 0.12, H); c.fill();
    }
    c.restore();

    this.drawSubstrate(c, W, H, dim);
    this.drawRocks(c, W, H, dim);
    this.drawOrnament(c, W, H, dim);
    this.drawCoral(c, W, H, now, dim);
    this.drawPlants(c, W, H, now, dim);

    for (const p of this.pellets) {
      c.fillStyle = `rgba(214,158,74,${dim})`;
      c.beginPath(); c.arc(p.x * W, p.y * H, 3.2, 0, 6.29); c.fill();
    }
    for (const b of this.bubbles) {
      c.strokeStyle = `rgba(198,232,226,${0.5 * dim})`; c.lineWidth = 1;
      c.beginPath(); c.arc(b.x * W + Math.sin(b.w) * 4, b.y * H, b.r, 0, 6.29); c.stroke();
    }

    const f = this.fish, x = f.x * W, y = f.y * H, s = Math.min(W, H) * 0.085;
    c.save(); c.translate(x, y); c.scale(f.face, 1);
    c.rotate(Math.max(-0.4, Math.min(0.4, f.vy * 40)));

    c.fillStyle = `rgba(90,60,42,${0.18 * dim})`;
    c.beginPath(); c.ellipse(2, s * 0.5, s * 0.9, s * 0.16, 0, 0, 6.29); c.fill();

    const tail = Math.sin(f.wag) * 0.5;
    const dk = this.lights ? 1 : 0.55;

    // forked tail: two translucent lobes with fin rays, peduncle between them
    const drawLobe = (dir) => {
      const t = tail * dir;
      c.beginPath();
      c.moveTo(-s * 0.86, 0);
      c.quadraticCurveTo(-s * 1.25, dir * s * 0.18 + t * s * .2, -s * 1.86, dir * s * 0.9 + t * s * .55);
      c.quadraticCurveTo(-s * 1.5, dir * s * 0.62 + t * s * .4, -s * 1.32, dir * s * 0.3 + t * s * .3);
      c.quadraticCurveTo(-s * 1.15, dir * s * 0.08, -s * 0.86, 0);
      c.closePath(); c.fill();
      c.save(); c.clip();
      c.strokeStyle = `rgba(120,50,20,${0.3 * dk})`; c.lineWidth = Math.max(0.8, s * 0.025);
      for (let i = 1; i <= 3; i++) {
        c.beginPath(); c.moveTo(-s * 0.86, 0);
        c.lineTo(-s * (0.9 + i * 0.3), dir * s * (0.15 + i * 0.26) + t * s * (0.2 + i * 0.15));
        c.stroke();
      }
      c.restore();
    };
    c.fillStyle = this.lights ? "rgba(219,102,52,.88)" : "rgba(110,66,42,.88)";
    drawLobe(1); drawLobe(-1);

    // pectoral fin, opposite phase from the tail
    const pec = Math.sin(f.wag + 1.6) * 0.4;
    c.fillStyle = this.lights ? "rgba(224,146,84,.85)" : "rgba(140,90,60,.85)";
    c.beginPath();
    c.moveTo(s * 0.05, s * 0.3);
    c.quadraticCurveTo(s * 0.02, s * 0.62 + pec * s * 0.16, -s * 0.18, s * 0.68 + pec * s * 0.22);
    c.quadraticCurveTo(s * 0.06, s * 0.56, s * 0.22, s * 0.36);
    c.fill();
    // small anal fin
    c.fillStyle = this.lights ? "rgba(200,120,64,.8)" : "rgba(120,76,48,.8)";
    c.beginPath();
    c.moveTo(-s * 0.32, s * 0.5); c.lineTo(-s * 0.5, s * 0.86); c.lineTo(-s * 0.06, s * 0.58); c.closePath(); c.fill();

    // body: deeper, more tapered teardrop instead of a plain ellipse
    const body = c.createLinearGradient(0, -s * 0.62, 0, s * 0.62);
    body.addColorStop(0, this.lights ? "#f8b46a" : "#9c6039");
    body.addColorStop(0.5, this.lights ? "#e8863f" : "#875530");
    body.addColorStop(1, this.lights ? "#c1541f" : "#7a4526");
    c.fillStyle = body;
    c.beginPath();
    c.moveTo(s * 1.02, 0);
    c.quadraticCurveTo(s * 0.7, -s * 0.62, -s * 0.15, -s * 0.58);
    c.quadraticCurveTo(-s * 0.65, -s * 0.5, -s * 0.86, 0);
    c.quadraticCurveTo(-s * 0.65, s * 0.5, -s * 0.15, s * 0.58);
    c.quadraticCurveTo(s * 0.7, s * 0.62, s * 1.02, 0);
    c.fill();

    // three faint bands for visual interest + a belly highlight + gill mark
    c.save(); c.clip(new Path2D(`M ${s*1.02} 0 Q ${s*0.7} ${-s*0.62} ${-s*0.15} ${-s*0.58} Q ${-s*0.65} ${-s*0.5} ${-s*0.86} 0 Q ${-s*0.65} ${s*0.5} ${-s*0.15} ${s*0.58} Q ${s*0.7} ${s*0.62} ${s*1.02} 0 Z`));
    c.strokeStyle = `rgba(255,255,255,${this.lights ? 0.16 : 0.07})`;
    c.lineWidth = s * 0.09;
    for (const bx of [-0.5, -0.1, 0.35]) {
      c.beginPath(); c.moveTo(bx * s, -s * 0.6); c.lineTo(bx * s, s * 0.6); c.stroke();
    }
    c.fillStyle = `rgba(255,240,210,${this.lights ? 0.28 : 0.12})`;
    c.beginPath(); c.ellipse(s * 0.05, s * 0.32, s * 0.75, s * 0.14, 0, 0, 6.29); c.fill();
    c.restore();
    c.strokeStyle = `rgba(80,45,20,${0.35 * dk})`; c.lineWidth = Math.max(1, s * 0.035);
    c.beginPath(); c.arc(s * 0.62, s * 0.04, s * 0.2, -1, 1); c.stroke();

    // dorsal fin, translucent with rays
    c.fillStyle = this.lights ? "rgba(233,153,86,.82)" : "rgba(140,90,60,.82)";
    c.beginPath();
    c.moveTo(-s * 0.28, -s * 0.52);
    c.quadraticCurveTo(-s * 0.05, -s * 1.28 - Math.sin(f.wag * 0.7) * s * 0.12, s * 0.5, -s * 0.5);
    c.quadraticCurveTo(s * 0.12, -s * 0.68, -s * 0.28, -s * 0.52);
    c.fill();

    // eye: pale ring, dark pupil, glint
    c.fillStyle = this.lights ? "#fdf6e8" : "#c9b89a";
    c.beginPath(); c.arc(s * 0.6, -s * 0.14, s * 0.19, 0, 6.29); c.fill();
    c.fillStyle = "#17110d";
    c.beginPath(); c.arc(s * 0.66, -s * 0.14, s * 0.095, 0, 6.29); c.fill();
    c.fillStyle = "rgba(255,255,255,.85)";
    c.beginPath(); c.arc(s * 0.63, -s * 0.18, s * 0.03, 0, 6.29); c.fill();

    c.strokeStyle = "#7a3512"; c.lineWidth = Math.max(1.4, s * 0.07); c.lineCap = "round";
    c.beginPath();
    c.arc(s * 0.98, s * 0.1, s * 0.16, 0.5 - f.mouth * 0.7, 1.9 + f.mouth * 0.7);
    c.stroke();
    c.restore();

    // the mood label stays until the mood actually changes - it is the fish's
    // current state, not a notification
    if (this.mood) {
      const [label, col] = MOOD[this.mood];
      c.fillStyle = col;
      c.font = "700 12px 'Space Mono', ui-monospace, monospace";
      c.textAlign = "center";
      c.fillText(label, x, y - s * 1.5);
    }
  }

  traceClass(tok, user) {
    if (STIMULI.has(tok)) return user ? "tr-you" : "tr-world";
    if (MOOD[tok]) return "tr-mood";
    if (EATING.has(tok)) return "tr-eat";
    return "tr-move";
  }

  paintHud(tok) {
    const now = performance.now();
    const el = $("#tok");
    if (this.setActivity(tok, now)) {
      el.classList.add("fade");
      setTimeout(() => { el.textContent = this.activity; el.classList.remove("fade"); }, 180);
    } else if (el.textContent === "—") {
      el.textContent = this.activity;
    }
    $("#rawtok").textContent = tok;
    $("#ticks").textContent = `${this.ticks.toLocaleString()} ticks · ${clock(this.ticks)}`;
    $("#ctxbar").style.width = Math.min(100, (this.m.len / this.m.cap) * 100) + "%";
    $("#ctx").textContent = `${this.m.len} / ${this.m.cap}`;
    for (const name of Object.keys(GAUGE)) {
      $(`#g-${name}`).style.width = (this.gauges[name] * 100).toFixed(1) + "%";
      $(`#v-${name}`).textContent = Math.round(this.gauges[name] * 100) + "%";
    }
    const blind = this.gauges.drowsy > HUNGER_BLIND;
    $("#row-hunger").classList.toggle("unknown", blind);
    if (blind) $("#v-hunger").textContent = "?";
    $("#lightstate").textContent = this.lights ? "on" : "off";
    $("#light").classList.toggle("off", !this.lights);
    $("#thoughts").innerHTML = this.thoughts.length
      ? this.thoughts.map(({ tick, mood }) => {
          const [label, col] = MOOD[mood];
          return `<li><span style="color:${col}">${label}</span><b>${clock(tick)}</b></li>`;
        }).join("")
      : '<li class="empty">nothing yet</li>';

    const box = $("#trace");
    if ($("#tracebox").open) {
      const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
      box.innerHTML = this.trace.map(({ tick, tok, user }) =>
        `<span class="${this.traceClass(tok, user)}" title="tick ${tick} · ${clock(tick)}">${esc(tok)}</span>`
      ).join(" ");   // the space matters: adjacent inline spans with nowrap give
                     // the line no break opportunity and blow out the grid column
      if (stick) box.scrollTop = box.scrollHeight;     // follow only if already at the end
    }
  }
}

(async function main() {
  const status = $("#status");
  try {
    const model = await FishModel.load("");
    const parity = await (await fetch("parity.json")).json();
    const r = model.selfCheck(parity);
    $("#parity").textContent = r.maxAbsDiff.toExponential(2);
    $("#paritybadge").classList.add(r.pass ? "ok" : "bad");
    $("#paritybadge").title = `max |JS - PyTorch| over ${parity.ids.length} positions`;
    $("#nparams").textContent = (model.cfg.n_params / 1000).toFixed(0) + "k";

    const cv = $("#tank");
    const fit = () => {
      const d = window.devicePixelRatio || 1;
      cv.width = cv.clientWidth * d; cv.height = cv.clientHeight * d;
      cv.getContext("2d").setTransform(d, 0, 0, d, 0, 0);
    };
    fit(); addEventListener("resize", fit);

    const tank = new Tank(model, cv);
    status.remove();

    for (const [sel, tok] of [["#feed", "<FOOD>"], ["#tap", "<TAP>"], ["#hand", "<HAND>"]])
      $(sel).onclick = () => tank.inject(tok);
    $("#light").onclick = () => tank.inject(tank.lights ? "<LIGHT_OFF>" : "<LIGHT_ON>");

    setInterval(() => tank.tick(), TICK_MS);
    let last = performance.now();
    (function frame(now) {
      tank.physics(Math.min(0.05, (now - last) / 1000)); last = now;
      tank.draw(now);
      requestAnimationFrame(frame);
    })(last);
  } catch (e) {
    status.textContent = "failed to start: " + e.message;
    status.classList.add("bad");
  }
})();

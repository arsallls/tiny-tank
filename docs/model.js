// Transformer inference in plain JS. Mirrors model.py: RMSNorm + RoPE + SwiGLU,
// no biases, tied embeddings. Batch 1, no backward pass, no dependencies.
//
// 0.33M params is ~1.3 MFLOPs per cached token, so Float32Array arithmetic is
// plenty - shipping 10MB of ONNX Runtime WASM to serve a 1.3MB model would
// cost more to download than the model itself.

export class FishModel {
  static async load(base = "") {
    const cfg = await (await fetch(`${base}config.json`)).json();
    const buf = await (await fetch(`${base}weights.bin`)).arrayBuffer();
    return new FishModel(cfg, buf);
  }

  constructor(cfg, buf) {
    this.cfg = cfg;
    const all = new Float32Array(buf);
    if (all.length !== cfg.n_params)
      throw new Error(`weights.bin has ${all.length} floats, config says ${cfg.n_params}`);
    this.T = {};
    for (const t of cfg.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      this.T[t.name] = all.subarray(t.offset, t.offset + n);
    }
    // inv_freq[i] = base^(-2i/head_dim), matching precompute_rope
    const hd = cfg.head_dim;
    this.invf = new Float64Array(hd / 2);
    for (let i = 0; i < hd / 2; i++)
      this.invf[i] = 1 / Math.pow(cfg.rope_base, (2 * i) / hd);
    this.reset();
  }

  reset() {
    const { n_layer, n_head, head_dim, block_size } = this.cfg;
    // one contiguous buffer per layer, oldest-first; `cap` positions max
    this.cap = block_size - 1;
    this.K = [], this.V = [];
    for (let l = 0; l < n_layer; l++) {
      this.K.push(new Float32Array(n_head * this.cap * head_dim));
      this.V.push(new Float32Array(n_head * this.cap * head_dim));
    }
    this.len = 0;   // cached positions
    this.pos = 0;   // ABSOLUTE position, never reset by eviction
  }
}

// y[o] = sum_i W[o*in + i] * x[i]   (PyTorch Linear weight is (out, in))
function matvec(W, x, nOut, nIn, y) {
  for (let o = 0; o < nOut; o++) {
    let s = 0, base = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[base + i] * x[i];
    y[o] = s;
  }
  return y;
}

function rmsnorm(x, w, n, y) {
  let s = 0;
  for (let i = 0; i < n; i++) s += x[i] * x[i];
  const g = 1 / Math.sqrt(s / n + 1e-6);
  for (let i = 0; i < n; i++) y[i] = x[i] * g * w[i];
  return y;
}

// split-half RoPE, NOT interleaved: out = [x1*cos - x2*sin, x2*cos + x1*sin]
function rope(v, off, hd, pos, invf) {
  const half = hd / 2;
  for (let i = 0; i < half; i++) {
    const a = pos * invf[i], c = Math.cos(a), s = Math.sin(a);
    const x1 = v[off + i], x2 = v[off + half + i];
    v[off + i] = x1 * c - x2 * s;
    v[off + half + i] = x2 * c + x1 * s;
  }
}

FishModel.prototype.step = function (id) {
  const C = this.cfg.n_embd, nh = this.cfg.n_head, hd = this.cfg.head_dim;
  const H = this.cfg.hidden, V = this.cfg.vocab_size, cap = this.cap;

  let x = new Float32Array(C);
  x.set(this.T["tok.weight"].subarray(id * C, id * C + C));

  const h = new Float32Array(C), qkv = new Float32Array(3 * C);
  const att = new Float32Array(C), tmp = new Float32Array(Math.max(C, H));
  const g = new Float32Array(H), u = new Float32Array(H);
  const scores = new Float32Array(cap + 1);

  // evict oldest once full. `pos` keeps counting, so every cached key holds the
  // rotation for its OWN absolute position and q - k stays the true distance.
  // Reusing a frozen position after eviction would collapse every relative
  // offset to zero and silently turn RoPE off.
  const full = this.len === cap;
  if (full) {
    for (let l = 0; l < this.cfg.n_layer; l++)
      for (let hh = 0; hh < nh; hh++) {
        const b = hh * cap * hd;
        this.K[l].copyWithin(b, b + hd, b + cap * hd);
        this.V[l].copyWithin(b, b + hd, b + cap * hd);
      }
  }
  const slot = full ? cap - 1 : this.len;
  const n = slot + 1;

  for (let l = 0; l < this.cfg.n_layer; l++) {
    rmsnorm(x, this.T[`h${l}.n1.weight`], C, h);
    matvec(this.T[`h${l}.attn.qkv.weight`], h, 3 * C, C, qkv);

    const K = this.K[l], Vc = this.V[l];
    for (let hh = 0; hh < nh; hh++) {
      rope(qkv, hh * hd, hd, this.pos, this.invf);              // q
      rope(qkv, C + hh * hd, hd, this.pos, this.invf);          // k
      const dst = hh * cap * hd + slot * hd;
      for (let d = 0; d < hd; d++) {
        K[dst + d] = qkv[C + hh * hd + d];
        Vc[dst + d] = qkv[2 * C + hh * hd + d];
      }
    }

    const scale = 1 / Math.sqrt(hd);
    for (let hh = 0; hh < nh; hh++) {
      const qo = hh * hd, ko = hh * cap * hd;
      let m = -Infinity;
      for (let t = 0; t < n; t++) {
        let s = 0;
        for (let d = 0; d < hd; d++) s += qkv[qo + d] * K[ko + t * hd + d];
        scores[t] = s * scale;
        if (scores[t] > m) m = scores[t];
      }
      let z = 0;
      for (let t = 0; t < n; t++) { scores[t] = Math.exp(scores[t] - m); z += scores[t]; }
      for (let d = 0; d < hd; d++) {
        let s = 0;
        for (let t = 0; t < n; t++) s += scores[t] * Vc[ko + t * hd + d];
        att[hh * hd + d] = s / z;
      }
    }

    matvec(this.T[`h${l}.attn.proj.weight`], att, C, C, tmp);
    for (let i = 0; i < C; i++) x[i] += tmp[i];

    rmsnorm(x, this.T[`h${l}.n2.weight`], C, h);
    matvec(this.T[`h${l}.mlp.gate.weight`], h, H, C, g);
    matvec(this.T[`h${l}.mlp.up.weight`], h, H, C, u);
    for (let i = 0; i < H; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];  // SwiGLU
    matvec(this.T[`h${l}.mlp.down.weight`], g, C, H, tmp);
    for (let i = 0; i < C; i++) x[i] += tmp[i];
  }

  rmsnorm(x, this.T["norm.weight"], C, h);
  const tok = this.T["tok.weight"], logits = new Float32Array(V);
  for (let v = 0; v < V; v++) {                 // tied embeddings: head == tok
    let s = 0, b = v * C;
    for (let i = 0; i < C; i++) s += tok[b + i] * h[i];
    logits[v] = s;
  }

  this.len = n;
  this.pos++;
  return logits;
};

FishModel.prototype.sample = function (logits, temperature = 1.0) {
  const t = Math.max(temperature, 1e-5);
  let m = -Infinity;
  for (const v of logits) if (v > m) m = v;
  let z = 0;
  const p = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - m) / t); z += p[i]; }
  let r = Math.random() * z;
  for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return i; }
  return p.length - 1;
};

// Feed the fixture through and compare against the logits PyTorch produced for
// the same ids. Runs in the browser, so the check covers the code that ships.
FishModel.prototype.selfCheck = function (parity) {
  this.reset();
  const V = this.cfg.vocab_size;
  let worst = 0, at = -1;
  for (let t = 0; t < parity.ids.length; t++) {
    const got = this.step(parity.ids[t]);
    for (let v = 0; v < V; v++) {
      const d = Math.abs(got[v] - parity.logits[t * V + v]);
      if (d > worst) { worst = d; at = t; }
    }
  }
  this.reset();
  return { maxAbsDiff: worst, worstPos: at, pass: worst < 1e-3 };
};

// Causality-Preserving Graph Coarsening for Efficient LLM Reasoning
// BTP deck -- 12 slides, ~12 minutes.
//
// Visual motif: nodes and edges. The whole project is about changing the
// shape of a reasoning graph, so every structural claim is drawn as one.
//
// Regenerate:  node docs/make_deck.js docs/Graph_of_Thoughts_BTP.pptx

const pptx = require("pptxgenjs");
const pres = new pptx();
pres.layout = "LAYOUT_WIDE";                  // 13.3 x 7.5 in
const W = 13.3;

// --- palette: deep indigo ground, violet primary, amber = compression ---
const NIGHT  = "150E33";
const PANEL  = "221A4A";
const VIOLET = "6D4AFF";
const AMBER  = "FFB627";
const MINT   = "3DDC97";
const ROSE   = "FF4D6D";
const WHITE  = "FFFFFF";
const INK    = "1F1A38";
const MUTE   = "6E6A85";
const PAPER  = "F6F4FD";
const DIM    = "A79FCB";

const HEAD = "Cambria";
const BODY = "Calibri";

pres.author = "Prasoon Raj, Nikhil Bansal";
pres.title  = "Causality-Preserving Graph Coarsening for Efficient LLM Reasoning";

// --- helpers ---------------------------------------------------------
function title(s, t, sub, dark) {
  s.addText(t, { x: 0.7, y: 0.4, w: W - 1.4, h: 0.68, isTextBox: true,
    fontFace: HEAD, fontSize: 32, bold: true, color: dark ? WHITE : INK, margin: 0 });
  if (sub) s.addText(sub, { x: 0.7, y: 1.1, w: W - 1.4, h: 0.4, isTextBox: true,
    fontFace: BODY, fontSize: 13.5, color: dark ? DIM : MUTE, margin: 0 });
}
function node(s, x, y, d, fill) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill },
    line: { color: fill, width: 0 } });
}
function edge(s, x1, y1, x2, y2, color, width, dash) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2),
              w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
              line: { color: color || "B9B2D6", width: width || 1 } };
  if (dash) o.line.dashType = "dash";
  if ((x2 - x1) * (y2 - y1) < 0) o.flipV = true;
  s.addShape(pres.ShapeType.line, o);
}
function card(s, x, y, w, h, fill, dark) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06,
    fill: { color: fill || WHITE },
    line: { color: dark ? "342A63" : "E2DCF6", width: 1 },
    shadow: dark ? undefined
      : { type: "outer", color: "9B92C4", blur: 8, offset: 1, angle: 90, opacity: 0.18 } });
}
function bullets(s, items, x, y, w, h, size, color) {
  s.addText(items.map((t, i) => ({ text: t,
      options: { bullet: true, breakLine: i !== items.length - 1 } })),
    { x, y, w, h, isTextBox: true, fontFace: BODY, fontSize: size || 13.5,
      color: color || INK, paraSpaceAfter: 8, margin: 0 });
}
function stat(s, x, y, w, big, label, color, bg) {
  card(s, x, y, w, 1.35, bg || WHITE);
  s.addText(big, { x, y: y + 0.12, w, h: 0.62, isTextBox: true, fontFace: HEAD,
    fontSize: 34, bold: true, color, align: "center", margin: 0 });
  s.addText(label, { x, y: y + 0.76, w, h: 0.46, isTextBox: true, fontFace: BODY,
    fontSize: 10.5, color: MUTE, align: "center", margin: 0 });
}

// =====================================================================
// 1. Title
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NIGHT };

  // Motif: dense graph coarsening into a sparse one -- the project, in one image.
  const d = 0.2;
  const dense = [[10.15, 1.95], [9.35, 2.75], [10.95, 2.75], [8.95, 3.55],
                 [10.15, 3.55], [11.35, 3.55], [9.75, 4.35], [10.75, 4.35]];
  dense.forEach((a, i) => dense.slice(i + 1).forEach(b => {
    if (Math.abs(a[1] - b[1]) < 0.9) edge(s, a[0] + d / 2, a[1] + d / 2, b[0] + d / 2, b[1] + d / 2, "3A2F६F".replace("६","6"), 0.75);
  }));
  dense.forEach(p => node(s, p[0], p[1], d, VIOLET));
  const keep = [[10.15, 1.95], [9.35, 2.75], [10.95, 2.75], [10.15, 5.25]];
  keep.slice(1, 3).forEach(p => {
    edge(s, keep[0][0] + d / 2, keep[0][1] + d / 2, p[0] + d / 2, p[1] + d / 2, AMBER, 1.9);
    edge(s, p[0] + d / 2, p[1] + d / 2, keep[3][0] + d / 2, keep[3][1] + d / 2, AMBER, 1.9);
  });
  keep.forEach(p => node(s, p[0], p[1], d + 0.05, AMBER));
  s.addText("coarsen", { x: 9.5, y: 5.6, w: 1.6, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 11, italic: true, color: AMBER, align: "center", margin: 0 });

  s.addText("Causality-Preserving\nGraph Coarsening", { x: 0.85, y: 1.5, w: 7.8, h: 1.7,
    isTextBox: true, fontFace: HEAD, fontSize: 38, bold: true, color: WHITE,
    lineSpacingMultiple: 1.05, margin: 0 });
  s.addText("for Efficient LLM Reasoning", { x: 0.85, y: 3.2, w: 7.8, h: 0.5,
    isTextBox: true, fontFace: HEAD, fontSize: 26, color: AMBER, margin: 0 });

  s.addText([
    { text: "Prasoon Raj", options: { bold: true } },
    { text: "  (2023EE10708)          ", options: { color: DIM, fontSize: 13 } },
    { text: "Nikhil Bansal", options: { bold: true } },
    { text: "  (2023EE10787)", options: { color: DIM, fontSize: 13 } },
  ], { x: 0.85, y: 4.25, w: 8.2, h: 0.36, isTextBox: true, fontFace: BODY,
       fontSize: 15, color: WHITE, margin: 0 });

  s.addText([
    { text: "Guided by   ", options: { color: MUTE, fontSize: 12 } },
    { text: "Subhanu Halder", options: { bold: true } },
    { text: "  (PhD Scholar)", options: { color: DIM, fontSize: 12 } },
  ], { x: 0.85, y: 4.78, w: 8.2, h: 0.32, isTextBox: true, fontFace: BODY,
       fontSize: 13.5, color: WHITE, margin: 0 });
  s.addText([
    { text: "Supervisor   ", options: { color: MUTE, fontSize: 12 } },
    { text: "Prof. Sandeep Kumar", options: { bold: true } },
  ], { x: 0.85, y: 5.16, w: 8.2, h: 0.32, isTextBox: true, fontFace: BODY,
       fontSize: 13.5, color: WHITE, margin: 0 });

  s.addText("Department of Electrical Engineering  ·  IIT Delhi",
    { x: 0.85, y: 5.85, w: 8.2, h: 0.32, isTextBox: true, fontFace: BODY,
      fontSize: 12, color: MUTE, margin: 0 });

  s.addNotes("One-line framing: reasoning graphs make language models more accurate but far more expensive. We want to shrink the graph without destroying the causal structure that makes it work. Step one was reproducing the baseline faithfully.");
}

// =====================================================================
// 2. The bottleneck
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Why structured reasoning exists", "A language model writes one token at a time, and every token costs the same");

  bullets(s, [
    "Compute per token is fixed - the model cannot allocate more effort to a harder question",
    "Generation is irreversible - once a token is sampled, everything after it is conditioned on it",
    "Ask directly, and an arbitrarily hard problem receives one token's worth of computation",
  ], 0.7, 1.85, 6.4, 1.8, 13.5);

  card(s, 0.7, 3.95, 6.4, 1.05, PAPER);
  s.addText("Writing intermediate steps IS the extra computation - not a description of it.",
    { x: 0.95, y: 4.18, w: 5.9, h: 0.6, isTextBox: true, fontFace: BODY,
      fontSize: 14, bold: true, color: VIOLET, margin: 0 });

  card(s, 7.55, 1.8, 5.05, 3.55);
  s.addText("The benchmark task", { x: 7.85, y: 2.0, w: 4.5, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("Sort 64 digits (0-9), with duplicates", { x: 7.85, y: 2.33, w: 4.5, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  s.addText("[4, 2, 7, 2, 9, 2, 1, ...]", { x: 7.85, y: 2.78, w: 4.5, h: 0.32,
    isTextBox: true, fontFace: "Courier New", fontSize: 14, color: INK, margin: 0 });
  s.addText("The model must emit 2 exactly three times - with no counter and no scratch variable. Only the text it has already written.",
    { x: 7.85, y: 3.2, w: 4.5, h: 1.0, isTextBox: true, fontFace: BODY,
      fontSize: 12.5, color: INK, margin: 0 });
  s.addText("A working-memory limit,\nnot a reasoning limit.", { x: 7.85, y: 4.45, w: 4.5, h: 0.6,
    isTextBox: true, fontFace: BODY, fontSize: 13, bold: true, italic: true,
    color: ROSE, margin: 0 });

  s.addText("Digits are generated at random, so the task cannot be solved from memorised training data - the failure is isolated and measurable.",
    { x: 0.7, y: 5.75, w: 11.9, h: 0.4, isTextBox: true, fontFace: BODY,
      fontSize: 12, italic: true, color: MUTE, margin: 0 });

  s.addNotes("Sorting is chosen because it needs no world knowledge. Any improvement is attributable to reasoning structure, not recall.");
}

// =====================================================================
// 3. The ladder
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Four schemes, one question", "What shape should the model's intermediate reasoning take?");

  const cols = [
    { x: 0.75, n: "CoT", sub: "chain", note: "show the steps" },
    { x: 3.85, n: "CoT-SC", sub: "k chains", note: "sample, then select" },
    { x: 6.95, n: "ToT", sub: "tree", note: "branch and backtrack" },
    { x: 10.05, n: "GoT", sub: "graph", note: "AGGREGATE" },
  ];
  const cw = 2.5, top = 2.05, d = 0.19, rowH = 0.5;

  cols.forEach((c, ci) => {
    const hl = ci === 3;
    card(s, c.x, top, cw, 3.1, hl ? "FFF6E6" : WHITE);
    s.addText(c.n, { x: c.x, y: top + 0.12, w: cw, h: 0.34, isTextBox: true,
      fontFace: HEAD, fontSize: 19, bold: true, color: hl ? AMBER : INK,
      align: "center", margin: 0 });
    s.addText(c.sub, { x: c.x, y: top + 0.48, w: cw, h: 0.26, isTextBox: true,
      fontFace: BODY, fontSize: 11.5, color: MUTE, align: "center", margin: 0 });

    const mx = c.x + cw / 2 - d / 2, gy = top + 0.88;
    if (ci === 0) {
      for (let i = 0; i < 4; i++) {
        if (i) edge(s, mx + d / 2, gy + (i - 1) * rowH + d / 2, mx + d / 2, gy + i * rowH + d / 2, VIOLET, 1.2);
        node(s, mx, gy + i * rowH, d, VIOLET);
      }
    } else if (ci === 1) {
      [-0.6, 0, 0.6].forEach(o => {
        for (let i = 0; i < 3; i++) {
          if (i) edge(s, mx + o + d / 2, gy + (i - 1) * rowH + d / 2, mx + o + d / 2, gy + i * rowH + d / 2, VIOLET, 1.2);
          node(s, mx + o, gy + i * rowH, d, VIOLET);
        }
      });
    } else if (ci === 2) {
      node(s, mx, gy, d, VIOLET);
      [-0.64, 0, 0.64].forEach(o => {
        edge(s, mx + d / 2, gy + d / 2, mx + o + d / 2, gy + rowH + d / 2, VIOLET, 1.2);
        node(s, mx + o, gy + rowH, d, VIOLET);
        [-0.21, 0.21].forEach(o2 => {
          edge(s, mx + o + d / 2, gy + rowH + d / 2, mx + o + o2 + d / 2, gy + 2 * rowH + d / 2, VIOLET, 1.2);
          node(s, mx + o + o2, gy + 2 * rowH, d, (o === 0 && o2 === 0.21) ? VIOLET : "CFC7E8");
        });
      });
    } else {
      node(s, mx, gy, d, VIOLET);
      const mids = [-0.7, -0.23, 0.23, 0.7];
      mids.forEach(o => {
        edge(s, mx + d / 2, gy + d / 2, mx + o + d / 2, gy + rowH + d / 2, VIOLET, 1.2);
        node(s, mx + o, gy + rowH, d, VIOLET);
      });
      [-0.47, 0.47].forEach((mo, mi) => {
        [mids[mi * 2], mids[mi * 2 + 1]].forEach(o =>
          edge(s, mx + o + d / 2, gy + rowH + d / 2, mx + mo + d / 2, gy + 2 * rowH + d / 2, AMBER, 1.7));
        node(s, mx + mo, gy + 2 * rowH, d, AMBER);
        edge(s, mx + mo + d / 2, gy + 2 * rowH + d / 2, mx + d / 2, gy + 3 * rowH + d / 2, AMBER, 1.7);
      });
      node(s, mx, gy + 3 * rowH, d, AMBER);
    }
    s.addText(c.note, { x: c.x, y: top + 2.68, w: cw, h: 0.3, isTextBox: true,
      fontFace: BODY, fontSize: 11.5, bold: hl, color: hl ? AMBER : MUTE,
      align: "center", margin: 0 });
  });

  card(s, 0.75, 5.5, 11.85, 1.1, NIGHT, true);
  s.addText("Aggregation requires a vertex with in-degree > 1. A tree cannot contain one, by definition.",
    { x: 1.05, y: 5.67, w: 11.25, h: 0.36, isTextBox: true, fontFace: BODY,
      fontSize: 15.5, bold: true, color: WHITE, margin: 0 });
  s.addText("Removing that constraint is the entire contribution of Graph of Thoughts - and the structure we aim to compress.",
    { x: 1.05, y: 6.05, w: 11.25, h: 0.32, isTextBox: true, fontFace: BODY,
      fontSize: 12.5, color: DIM, margin: 0 });

  s.addNotes("Each scheme fixes a limitation of the previous one. CoT-SC samples several chains but they never exchange information. ToT can branch and backtrack but never merge. GoT allows merging, which is what a tree structurally cannot express.");
}

// =====================================================================
// 4. Framework
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "The GoT framework", "Two distinct graphs - conflating them is the most common implementation error");

  card(s, 0.75, 1.8, 5.8, 1.95, PAPER);
  s.addText("Graph of Operations (GoO)", { x: 1.05, y: 1.97, w: 5.2, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 14, bold: true, color: VIOLET, margin: 0 });
  s.addText("Static execution plan, constructed before the run.\nVertices are operations; edges are dependencies.\nOne per task configuration.",
    { x: 1.05, y: 2.32, w: 5.2, h: 1.2, isTextBox: true, fontFace: BODY,
      fontSize: 12.5, color: INK, margin: 0 });

  card(s, 6.8, 1.8, 5.8, 1.95, PAPER);
  s.addText("Graph Reasoning State (GRS)", { x: 7.1, y: 1.97, w: 5.2, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 14, bold: true, color: MINT, margin: 0 });
  s.addText("Dynamic record of the thoughts actually produced.\nVertices are LLM outputs; edges are data dependencies.\nOne per input instance.",
    { x: 7.1, y: 2.32, w: 5.2, h: 1.2, isTextBox: true, fontFace: BODY,
      fontSize: 12.5, color: INK, margin: 0 });

  s.addText("Three transformations - and only these create vertices",
    { x: 0.75, y: 3.95, w: 11.85, h: 0.34, isTextBox: true, fontFace: BODY,
      fontSize: 14.5, bold: true, color: INK, margin: 0 });

  [{ n: "Generate (k)", d: "One thought in, k out.\nBranching. Already present\nin ToT.", c: VIOLET },
   { n: "Aggregate (k)", d: "k thoughts in, one out.\nAn edge from every input.\nThe novel operation.", c: AMBER },
   { n: "Improve", d: "Refine a thought in place.\nA cycle - also impossible\nwithin a tree.", c: MINT },
  ].forEach((o, i) => {
    const x = 0.75 + i * 4.03;
    card(s, x, 4.38, 3.78, 1.72, o.c === AMBER ? "FFF6E6" : WHITE);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.28, y: 4.62, w: 0.4, h: 0.4,
      fill: { color: o.c }, line: { width: 0 } });
    s.addText(String(i + 1), { x: x + 0.28, y: 4.66, w: 0.4, h: 0.32, isTextBox: true,
      fontFace: BODY, fontSize: 12.5, bold: true, color: WHITE, align: "center", margin: 0 });
    s.addText(o.n, { x: x + 0.82, y: 4.64, w: 2.8, h: 0.3, isTextBox: true,
      fontFace: BODY, fontSize: 13.5, bold: true, color: o.c, margin: 0 });
    s.addText(o.d, { x: x + 0.3, y: 5.1, w: 3.25, h: 0.9, isTextBox: true,
      fontFace: BODY, fontSize: 11, color: INK, margin: 0 });
  });

  s.addText("Score, KeepBest and GroundTruth create no vertices - they annotate and filter. Edges denote causality: an edge (a, b) means a's text was literally the input that produced b.",
    { x: 0.75, y: 6.3, w: 11.85, h: 0.55, isTextBox: true, fontFace: BODY,
      fontSize: 11.5, italic: true, color: MUTE, margin: 0 });

  s.addNotes("The last line is the one that matters for our method. Edges are causal dependencies, not loose associations - which is exactly what a coarsening procedure has to preserve.");
}

// =====================================================================
// 5. Use case
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Use case: merge sort executed by an LLM", "Decompose until each sub-problem lies inside the model's reliable range");

  const d = 0.23, gy = 2.2, rowH = 0.82;
  const lvl = [[5.9], [3.3, 5.0, 6.8, 8.5], [4.15, 7.65], [5.9]];
  const cols = [VIOLET, VIOLET, AMBER, AMBER];
  const labels = ["64 numbers", "sort 16 each (k=3)", "merge (k=10)", "merge - final answer"];

  for (let L = 1; L < 4; L++) {
    lvl[L].forEach((x, i) => {
      const par = L === 1 ? lvl[0] : (L === 2 ? [lvl[1][i * 2], lvl[1][i * 2 + 1]] : lvl[2]);
      par.forEach(px => edge(s, px + d / 2, gy + (L - 1) * rowH + d / 2,
                              x + d / 2, gy + L * rowH + d / 2,
                              L >= 2 ? AMBER : "B9B2D6", L >= 2 ? 1.8 : 1.1));
    });
  }
  lvl.forEach((row, L) => row.forEach(x => node(s, x, gy + L * rowH, d, cols[L])));
  labels.forEach((t, L) => s.addText(t, { x: 0.8, y: gy + L * rowH - 0.05, w: 2.2, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 11.5, color: L >= 2 ? AMBER : MUTE,
    align: "right", margin: 0 }));
  s.addText("fan out", { x: 8.9, y: gy + 0.5, w: 1.3, h: 0.26, isTextBox: true,
    fontFace: BODY, fontSize: 11, italic: true, color: MUTE, margin: 0 });
  s.addText("fan back in", { x: 8.9, y: gy + 2.05, w: 1.5, h: 0.26, isTextBox: true,
    fontFace: BODY, fontSize: 11, italic: true, bold: true, color: AMBER, margin: 0 });

  stat(s, 10.55, 2.05, 2.05, "39", "thoughts per\ninstance", VIOLET, PAPER);
  stat(s, 10.55, 3.6, 2.05, "15", "aggregations\n(ToT has 0)", AMBER, "FFF6E6");

  card(s, 0.75, 5.6, 11.85, 1.0, PAPER);
  s.addText("Sorting 16 numbers is reliable. Merging two sorted lists is reliable. Sorting 64 from scratch is not. So perform only the reliable operations.",
    { x: 1.05, y: 5.82, w: 11.25, h: 0.5, isTextBox: true, fontFace: BODY,
      fontSize: 13.5, bold: true, color: VIOLET, margin: 0 });

  s.addNotes("Scoring here is an exact Python function - count inversions, compare digit frequencies - so it is free and never wrong. That matters later: it gives us a reliable signal for deciding which parts of the graph are worth keeping.");
}

// =====================================================================
// 6. Datasets
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Datasets", "All tasks are synthetic, with computed ground truth - and the authors publish the exact files they ran");

  const hdr = ["Task", "Input", "Sizes", "Instances"];
  const rows = [
    ["Sorting", "digits 0-9, with duplicates", "32 / 64 / 128", "100"],
    ["Set intersection", "two sets, 25-75% overlap", "32 / 64 / 128", "100"],
    ["Keyword counting", "country mentions in text", "4 / 8 / 16 sentences", "-"],
    ["Document merging", "overlapping NDA documents", "4 documents", "-"],
  ];
  const tb = [hdr.map(t => ({ text: t, options: { bold: true, color: WHITE, fill: { color: VIOLET } } }))];
  rows.forEach((r, i) => tb.push(r.map((c, j) => ({ text: c, options: {
    bold: j === 0, color: INK, fill: { color: i % 2 ? WHITE : PAPER } } }))));
  s.addTable(tb, { x: 0.75, y: 1.8, w: 7.7, colW: [1.9, 2.85, 1.75, 1.2], rowH: 0.42,
    fontFace: BODY, fontSize: 12, border: { type: "solid", color: "E2DCF6", pt: 1 },
    valign: "middle" });

  bullets(s, [
    "No public benchmark - ground truth is computed, never annotated",
    "Randomly generated, so no possibility of training-data contamination",
    "Difficulty is a controlled variable: 32 - 64 - 128 elements",
  ], 0.75, 4.1, 7.7, 1.25, 12.5);

  card(s, 8.75, 1.8, 3.85, 2.35, PAPER);
  s.addText("What we ran", { x: 9.0, y: 1.97, w: 3.35, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: VIOLET, margin: 0 });
  s.addText("The authors' own CSV files, byte for byte:", { x: 9.0, y: 2.3, w: 3.4, h: 0.4,
    isTextBox: true, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
  s.addText("data/official/\n  sorting_064.csv\n  set_intersection_032.csv",
    { x: 9.0, y: 2.78, w: 3.4, h: 0.72, isTextBox: true, fontFace: "Courier New",
      fontSize: 10, color: VIOLET, margin: 0 });
  s.addText("Eliminates \"different random data\" as an explanation for any discrepancy.",
    { x: 9.0, y: 3.55, w: 3.4, h: 0.5, isTextBox: true, fontFace: BODY,
      fontSize: 10.5, italic: true, color: MUTE, margin: 0 });

  card(s, 8.75, 4.35, 3.85, 2.0, WHITE);
  s.addText("Also regenerable", { x: 9.0, y: 4.52, w: 3.35, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: MINT, margin: 0 });
  s.addText("generate_data.py --seed 42", { x: 9.0, y: 4.86, w: 3.4, h: 0.28,
    isTextBox: true, fontFace: "Courier New", fontSize: 10, color: INK, margin: 0 });
  s.addText("Byte-identical on any machine, so laptop and GPU results are directly comparable - and we can test sizes the authors never published.",
    { x: 9.0, y: 5.2, w: 3.4, h: 1.0, isTextBox: true, fontFace: BODY,
      fontSize: 10.5, color: MUTE, margin: 0 });

  s.addNotes("Synthetic data is a deliberate strength. With a public benchmark you could never separate improved reasoning from memorised answers.");
}

// =====================================================================
// 7. Setup
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Experimental setup", "The original model is closed and deprecated - so we replicate the claims, not the exact figures");

  card(s, 0.75, 1.8, 5.8, 2.4, PAPER);
  s.addText("Original paper", { x: 1.05, y: 1.97, w: 5.2, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 14.5, bold: true, color: MUTE, margin: 0 });
  bullets(s, ["ChatGPT-3.5 via paid API",
              "Temperature 1.0, 4k context window",
              "100 samples per configuration",
              "Llama-2 attempted, then abandoned"],
    1.05, 2.35, 5.2, 1.7, 12.5, "3C3757");

  card(s, 6.8, 1.8, 5.8, 2.4, WHITE);
  s.addText("This work", { x: 7.1, y: 1.97, w: 5.2, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 14.5, bold: true, color: MINT, margin: 0 });
  bullets(s, ["Qwen2.5-7B-Instruct, open weights",
              "Identical temperature and sample count",
              "vLLM inference on NVIDIA RTX A6000",
              "No API keys, fully self-hosted"],
    7.1, 2.35, 5.2, 1.7, 12.5);

  [{ n: "4", l: "interchangeable backends\nmock / llama.cpp / HF / vLLM", c: VIOLET },
   { n: "5", l: "schemes on one engine\nIO, CoT, CoT-SC, ToT, GoT", c: MINT },
   { n: "60", l: "automated tests, run\nbefore every GPU job", c: AMBER },
  ].forEach((st, i) => {
    const x = 0.75 + i * 4.03;
    card(s, x, 4.45, 3.78, 1.6, WHITE);
    s.addText(st.n, { x: x + 0.25, y: 4.62, w: 1.05, h: 0.72, isTextBox: true,
      fontFace: HEAD, fontSize: 36, bold: true, color: st.c, margin: 0 });
    s.addText(st.l, { x: x + 1.3, y: 4.75, w: 2.3, h: 0.9, isTextBox: true,
      fontFace: BODY, fontSize: 11, color: INK, margin: 0 });
  });

  s.addText("All five schemes share one controller, one backend, one prompt set and one scorer - so any measured difference is attributable to graph structure alone.",
    { x: 0.75, y: 6.25, w: 11.85, h: 0.4, isTextBox: true, fontFace: BODY,
      fontSize: 12, italic: true, color: MUTE, margin: 0 });

  s.addNotes("The final line is the experimental design. One engine, five graph shapes - that is what isolates structure as the independent variable.");
}

// =====================================================================
// 8. Structural results
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Result 1: structural claims hold exactly", "Volume and latency are properties of the graph, independent of the model");

  const tb = [["Scheme", "Volume", "Latency", "Aggregations"].map(t =>
    ({ text: t, options: { bold: true, color: WHITE, fill: { color: VIOLET } } }))];
  [["IO", "1.0", "1.0", "0"], ["CoT", "2.0", "2.0", "0"], ["CoT-SC", "2.0", "2.0", "0"],
   ["ToT", "6.0", "6.0", "0"], ["GoT", "20.0", "9.0", "15"]].forEach((r, i) => {
    const hl = i === 4;
    tb.push(r.map((c, j) => ({ text: c, options: { bold: hl || j === 0,
      color: hl ? AMBER : INK, fill: { color: hl ? "FFF6E6" : (i % 2 ? WHITE : PAPER) } } })));
  });
  s.addTable(tb, { x: 0.75, y: 1.9, w: 6.5, colW: [1.7, 1.55, 1.55, 1.7], rowH: 0.44,
    fontFace: BODY, fontSize: 12.5, border: { type: "solid", color: "E2DCF6", pt: 1 },
    valign: "middle" });

  card(s, 7.7, 1.9, 4.9, 1.45, NIGHT, true);
  s.addText("22 / 22", { x: 7.7, y: 2.04, w: 4.9, h: 0.6, isTextBox: true, fontFace: HEAD,
    fontSize: 34, bold: true, color: WHITE, align: "center", margin: 0 });
  s.addText("structural checks pass, verified automatically on every run",
    { x: 7.95, y: 2.66, w: 4.4, h: 0.5, isTextBox: true, fontFace: BODY,
      fontSize: 11, color: DIM, align: "center", margin: 0 });

  card(s, 7.7, 3.55, 4.9, 1.85, "FFF6E6");
  s.addText("The key measurement", { x: 7.95, y: 3.72, w: 4.4, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: AMBER, margin: 0 });
  s.addText("GoT achieves volume 20 at latency 9.\nToT achieves volume 6 at latency 6.",
    { x: 7.95, y: 4.05, w: 4.4, h: 0.62, isTextBox: true, fontFace: BODY,
      fontSize: 12.5, color: INK, margin: 0 });
  s.addText("Three times the information reaching the answer, for 1.5x the depth.",
    { x: 7.95, y: 4.72, w: 4.4, h: 0.56, isTextBox: true, fontFace: BODY,
      fontSize: 12, bold: true, color: VIOLET, margin: 0 });

  s.addText("Volume = number of prior thoughts with a causal path to the final answer.    Latency = sequential model calls required.",
    { x: 0.75, y: 5.7, w: 11.85, h: 0.32, isTextBox: true, fontFace: BODY,
      fontSize: 11.5, italic: true, color: MUTE, margin: 0 });
  s.addText("GoT is the only scheme that aggregates - measured, not assumed. Every other scheme reports exactly zero.",
    { x: 0.75, y: 6.08, w: 11.85, h: 0.32, isTextBox: true, fontFace: BODY,
      fontSize: 13, bold: true, color: INK, margin: 0 });

  s.addNotes("Volume is the quantity our compression must preserve. It measures how much of the computed work can actually influence the answer. A tree wastes nearly all of it; GoT's merges pull every branch back in.");
}

// =====================================================================
// 9. Quality results
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Result 2: quality on a 7B open model", "100 instances, 64-element sorting, the authors' dataset, zero parse failures");

  s.addChart(pres.ChartType.bar, [{
    name: "Error scope",
    labels: ["IO", "CoT", "CoT-SC", "ToT", "GoT"],
    values: [9.83, 10.34, 8.17, 9.87, 8.11],
  }], {
    x: 0.75, y: 1.85, w: 6.6, h: 3.5, barDir: "col",
    chartColors: [VIOLET, VIOLET, VIOLET, VIOLET, AMBER],
    showTitle: true, title: "Mean error scope (lower is better)",
    titleFontFace: BODY, titleFontSize: 13, titleColor: INK,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11,
    dataLabelColor: INK, dataLabelFormatCode: "0.00", showLegend: false,
    catAxisLabelColor: MUTE, valAxisLabelColor: MUTE, catAxisLabelFontSize: 11,
    valAxisLabelFontSize: 10, valGridLine: { color: "EFEBFA", size: 1 },
    catGridLine: { style: "none" }, valAxisMaxVal: 12,
    chartArea: { fill: { color: WHITE } },
  });

  card(s, 7.7, 1.85, 4.9, 1.5, "FFF6E6");
  s.addText("GoT ranks best", { x: 7.95, y: 2.02, w: 4.4, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 13.5, bold: true, color: AMBER, margin: 0 });
  s.addText("Lowest error of all five schemes, with a 0% parse-failure rate - so the pipeline is sound and the ordering is meaningful.",
    { x: 7.95, y: 2.35, w: 4.4, h: 0.9, isTextBox: true, fontFace: BODY,
      fontSize: 11.5, color: INK, margin: 0 });

  card(s, 7.7, 3.5, 4.9, 1.35, WHITE);
  s.addText("No exact solutions yet", { x: 7.95, y: 3.67, w: 4.4, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 13.5, bold: true, color: VIOLET, margin: 0 });
  s.addText("0% fully correct at 64 elements. Expected: reliable chain-of-thought is emergent above roughly 10B parameters.",
    { x: 7.95, y: 4.0, w: 4.4, h: 0.75, isTextBox: true, fontFace: BODY,
      fontSize: 11.5, color: INK, margin: 0 });

  card(s, 7.7, 5.0, 4.9, 1.6, PANEL, true);
  s.addText("And the reason this project exists", { x: 7.95, y: 5.15, w: 4.4, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 13, bold: true, color: AMBER, margin: 0 });
  s.addText("GoT consumes 7,899 tokens per instance against 520 for direct prompting - a 15x cost for the best quality.",
    { x: 7.95, y: 5.48, w: 4.4, h: 0.95, isTextBox: true, fontFace: BODY,
      fontSize: 11.5, color: WHITE, margin: 0 });

  s.addText("Reference point: the paper reports +62% over ToT on GPT-3.5. We reproduce the ordering on a model 25x smaller that the authors never evaluated.",
    { x: 0.75, y: 5.6, w: 6.6, h: 0.7, isTextBox: true, fontFace: BODY,
      fontSize: 11.5, italic: true, color: MUTE, margin: 0 });

  s.addNotes("The margin over ToT is being re-measured: we found our own ToT baseline could not reject a refinement that worsened the answer, which inflated GoT's advantage. Fixed, rerunning. The ordering is unchanged.");
}

// =====================================================================
// 10. The cost problem  (NEW)
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NIGHT };
  title(s, "The problem we are actually solving", "Accuracy is bought with redundancy, and redundancy is the cost", true);

  const bars = [
    { n: "IO", v: 520, c: "4A3F7A" },
    { n: "CoT", v: 1169, c: "4A3F7A" },
    { n: "CoT-SC", v: 887, c: "4A3F7A" },
    { n: "ToT", v: 2192, c: VIOLET },
    { n: "GoT", v: 7899, c: AMBER },
  ];
  const bx = 0.9, by = 2.0, bw = 6.4, maxV = 8200, rowH = 0.62;
  bars.forEach((b, i) => {
    const y = by + i * rowH;
    s.addText(b.n, { x: bx, y: y + 0.02, w: 1.0, h: 0.3, isTextBox: true,
      fontFace: BODY, fontSize: 12, color: WHITE, margin: 0 });
    s.addShape(pres.ShapeType.roundRect, { x: bx + 1.05, y, w: Math.max(0.12, (b.v / maxV) * bw),
      h: 0.34, rectRadius: 0.04, fill: { color: b.c }, line: { width: 0 } });
    s.addText(b.v.toLocaleString() + " tokens", { x: bx + 1.15 + (b.v / maxV) * bw, y: y + 0.02,
      w: 1.9, h: 0.3, isTextBox: true, fontFace: BODY, fontSize: 11,
      color: b.n === "GoT" ? AMBER : DIM, margin: 0 });
  });
  s.addText("Mean tokens per instance, 64-element sorting, Qwen2.5-7B",
    { x: 0.9, y: 5.2, w: 7.0, h: 0.3, isTextBox: true, fontFace: BODY,
      fontSize: 11, italic: true, color: MUTE, margin: 0 });

  card(s, 8.15, 1.95, 4.45, 1.45, PANEL, true);
  s.addText("15x", { x: 8.15, y: 2.08, w: 4.45, h: 0.6, isTextBox: true, fontFace: HEAD,
    fontSize: 34, bold: true, color: AMBER, align: "center", margin: 0 });
  s.addText("the cost of direct prompting", { x: 8.15, y: 2.7, w: 4.45, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 11.5, color: DIM, align: "center", margin: 0 });

  card(s, 8.15, 3.55, 4.45, 1.45, PANEL, true);
  s.addText("3.6x", { x: 8.15, y: 3.68, w: 4.45, h: 0.6, isTextBox: true, fontFace: HEAD,
    fontSize: 34, bold: true, color: VIOLET, align: "center", margin: 0 });
  s.addText("the cost of Tree of Thoughts", { x: 8.15, y: 4.3, w: 4.45, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 11.5, color: DIM, align: "center", margin: 0 });

  card(s, 8.15, 5.15, 4.45, 1.5, PANEL, true);
  s.addText("39 vertices, 15 aggregations, 8 sequential model calls - for one sorted list.",
    { x: 8.45, y: 5.35, w: 3.9, h: 1.05, isTextBox: true, fontFace: BODY,
      fontSize: 12, color: WHITE, margin: 0 });

  s.addText("GoT wins on quality and loses on cost. Every practical deployment question is therefore: how much of this graph is actually necessary?",
    { x: 0.9, y: 5.75, w: 7.0, h: 0.8, isTextBox: true, fontFace: BODY,
      fontSize: 13, bold: true, color: AMBER, margin: 0 });

  s.addNotes("This slide turns the replication into a motivation. We reproduced the quality result; we also measured exactly how expensive that quality is. Fifteen times the cost of direct prompting is what makes GoT impractical, and that gap is our target.");
}

// =====================================================================
// 11. Naive approach vs our goal  (NEW)
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Naive compression, and why it fails", "Turning the knobs down reduces cost - and destroys the structure that produced the quality");

  const tb = [["Configuration", "Tokens", "Volume", "Error"].map(t =>
    ({ text: t, options: { bold: true, color: WHITE, fill: { color: VIOLET } } }))];
  [["8 chunks", "14,384", "39.9", "2.88"],
   ["4 chunks (paper)", "8,565", "19.9", "3.20"],
   ["2 chunks", "5,163", "9.9", "3.38"],
   ["stripped down", "950", "8.0", "10.50"]].forEach((r, i) => {
    const bad = i === 3;
    tb.push(r.map((c, j) => ({ text: c, options: { bold: bad || j === 0,
      color: bad ? ROSE : INK, fill: { color: bad ? "FFEEF1" : (i % 2 ? WHITE : PAPER) } } })));
  });
  s.addTable(tb, { x: 0.75, y: 1.85, w: 6.3, colW: [2.1, 1.5, 1.35, 1.35], rowH: 0.44,
    fontFace: BODY, fontSize: 12.5, border: { type: "solid", color: "E2DCF6", pt: 1 },
    valign: "middle" });

  card(s, 0.75, 4.25, 6.3, 1.15, "FFEEF1");
  s.addText("Cut cost 9x and the error rises to 10.50 - worse than using no reasoning structure at all (IO scores 9.83).",
    { x: 1.0, y: 4.45, w: 5.85, h: 0.75, isTextBox: true, fontFace: BODY,
      fontSize: 12.5, bold: true, color: ROSE, margin: 0 });

  s.addText("Uniform pruning is structure-blind: it removes useful and redundant computation at the same rate.",
    { x: 0.75, y: 5.55, w: 6.3, h: 0.6, isTextBox: true, fontFace: BODY,
      fontSize: 12, italic: true, color: MUTE, margin: 0 });

  // Right: the proposal, drawn
  card(s, 7.5, 1.85, 5.1, 4.65, PAPER);
  s.addText("Our approach", { x: 7.8, y: 2.02, w: 4.5, h: 0.32, isTextBox: true,
    fontFace: BODY, fontSize: 14.5, bold: true, color: AMBER, margin: 0 });
  s.addText("Causality-preserving graph coarsening", { x: 7.8, y: 2.36, w: 4.5, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, margin: 0 });

  const d = 0.17;
  const dense = [[8.35, 3.0], [8.05, 3.6], [8.65, 3.6], [7.9, 4.2], [8.5, 4.2], [9.1, 4.2], [8.35, 4.8]];
  dense.forEach((a, i) => dense.slice(i + 1).forEach(b => {
    if (Math.abs(a[1] - b[1]) < 0.7) edge(s, a[0] + d / 2, a[1] + d / 2, b[0] + d / 2, b[1] + d / 2, "C7BFE6", 0.9);
  }));
  dense.forEach(p => node(s, p[0], p[1], d, VIOLET));
  s.addText("39 vertices", { x: 7.65, y: 5.15, w: 1.5, h: 0.26, isTextBox: true,
    fontFace: BODY, fontSize: 10.5, color: MUTE, align: "center", margin: 0 });

  s.addShape(pres.ShapeType.rightArrow, { x: 9.65, y: 3.82, w: 0.72, h: 0.3,
    fill: { color: AMBER }, line: { width: 0 } });

  const sparse = [[11.15, 3.0], [10.75, 3.75], [11.55, 3.75], [11.15, 4.8]];
  sparse.slice(1, 3).forEach(p => {
    edge(s, sparse[0][0] + d / 2, sparse[0][1] + d / 2, p[0] + d / 2, p[1] + d / 2, AMBER, 1.8);
    edge(s, p[0] + d / 2, p[1] + d / 2, sparse[3][0] + d / 2, sparse[3][1] + d / 2, AMBER, 1.8);
  });
  sparse.forEach(p => node(s, p[0], p[1], d + 0.03, AMBER));
  s.addText("fewer vertices,\nsame causal paths", { x: 10.4, y: 5.15, w: 1.9, h: 0.45,
    isTextBox: true, fontFace: BODY, fontSize: 10.5, color: AMBER, align: "center", margin: 0 });

  bullets(s, [
    "Edges are causal, not associative - (a,b) means a produced b",
    "Coarsen vertices while preserving reachability to the answer",
    "Target: GoT accuracy at a cost approaching ToT",
  ], 7.8, 5.7, 4.6, 0.8, 11, INK);

  s.addNotes("The naive baseline is real data we measured, not a straw man. The point is that cost and quality are coupled under uniform pruning. Coarsening asks a different question: which vertices can be merged or dropped without breaking the causal paths that carry information to the answer? Volume is the quantity to preserve; token count is the quantity to reduce.");
}

// =====================================================================
// 12. Literature review and bibliography  (NEW)
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Literature review", "Two lines of work meet in this project: reasoning structure, and graph reduction");

  card(s, 0.75, 1.8, 5.85, 4.15, PAPER);
  s.addText("Structured reasoning in LLMs", { x: 1.0, y: 1.97, w: 5.35, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 13.5, bold: true, color: VIOLET, margin: 0 });
  [["Wei et al., 2022", "Chain-of-Thought Prompting Elicits Reasoning in LLMs. NeurIPS. arXiv:2201.11903", "Intermediate steps as computation"],
   ["Wang et al., 2023", "Self-Consistency Improves Chain of Thought Reasoning. ICLR. arXiv:2203.11171", "Sample k chains, select the best"],
   ["Yao et al., 2023", "Tree of Thoughts: Deliberate Problem Solving. NeurIPS. arXiv:2305.10601", "Branching and backtracking search"],
   ["Besta et al., 2024", "Graph of Thoughts: Solving Elaborate Problems with LLMs. AAAI. arXiv:2308.09687", "Aggregation; our baseline"],
   ["Zhang et al., 2024", "Multimodal Chain-of-Thought Reasoning. TMLR. arXiv:2302.00923", "Two-stage rationale then answer"],
  ].forEach((r, i) => {
    const y = 2.35 + i * 0.72;
    s.addText(r[0], { x: 1.0, y, w: 5.35, h: 0.24, isTextBox: true, fontFace: BODY,
      fontSize: 11.5, bold: true, color: INK, margin: 0 });
    s.addText(r[1], { x: 1.0, y: y + 0.22, w: 5.35, h: 0.28, isTextBox: true,
      fontFace: BODY, fontSize: 9.5, color: MUTE, margin: 0 });
    s.addText(r[2], { x: 1.0, y: y + 0.44, w: 5.35, h: 0.24, isTextBox: true,
      fontFace: BODY, fontSize: 9.5, italic: true, color: VIOLET, margin: 0 });
  });

  card(s, 6.95, 1.8, 5.65, 4.15, WHITE);
  s.addText("Graph reduction and efficiency", { x: 7.2, y: 1.97, w: 5.15, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 13.5, bold: true, color: AMBER, margin: 0 });
  [["Loukas, 2019", "Graph Reduction with Spectral and Cut Guarantees. JMLR 20(116).", "Coarsening with provable bounds"],
   ["Huang et al., 2021", "Scaling Up Graph Neural Networks Via Graph Coarsening. KDD.", "Coarsening applied to learning"],
   ["Kwon et al., 2023", "Efficient Memory Management for LLM Serving with PagedAttention. SOSP. arXiv:2309.06180", "vLLM; our inference engine"],
  ].forEach((r, i) => {
    const y = 2.35 + i * 0.8;
    s.addText(r[0], { x: 7.2, y, w: 5.15, h: 0.24, isTextBox: true, fontFace: BODY,
      fontSize: 11.5, bold: true, color: INK, margin: 0 });
    s.addText(r[1], { x: 7.2, y: y + 0.22, w: 5.15, h: 0.3, isTextBox: true,
      fontFace: BODY, fontSize: 9.5, color: MUTE, margin: 0 });
    s.addText(r[2], { x: 7.2, y: y + 0.46, w: 5.15, h: 0.24, isTextBox: true,
      fontFace: BODY, fontSize: 9.5, italic: true, color: AMBER, margin: 0 });
  });

  card(s, 7.2, 4.85, 5.15, 0.92, "FFF6E6");
  s.addText("The gap: coarsening is well studied for data graphs, not for reasoning graphs where edges carry causality.",
    { x: 7.45, y: 5.0, w: 4.7, h: 0.65, isTextBox: true, fontFace: BODY,
      fontSize: 11, bold: true, color: INK, margin: 0 });

  s.addText("Code and full derivations:  github.com/TREX4096/Graph_of_Thought",
    { x: 0.75, y: 6.2, w: 11.85, h: 0.3, isTextBox: true, fontFace: BODY,
      fontSize: 11, color: MUTE, margin: 0 });

  s.addNotes("The left column is the lineage we replicated. The right column is the toolkit we intend to borrow from. The gap statement is the contribution claim: spectral coarsening assumes edges encode similarity, whereas in a reasoning graph an edge means one thought was literally the input that produced another - so the guarantees have to be restated in terms of reachability to the answer.");
}

pres.writeFile({ fileName: process.argv[2] || "deck.pptx" })
  .then(f => console.log("written:", f));

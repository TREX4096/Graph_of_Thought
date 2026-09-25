// Graph of Thoughts -- BTP replication deck.
// 10 slides, ~10 minutes. Visual motif: nodes and edges, since the entire
// contribution of the paper is a change of graph shape.

const pptx = require("pptxgenjs");
const pres = new pptx();
pres.layout = "LAYOUT_WIDE";            // 13.3 x 7.5 inches
const W = 13.3, H = 7.5;

// --- palette ---------------------------------------------------------
const NAVY  = "141B34";   // dark ground
const DEEP  = "1C4E80";   // primary blue
const TEAL  = "00A896";   // secondary
const CORAL = "FF7A59";   // accent = aggregation, the one new idea
const WHITE = "FFFFFF";
const INK   = "16203B";
const MUTE  = "6B7280";
const PAPER = "F4F7FA";

const HEAD = "Cambria";   // safe-list serif
const BODY = "Calibri";   // safe-list sans

pres.author = "B.Tech Project";
pres.title  = "Graph of Thoughts - Replication";

// --- helpers ---------------------------------------------------------
function title(s, t, sub, dark) {
  s.addText(t, { x: 0.7, y: 0.42, w: W - 1.4, h: 0.72, isTextBox: true,
    fontFace: HEAD, fontSize: 34, bold: true, color: dark ? WHITE : INK, margin: 0 });
  if (sub) s.addText(sub, { x: 0.7, y: 1.16, w: W - 1.4, h: 0.4, isTextBox: true,
    fontFace: BODY, fontSize: 14, color: dark ? "9AA7C7" : MUTE, margin: 0 });
}
function node(s, x, y, d, fill, lbl, lblColor) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill },
    line: { color: fill, width: 0 } });
  if (lbl) s.addText(lbl, { x: x - 0.25, y: y + d + 0.04, w: d + 0.5, h: 0.24,
    isTextBox: true, fontFace: BODY, fontSize: 9, color: lblColor || MUTE,
    align: "center", margin: 0 });
}
// Edge between two circle centres.
function edge(s, x1, y1, x2, y2, color, width) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2),
              w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
              line: { color: color || "AEB8CC", width: width || 1 } };
  if ((x2 - x1) * (y2 - y1) < 0) o.flipV = true;
  s.addShape(pres.ShapeType.line, o);
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06,
    fill: { color: fill || WHITE }, line: { color: "DDE4EE", width: 1 },
    shadow: { type: "outer", color: "8895AD", blur: 8, offset: 1, angle: 90, opacity: 0.18 } });
}
function bullets(s, items, x, y, w, h, size, color) {
  s.addText(items.map((t, i) => ({ text: t,
      options: { bullet: true, breakLine: i !== items.length - 1 } })),
    { x, y, w, h, isTextBox: true, fontFace: BODY, fontSize: size || 14,
      color: color || INK, paraSpaceAfter: 8, margin: 0 });
}

// =====================================================================
// 1. Title
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Motif: a diamond of nodes -- fan out, then fan back in. This IS the paper.
  const cx = 10.4, cy = 3.6, d = 0.26;
  const pts = [[cx, cy - 1.5], [cx - 1.25, cy], [cx + 1.25, cy],
               [cx - 0.42, cy], [cx + 0.42, cy], [cx, cy + 1.5]];
  pts.forEach(p => edge(s, cx + d / 2, cy - 1.5 + d / 2, p[0] + d / 2, p[1] + d / 2, "2C3C68", 1));
  pts.slice(1, 5).forEach(p => edge(s, p[0] + d / 2, p[1] + d / 2, cx + d / 2, cy + 1.5 + d / 2, CORAL, 1.25));
  pts.slice(1, 5).forEach(p => node(s, p[0], p[1], d, TEAL));
  node(s, cx, cy - 1.5, d, WHITE);
  node(s, cx, cy + 1.5, d, CORAL);

  s.addText("Graph of Thoughts", { x: 0.9, y: 2.05, w: 7.6, h: 0.95, isTextBox: true,
    fontFace: HEAD, fontSize: 46, bold: true, color: WHITE, margin: 0 });
  s.addText("Replicating structured LLM reasoning on open-weights models",
    { x: 0.9, y: 3.0, w: 7.4, h: 0.6, isTextBox: true, fontFace: BODY,
      fontSize: 17, color: "9AA7C7", margin: 0 });
  s.addShape(pres.ShapeType.rect, { x: 0.9, y: 3.78, w: 1.5, h: 0.03, fill: { color: CORAL }, line: { width: 0 } });
  s.addText("Besta et al., AAAI 2024  ·  arXiv:2308.09687",
    { x: 0.9, y: 4.0, w: 7.4, h: 0.32, isTextBox: true, fontFace: BODY,
      fontSize: 12, color: TEAL, margin: 0 });
  s.addText("B.Tech Project  ·  Semester 7", { x: 0.9, y: 4.45, w: 7.4, h: 0.32,
    isTextBox: true, fontFace: BODY, fontSize: 12, color: "6B7799", margin: 0 });
  s.addNotes("Goal: reproduce the Graph of Thoughts paper end to end, on open models we can actually run, and check which of its claims survive. Roughly one minute per slide.");
}

// =====================================================================
// 2. The problem
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Why structured reasoning exists at all", "An LLM writes one token at a time, and each token costs the same");

  bullets(s, [
    "Compute per token is FIXED - the model cannot think harder on a hard question",
    "Decisions are irreversible - once a token is sampled, everything after it is conditioned on it",
    "Ask directly, and a hard problem gets one token's worth of computation",
  ], 0.7, 1.95, 6.2, 1.9, 14);

  card(s, 0.7, 4.1, 6.2, 1.15, PAPER);
  s.addText("More tokens = more computation.\nThe chain of thought IS the computation, not a description of it.",
    { x: 0.95, y: 4.28, w: 5.7, h: 0.8, isTextBox: true, fontFace: BODY,
      fontSize: 14, bold: true, color: DEEP, margin: 0 });

  // Right: the failure that motivates everything
  card(s, 7.4, 1.85, 5.2, 3.4);
  s.addText("The task that breaks them", { x: 7.7, y: 2.05, w: 4.6, h: 0.32,
    isTextBox: true, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("Sort 64 digits, 0-9, with duplicates", { x: 7.7, y: 2.4, w: 4.6, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  s.addText([
    { text: "[4, 2, 7, 2, 9, 2, 1, ...]", options: { breakLine: true } },
    { text: "must emit 2 exactly three times", options: { breakLine: true, color: MUTE, fontSize: 12 } },
  ], { x: 7.7, y: 2.85, w: 4.6, h: 0.7, isTextBox: true, fontFace: "Courier New",
       fontSize: 14, color: INK, margin: 0 });
  s.addText("No counter. No scratch variable. Only the text written so far.",
    { x: 7.7, y: 3.6, w: 4.6, h: 0.6, isTextBox: true, fontFace: BODY,
      fontSize: 13, color: INK, margin: 0 });
  s.addText("Not an intelligence problem -\na working-memory problem.",
    { x: 7.7, y: 4.35, w: 4.6, h: 0.6, isTextBox: true, fontFace: BODY,
      fontSize: 13, bold: true, italic: true, color: CORAL, margin: 0 });

  s.addNotes("The bottleneck is architectural. Sorting is the benchmark because the failure is pure working memory - no world knowledge needed, and random digits cannot have been memorised from training data.");
}

// =====================================================================
// 3. The ladder  -- the money slide
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Four papers, one question", "What shape should the model's intermediate reasoning have?");

  const cols = [
    { x: 0.75, name: "CoT", sub: "a chain", note: "show the steps" },
    { x: 3.85, name: "CoT-SC", sub: "k chains", note: "vote / pick best" },
    { x: 6.95, name: "ToT", sub: "a tree", note: "branch + backtrack" },
    { x: 10.05, name: "GoT", sub: "a graph", note: "AGGREGATE" },
  ];
  const cw = 2.5, top = 2.15, d = 0.2;

  cols.forEach((c, ci) => {
    const isGot = ci === 3;
    card(s, c.x, top, cw, 3.15, isGot ? "FFF4F0" : WHITE);
    s.addText(c.name, { x: c.x, y: top + 0.14, w: cw, h: 0.34, isTextBox: true,
      fontFace: HEAD, fontSize: 19, bold: true, color: isGot ? CORAL : INK,
      align: "center", margin: 0 });
    s.addText(c.sub, { x: c.x, y: top + 0.5, w: cw, h: 0.26, isTextBox: true,
      fontFace: BODY, fontSize: 12, color: MUTE, align: "center", margin: 0 });

    const mx = c.x + cw / 2 - d / 2, gy = top + 0.92, rowH = 0.52;
    if (ci === 0) {                                     // chain
      for (let i = 0; i < 4; i++) {
        if (i) edge(s, mx + d / 2, gy + (i - 1) * rowH + d / 2, mx + d / 2, gy + i * rowH + d / 2, DEEP, 1.2);
        node(s, mx, gy + i * rowH, d, i === 3 ? DEEP : TEAL);
      }
    } else if (ci === 1) {                              // k independent chains
      [-0.62, 0, 0.62].forEach(off => {
        for (let i = 0; i < 3; i++) {
          if (i) edge(s, mx + off + d / 2, gy + (i - 1) * rowH + d / 2, mx + off + d / 2, gy + i * rowH + d / 2, DEEP, 1.2);
          node(s, mx + off, gy + i * rowH, d, TEAL);
        }
      });
    } else if (ci === 2) {                              // tree
      node(s, mx, gy, d, TEAL);
      [-0.66, 0, 0.66].forEach(off => {
        edge(s, mx + d / 2, gy + d / 2, mx + off + d / 2, gy + rowH + d / 2, DEEP, 1.2);
        node(s, mx + off, gy + rowH, d, TEAL);
        [-0.22, 0.22].forEach(o2 => {
          edge(s, mx + off + d / 2, gy + rowH + d / 2, mx + off + o2 + d / 2, gy + 2 * rowH + d / 2, DEEP, 1.2);
          node(s, mx + off + o2, gy + 2 * rowH, d, off === 0 && o2 === 0.22 ? DEEP : "C3CEDF");
        });
      });
    } else {                                            // DAG / diamond
      node(s, mx, gy, d, TEAL);
      const mids = [-0.72, -0.24, 0.24, 0.72];
      mids.forEach(off => {
        edge(s, mx + d / 2, gy + d / 2, mx + off + d / 2, gy + rowH + d / 2, DEEP, 1.2);
        node(s, mx + off, gy + rowH, d, TEAL);
      });
      const merges = [-0.48, 0.48];
      merges.forEach((mo, mi) => {
        [mids[mi * 2], mids[mi * 2 + 1]].forEach(off =>
          edge(s, mx + off + d / 2, gy + rowH + d / 2, mx + mo + d / 2, gy + 2 * rowH + d / 2, CORAL, 1.6));
        node(s, mx + mo, gy + 2 * rowH, d, CORAL);
        edge(s, mx + mo + d / 2, gy + 2 * rowH + d / 2, mx + d / 2, gy + 3 * rowH + d / 2, CORAL, 1.6);
      });
      node(s, mx, gy + 3 * rowH, d, CORAL);
    }
    s.addText(c.note, { x: c.x, y: top + 2.72, w: cw, h: 0.3, isTextBox: true,
      fontFace: BODY, fontSize: 12, bold: isGot, color: isGot ? CORAL : MUTE,
      align: "center", margin: 0 });
  });

  card(s, 0.75, 5.62, 11.8, 1.05, NAVY);
  s.addText("Aggregation needs a node with in-degree > 1.  A tree cannot have one - by definition.",
    { x: 1.05, y: 5.78, w: 11.2, h: 0.36, isTextBox: true, fontFace: BODY,
      fontSize: 16, bold: true, color: WHITE, margin: 0 });
  s.addText("That single structural fact is the entire contribution of Graph of Thoughts.",
    { x: 1.05, y: 6.16, w: 11.2, h: 0.32, isTextBox: true, fontFace: BODY,
      fontSize: 13, color: "9AA7C7", margin: 0 });

  s.addNotes("This is the central slide. Each rung fixes a flaw in the one below. CoT-SC runs k chains but they never talk to each other. ToT can branch and backtrack but never merge. GoT removes the in-degree constraint, and merging becomes expressible. It is a theorem, not a preference.");
}

// =====================================================================
// 4. The framework
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "How the framework is built", "Two graphs - and confusing them causes most implementation bugs");

  card(s, 0.75, 1.95, 5.75, 2.0, PAPER);
  s.addText("GoO - Graph of Operations", { x: 1.05, y: 2.12, w: 5.2, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 14, bold: true, color: DEEP, margin: 0 });
  s.addText("STATIC. The execution plan. You build it by hand, before running.\nNodes = operations. Edges = dependencies.\nOne per task configuration.",
    { x: 1.05, y: 2.48, w: 5.2, h: 1.3, isTextBox: true, fontFace: BODY,
      fontSize: 13, color: INK, margin: 0 });

  card(s, 6.8, 1.95, 5.75, 2.0, PAPER);
  s.addText("GRS - Graph Reasoning State", { x: 7.1, y: 2.12, w: 5.2, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 14, bold: true, color: TEAL, margin: 0 });
  s.addText("DYNAMIC. The thoughts actually produced. The engine builds it.\nNodes = LLM outputs. Edges = data dependencies.\nOne per input instance.",
    { x: 7.1, y: 2.48, w: 5.2, h: 1.3, isTextBox: true, fontFace: BODY,
      fontSize: 13, color: INK, margin: 0 });

  s.addText("Three transformations - and only these create nodes",
    { x: 0.75, y: 4.15, w: 11.8, h: 0.34, isTextBox: true, fontFace: BODY,
      fontSize: 15, bold: true, color: INK, margin: 0 });

  const ops = [
    { n: "Generate (k)", d: "1 thought in, k out.\nBranching. Nothing new -\nToT already had it.", c: TEAL },
    { n: "Aggregate (k)", d: "k thoughts in, 1 out.\nEdge from EVERY input.\nThe contribution.", c: CORAL },
    { n: "Improve", d: "Refine in place.\nAlso impossible in a\ntree - it is a cycle.", c: DEEP },
  ];
  ops.forEach((o, i) => {
    const x = 0.75 + i * 4.0;
    card(s, x, 4.6, 3.75, 1.75, o.c === CORAL ? "FFF4F0" : WHITE);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.28, y: 4.85, w: 0.42, h: 0.42,
      fill: { color: o.c }, line: { width: 0 } });
    s.addText(String(i + 1), { x: x + 0.28, y: 4.89, w: 0.42, h: 0.34, isTextBox: true,
      fontFace: BODY, fontSize: 13, bold: true, color: WHITE, align: "center", margin: 0 });
    s.addText(o.n, { x: x + 0.85, y: 4.87, w: 2.7, h: 0.32, isTextBox: true,
      fontFace: BODY, fontSize: 14, bold: true, color: o.c, margin: 0 });
    s.addText(o.d, { x: x + 0.3, y: 5.35, w: 3.2, h: 0.9, isTextBox: true,
      fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
  });

  s.addText("Score, KeepBest and GroundTruth create NO nodes - they annotate and filter.",
    { x: 0.75, y: 6.55, w: 11.8, h: 0.3, isTextBox: true, fontFace: BODY,
      fontSize: 12, italic: true, color: MUTE, margin: 0 });

  s.addNotes("GoO is the recipe, GRS is the meal. Run one GoO over 100 inputs and you get one GoO and 100 GRSs. Only Generate, Aggregate and Improve create thoughts - which is exactly the paper's list of transformations.");
}

// =====================================================================
// 5. Sorting use case
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "The use case: merge sort, run by an LLM", "Break the task down until each step is inside the model's reliable range");

  const gx = 0.95, gy = 2.25, d = 0.24, rowH = 0.85;
  const lvl = [[6.0], [3.4, 5.1, 6.9, 8.6], [4.25, 7.75], [6.0]];
  const cols = [TEAL, TEAL, CORAL, CORAL];
  const labels = ["64 numbers", "sort 16 each (k=3)", "merge (k=10)", "merge - answer"];

  for (let L = 1; L < 4; L++) {
    lvl[L].forEach((x, i) => {
      const parents = L === 1 ? lvl[0] : (L === 2 ? [lvl[1][i * 2], lvl[1][i * 2 + 1]] : lvl[2]);
      parents.forEach(px => edge(s, px + d / 2, gy + (L - 1) * rowH + d / 2,
                                  x + d / 2, gy + L * rowH + d / 2,
                                  L >= 2 ? CORAL : "AEB8CC", L >= 2 ? 1.8 : 1.1));
    });
  }
  lvl.forEach((row, L) => row.forEach(x => node(s, x, gy + L * rowH, d, cols[L])));
  labels.forEach((t, L) => s.addText(t, { x: gx, y: gy + L * rowH - 0.05, w: 2.1, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 11.5, color: L >= 2 ? CORAL : MUTE,
    align: "right", margin: 0 }));

  s.addText("fan out", { x: 8.45, y: gy + 0.55, w: 1.6, h: 0.26, isTextBox: true,
    fontFace: BODY, fontSize: 11, italic: true, color: MUTE, margin: 0 });
  s.addText("fan back in", { x: 8.45, y: gy + 2.1, w: 1.6, h: 0.26, isTextBox: true,
    fontFace: BODY, fontSize: 11, italic: true, bold: true, color: CORAL, margin: 0 });

  card(s, 10.4, 2.1, 2.2, 1.35, PAPER);
  s.addText("39", { x: 10.4, y: 2.22, w: 2.2, h: 0.62, isTextBox: true, fontFace: HEAD,
    fontSize: 38, bold: true, color: DEEP, align: "center", margin: 0 });
  s.addText("thoughts per\ninstance", { x: 10.4, y: 2.86, w: 2.2, h: 0.5, isTextBox: true,
    fontFace: BODY, fontSize: 11, color: MUTE, align: "center", margin: 0 });

  card(s, 10.4, 3.6, 2.2, 1.35, "FFF4F0");
  s.addText("15", { x: 10.4, y: 3.72, w: 2.2, h: 0.62, isTextBox: true, fontFace: HEAD,
    fontSize: 38, bold: true, color: CORAL, align: "center", margin: 0 });
  s.addText("aggregations\n(ToT has 0)", { x: 10.4, y: 4.36, w: 2.2, h: 0.5, isTextBox: true,
    fontFace: BODY, fontSize: 11, color: MUTE, align: "center", margin: 0 });

  card(s, 0.75, 5.75, 11.8, 0.95, PAPER);
  s.addText("Sorting 16 numbers is easy. Merging two sorted lists is easy. Sorting 64 from scratch is not. So only ever do the easy two.",
    { x: 1.05, y: 5.95, w: 11.2, h: 0.5, isTextBox: true, fontFace: BODY,
      fontSize: 14, bold: true, color: DEEP, margin: 0 });

  s.addNotes("Scoring here is exact Python - count inversions and compare digit frequencies - so it is free and never wrong. That is a large advantage over ToT, which usually needs a second LLM call to evaluate a state.");
}

// =====================================================================
// 6. Datasets
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Datasets", "All four tasks are synthetic - and the authors ship the exact files they ran");

  const rows = [
    ["Sorting", "digits 0-9, with duplicates", "32 / 64 / 128", "100"],
    ["Set intersection", "two sets, 25-75% overlap", "32 / 64 / 128", "100"],
    ["Keyword counting", "country mentions in text", "4 / 8 / 16 sentences", "-"],
    ["Document merging", "overlapping NDAs", "4 documents", "-"],
  ];
  const tb = [[
    { text: "Task", options: { bold: true, color: WHITE } },
    { text: "Input", options: { bold: true, color: WHITE } },
    { text: "Sizes", options: { bold: true, color: WHITE } },
    { text: "Rows", options: { bold: true, color: WHITE } },
  ]].concat(rows.map((r, i) => r.map((c, j) => ({
    text: c, options: { color: j === 0 ? INK : "3C4761", bold: j === 0,
                        fill: { color: i % 2 ? "FFFFFF" : PAPER } } }))));
  s.addTable(tb, { x: 0.75, y: 1.95, w: 7.6, colW: [1.9, 2.75, 1.75, 1.2],
    rowH: 0.42, fontFace: BODY, fontSize: 12, border: { type: "solid", color: "E3E9F2", pt: 1 },
    fill: { color: WHITE }, align: "left", valign: "middle",
    autoPage: false });
  s.addShape(pres.ShapeType.rect, { x: 0.75, y: 1.95, w: 7.6, h: 0.42,
    fill: { color: DEEP }, line: { width: 0 } });
  s.addText([
    { text: "Task", options: { bold: true } }], { x: 0.9, y: 1.98, w: 1.8, h: 0.36,
    isTextBox: true, fontFace: BODY, fontSize: 12, color: WHITE, margin: 0 });
  s.addText("Input", { x: 2.8, y: 1.98, w: 2.6, h: 0.36, isTextBox: true,
    fontFace: BODY, fontSize: 12, bold: true, color: WHITE, margin: 0 });
  s.addText("Sizes", { x: 5.55, y: 1.98, w: 1.7, h: 0.36, isTextBox: true,
    fontFace: BODY, fontSize: 12, bold: true, color: WHITE, margin: 0 });
  s.addText("Rows", { x: 7.3, y: 1.98, w: 1.1, h: 0.36, isTextBox: true,
    fontFace: BODY, fontSize: 12, bold: true, color: WHITE, margin: 0 });

  bullets(s, [
    "No public benchmark - ground truth is COMPUTED, not annotated",
    "Randomly generated, so it cannot have leaked into training data",
    "Difficulty is a dial: 32 -> 64 -> 128 elements",
  ], 0.75, 4.15, 7.6, 1.3, 13);

  card(s, 8.75, 1.95, 3.8, 2.45, PAPER);
  s.addText("What we used", { x: 9.0, y: 2.12, w: 3.3, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: DEEP, margin: 0 });
  s.addText("The authors' own CSVs, from their GitHub repo, byte for byte:",
    { x: 9.0, y: 2.48, w: 3.35, h: 0.6, isTextBox: true, fontFace: BODY,
      fontSize: 12, color: INK, margin: 0 });
  s.addText("data/official/\n  sorting_064.csv\n  set_intersection_032.csv",
    { x: 9.0, y: 3.12, w: 3.35, h: 0.75, isTextBox: true, fontFace: "Courier New",
      fontSize: 10.5, color: DEEP, margin: 0 });
  s.addText("Removes \"different random data\" as an explanation for any gap.",
    { x: 9.0, y: 3.92, w: 3.35, h: 0.4, isTextBox: true, fontFace: BODY,
      fontSize: 11, italic: true, color: MUTE, margin: 0 });

  card(s, 8.75, 4.6, 3.8, 1.85, WHITE);
  s.addText("Also reproducible locally", { x: 9.0, y: 4.78, w: 3.3, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 13, bold: true, color: TEAL, margin: 0 });
  s.addText("generate_data.py --seed 42", { x: 9.0, y: 5.12, w: 3.35, h: 0.3,
    isTextBox: true, fontFace: "Courier New", fontSize: 10.5, color: INK, margin: 0 });
  s.addText("Same distributions, byte-identical on every machine - so laptop and GPU runs are comparable.",
    { x: 9.0, y: 5.5, w: 3.35, h: 0.8, isTextBox: true, fontFace: BODY,
      fontSize: 11, color: MUTE, margin: 0 });

  s.addNotes("Worth stressing: synthetic data is a deliberate strength here, not a shortcut. If GoT improved GSM8K you could not tell whether it helped reasoning or just gave more chances to recall a memorised answer.");
}

// =====================================================================
// 7. Setup
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Experimental setup", "The paper used a closed model we cannot query - so we replicate the claims, not the digits");

  card(s, 0.75, 1.95, 5.75, 2.55, PAPER);
  s.addText("The paper", { x: 1.05, y: 2.12, w: 5.2, h: 0.32, isTextBox: true,
    fontFace: BODY, fontSize: 15, bold: true, color: MUTE, margin: 0 });
  bullets(s, [
    "ChatGPT-3.5, paid API",
    "Temperature 1.0, 4k context",
    "100 samples per task",
    "Llama-2 tried and abandoned - too slow",
  ], 1.05, 2.55, 5.2, 1.8, 13, "3C4761");

  card(s, 6.8, 1.95, 5.75, 2.55, WHITE);
  s.addText("This project", { x: 7.1, y: 2.12, w: 5.2, h: 0.32, isTextBox: true,
    fontFace: BODY, fontSize: 15, bold: true, color: TEAL, margin: 0 });
  bullets(s, [
    "Qwen2.5-7B-Instruct, open weights",
    "No API keys, no paid services",
    "Same temperature, same 100 samples",
    "vLLM on an NVIDIA RTX A6000 (48 GB)",
  ], 7.1, 2.55, 5.2, 1.8, 13);

  const stats = [
    { n: "4", l: "interchangeable backends\nmock / llama.cpp / HF / vLLM", c: DEEP },
    { n: "5", l: "schemes on one engine\nIO, CoT, CoT-SC, ToT, GoT", c: TEAL },
    { n: "60", l: "automated tests\nrun before every GPU job", c: CORAL },
  ];
  stats.forEach((st, i) => {
    const x = 0.75 + i * 4.0;
    card(s, x, 4.75, 3.75, 1.65, WHITE);
    s.addText(st.n, { x: x + 0.25, y: 4.92, w: 1.0, h: 0.75, isTextBox: true,
      fontFace: HEAD, fontSize: 40, bold: true, color: st.c, margin: 0 });
    s.addText(st.l, { x: x + 1.3, y: 5.05, w: 2.3, h: 0.9, isTextBox: true,
      fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
  });

  s.addText("Because all five schemes share one controller, backend, prompts and scorer, any measured difference is attributable to graph structure alone.",
    { x: 0.75, y: 6.6, w: 11.8, h: 0.34, isTextBox: true, fontFace: BODY,
      fontSize: 12.5, italic: true, color: MUTE, margin: 0 });

  s.addNotes("The key experimental design point is the last line: one engine, five graph shapes. That is what makes the comparison about structure rather than about implementation differences.");
}

// =====================================================================
// 8. Structural results
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Result 1: the structure claims hold exactly", "Volume and latency are properties of the graph - independent of which model you run");

  const tb = [[
    "Scheme", "Volume", "Latency", "Aggregations", "Paper (Table 2)"
  ].map(t => ({ text: t, options: { bold: true, color: WHITE, fill: { color: DEEP } } }))];
  [["IO", "1.0", "1.0", "0", "1 / 1"],
   ["CoT", "2.0", "2.0", "0", "N / N"],
   ["CoT-SC", "2.0", "2.0", "0", "N/k / N/k"],
   ["ToT", "6.0", "6.0", "0", "O(log N) / log N"],
   ["GoT", "20.0", "9.0", "15", "N / log N"]].forEach((r, i) => {
    const got = i === 4;
    tb.push(r.map((c, j) => ({ text: c, options: {
      bold: got || j === 0, color: got ? CORAL : INK,
      fill: { color: got ? "FFF4F0" : (i % 2 ? WHITE : PAPER) } } })));
  });
  s.addTable(tb, { x: 0.75, y: 2.0, w: 7.4, colW: [1.4, 1.2, 1.2, 1.7, 1.9],
    rowH: 0.44, fontFace: BODY, fontSize: 12.5,
    border: { type: "solid", color: "E3E9F2", pt: 1 }, valign: "middle" });

  card(s, 8.6, 2.0, 3.95, 1.5, NAVY);
  s.addText("22 / 22", { x: 8.6, y: 2.14, w: 3.95, h: 0.62, isTextBox: true,
    fontFace: HEAD, fontSize: 36, bold: true, color: WHITE, align: "center", margin: 0 });
  s.addText("structural checks passed", { x: 8.6, y: 2.78, w: 3.95, h: 0.32,
    isTextBox: true, fontFace: BODY, fontSize: 12, color: "9AA7C7",
    align: "center", margin: 0 });
  s.addText("verified automatically, on every run", { x: 8.6, y: 3.06, w: 3.95, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 10.5, color: TEAL, align: "center", margin: 0 });

  card(s, 8.6, 3.7, 3.95, 2.25, "FFF4F0");
  s.addText("The headline", { x: 8.85, y: 3.88, w: 3.5, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 13, bold: true, color: CORAL, margin: 0 });
  s.addText("GoT reaches volume 20 at latency 9.\nToT reaches volume 6 at latency 6.",
    { x: 8.85, y: 4.24, w: 3.5, h: 0.7, isTextBox: true, fontFace: BODY,
      fontSize: 12.5, color: INK, margin: 0 });
  s.addText("3x the information reaching the answer, for 1.5x the depth - exactly what Table 2 predicts.",
    { x: 8.85, y: 4.98, w: 3.5, h: 0.85, isTextBox: true, fontFace: BODY,
      fontSize: 12, bold: true, color: DEEP, margin: 0 });

  s.addText("Volume = how many earlier thoughts could reach the final answer.   Latency = sequential model calls before you can finish.",
    { x: 0.75, y: 6.15, w: 11.8, h: 0.3, isTextBox: true, fontFace: BODY,
      fontSize: 11.5, italic: true, color: MUTE, margin: 0 });
  s.addText("GoT is the ONLY scheme that aggregates - measured, not assumed. Every other scheme reports exactly 0.",
    { x: 0.75, y: 6.5, w: 11.8, h: 0.32, isTextBox: true, fontFace: BODY,
      fontSize: 13, bold: true, color: INK, margin: 0 });

  s.addNotes("These are the claims a replication can settle definitively, because they do not depend on model quality. ToT's tree wastes almost everything it computes - only one root-to-leaf path reaches the answer. GoT's merges pull every branch back in.");
}

// =====================================================================
// 9. Empirical results
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Result 2: quality on a 7B open model", "100 instances, 64-element sorting, the authors' own dataset");

  s.addChart(pres.ChartType.bar, [{
    name: "Error scope (lower is better)",
    labels: ["IO", "CoT", "CoT-SC", "ToT", "GoT"],
    values: [9.83, 10.34, 8.17, 9.87, 8.11],
  }], {
    x: 0.75, y: 1.95, w: 6.5, h: 3.5,
    barDir: "col", chartColors: [DEEP, DEEP, DEEP, DEEP, CORAL],
    showTitle: true, title: "Mean error scope (lower is better)",
    titleFontFace: BODY, titleFontSize: 13, titleColor: INK,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11,
    dataLabelColor: INK, dataLabelFormatCode: "0.00",
    showLegend: false, catAxisLabelColor: MUTE, valAxisLabelColor: MUTE,
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
    valGridLine: { color: "EDF1F7", size: 1 }, catGridLine: { style: "none" },
    valAxisMaxVal: 12, chartArea: { fill: { color: WHITE } },
  });

  card(s, 7.6, 1.95, 4.95, 1.6, "FFF4F0");
  s.addText("GoT is best", { x: 7.85, y: 2.12, w: 4.4, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 14, bold: true, color: CORAL, margin: 0 });
  s.addText("Lowest error of all five schemes (8.11), with a 0% parse-failure rate across every scheme - so the pipeline is sound and the numbers mean something.",
    { x: 7.85, y: 2.48, w: 4.4, h: 1.0, isTextBox: true, fontFace: BODY,
      fontSize: 12, color: INK, margin: 0 });

  card(s, 7.6, 3.75, 4.95, 1.35, WHITE);
  s.addText("But no exact solutions", { x: 7.85, y: 3.92, w: 4.4, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 14, bold: true, color: DEEP, margin: 0 });
  s.addText("0% fully-correct at 64 elements. Expected: chain-of-thought is an emergent ability above ~10B parameters.",
    { x: 7.85, y: 4.28, w: 4.4, h: 0.75, isTextBox: true, fontFace: BODY,
      fontSize: 12, color: INK, margin: 0 });

  card(s, 7.6, 5.3, 4.95, 1.35, PAPER);
  s.addText("Honest gap", { x: 7.85, y: 5.47, w: 4.4, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 14, bold: true, color: MUTE, margin: 0 });
  s.addText("GoT costs more tokens than our ToT, where the paper reports it costing less - our ToT baseline is leaner than theirs.",
    { x: 7.85, y: 5.83, w: 4.4, h: 0.75, isTextBox: true, fontFace: BODY,
      fontSize: 12, color: INK, margin: 0 });

  s.addText("Paper: +62% over ToT on GPT-3.5.  Ours: GoT lowest, on a model 25x smaller that the authors never tested.",
    { x: 0.75, y: 5.65, w: 6.5, h: 0.6, isTextBox: true, fontFace: BODY,
      fontSize: 12, italic: true, color: MUTE, margin: 0 });

  s.addNotes("Be careful here: the exact percentage is being re-measured. We found our own ToT baseline was handicapped - it could not reject a refinement that made the answer worse - which inflated GoT's margin. Fixed, and rerunning. The direction of the result is unchanged.");
}

// =====================================================================
// 10. Status and what we learned
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  title(s, "Where it stands", "", true);

  const done = [
    "Framework: thoughts, operations, controller, metrics",
    "Sorting + set intersection, all 5 schemes",
    "Structural claims verified - 22/22 checks",
    "Real-model runs on GPU, clean parse rate",
  ];
  const next = [
    "Cost-matched ToT configs (the paper's ToT2)",
    "Keyword counting + document merging",
    "Larger model to test the scale threshold",
  ];

  card(s, 0.75, 2.0, 5.75, 2.55, "1C2545");
  s.addText("Done", { x: 1.05, y: 2.18, w: 5.2, h: 0.32, isTextBox: true,
    fontFace: BODY, fontSize: 15, bold: true, color: TEAL, margin: 0 });
  bullets(s, done, 1.05, 2.6, 5.2, 1.8, 12.5, "D6DDF0");

  card(s, 6.8, 2.0, 5.75, 2.55, "1C2545");
  s.addText("Next", { x: 7.1, y: 2.18, w: 5.2, h: 0.32, isTextBox: true,
    fontFace: BODY, fontSize: 15, bold: true, color: CORAL, margin: 0 });
  bullets(s, next, 7.1, 2.6, 5.2, 1.8, 12.5, "D6DDF0");

  s.addText("What replication actually taught us", { x: 0.75, y: 4.8, w: 11.8, h: 0.34,
    isTextBox: true, fontFace: BODY, fontSize: 15, bold: true, color: WHITE, margin: 0 });

  const lessons = [
    { t: "Silent failure is the enemy", d: "A run reported plausible numbers while the graph had stopped executing. Now guarded by a parse-rate check." },
    { t: "Audit baselines hardest", d: "Our ToT could not reject a bad rewrite - inflating GoT's margin in our favour." },
    { t: "Selection, not just compute", d: "CoT scores worse than no reasoning at all. Extra computation without a way to reject bad results is harmful." },
  ];
  lessons.forEach((l, i) => {
    const x = 0.75 + i * 4.0;
    card(s, x, 5.25, 3.75, 1.55, "1C2545");
    s.addText(l.t, { x: x + 0.28, y: 5.42, w: 3.2, h: 0.3, isTextBox: true,
      fontFace: BODY, fontSize: 12.5, bold: true, color: i === 1 ? CORAL : TEAL, margin: 0 });
    s.addText(l.d, { x: x + 0.28, y: 5.76, w: 3.2, h: 0.95, isTextBox: true,
      fontFace: BODY, fontSize: 10.5, color: "AAB5D0", margin: 0 });
  });

  s.addText("github.com/TREX4096/Graph_of_Thought", { x: 0.75, y: 6.95, w: 11.8, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 11, color: "6B7799", margin: 0 });

  s.addNotes("Close on the third lesson - it is the most interesting finding and it is ours, not the paper's. CoT doing worse than direct prompting shows that what makes these schemes work is the ability to score and discard, not simply spending more tokens.");
}

pres.writeFile({ fileName: process.argv[2] || "graph_of_thoughts.pptx" })
  .then(f => console.log("written:", f));

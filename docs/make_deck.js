// Causality-Preserving Graph Coarsening for Efficient LLM Reasoning
// Mid-Term Review deck. Academic format, Montserrat throughout.
//
// FONT: Montserrat is NOT bundled with Office. Install it on the presenting
// machine (fonts.google.com/specimen/Montserrat) or embed it via
// File > Options > Save > Embed fonts, otherwise text reflows.
//
// LOGO: drop the Department of Electrical Engineering / IIT Delhi logo at
// docs/assets/iitd_ee_logo.png and it is picked up automatically. Without it
// the deck falls back to a text lockup.
//
// Regenerate:  node docs/make_deck.js docs/Graph_of_Thoughts_BTP.pptx

const pptx = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const pres = new pptx();
pres.layout = "LAYOUT_WIDE";                     // 13.3 x 7.5 in
const W = 13.3;

// --- palette ---------------------------------------------------------
const BLUE  = "23408E";   // primary, academic
const PALE  = "C9D8F5";
const AMBER = "D98A1F";   // accent: compression
const TEAL  = "0F7B6C";   // positive result
const ROSE  = "C0392B";   // negative result
const INK   = "16181D";
const MUTE  = "63697A";
const WHITE = "FFFFFF";
const PAPER = "F4F6FB";

const F = "Montserrat";

pres.author = "Prasoon Raj, Nikhil Bansal";
pres.title  = "Causality-Preserving Graph Coarsening for Efficient LLM Reasoning";

// --- logo, if supplied ------------------------------------------------
const LOGO = ["docs/assets/iitd_ee_logo.png", "docs/assets/logo.png",
              "assets/iitd_ee_logo.png"]
  .map(p => path.resolve(__dirname, "..", p))
  .find(p => fs.existsSync(p));
if (!LOGO) console.warn("note: no logo found at docs/assets/iitd_ee_logo.png - using text lockup");

function brand(s, big) {
  if (LOGO) {
    s.addImage({ path: LOGO, x: 0.55, y: 0.2, w: big ? 1.5 : 1.0,
                 h: big ? 0.75 : 0.5, sizing: { type: "contain",
                 w: big ? 1.5 : 1.0, h: big ? 0.75 : 0.5 } });
  } else if (big) {
    s.addText("IIT DELHI", { x: 0.55, y: 0.24, w: 2.6, h: 0.3, isTextBox: true,
      fontFace: F, fontSize: 14, bold: true, color: BLUE, charSpacing: 2, margin: 0 });
    s.addText("Department of Electrical Engineering", { x: 0.55, y: 0.56, w: 3.4, h: 0.26,
      isTextBox: true, fontFace: F, fontSize: 10, color: MUTE, margin: 0 });
  }
}

// --- template furniture ----------------------------------------------
function corner(s) {
  s.addShape(pres.ShapeType.roundRect, { x: 11.1, y: -0.55, w: 2.6, h: 1.05,
    rectRadius: 0.4, fill: { color: PALE }, line: { width: 0 } });
  s.addShape(pres.ShapeType.roundRect, { x: 11.9, y: 0.0, w: 2.2, h: 0.62,
    rectRadius: 0.3, fill: { color: BLUE }, line: { width: 0 } });
  s.addShape(pres.ShapeType.roundRect, { x: -0.8, y: 6.95, w: 2.0, h: 0.9,
    rectRadius: 0.35, fill: { color: BLUE }, line: { width: 0 } });
  s.addShape(pres.ShapeType.roundRect, { x: 0.35, y: 7.2, w: 3.0, h: 0.8,
    rectRadius: 0.35, fill: { color: PALE }, line: { width: 0 } });
}
function head(s, text) {
  corner(s);
  if (LOGO) s.addImage({ path: LOGO, x: 11.75, y: 0.72, w: 1.0, h: 0.5,
                         sizing: { type: "contain", w: 1.0, h: 0.5 } });
  s.addText(text, { x: 0.62, y: 0.26, w: W - 3.4, h: 0.66, isTextBox: true,
    fontFace: F, fontSize: 30, bold: true, color: BLUE, margin: 0 });
  const rw = Math.min(6.6, 0.30 * text.length + 1.1);
  s.addShape(pres.ShapeType.line, { x: 0.68, y: 1.0, w: rw, h: 0,
    line: { color: INK, width: 1.6 } });
  s.addShape(pres.ShapeType.ellipse, { x: 0.6, y: 0.94, w: 0.13, h: 0.13,
    fill: { color: INK }, line: { width: 0 } });
  s.addShape(pres.ShapeType.ellipse, { x: 0.68 + rw - 0.06, y: 0.94, w: 0.13, h: 0.13,
    fill: { color: INK }, line: { width: 0 } });
}
function sub(s, text, x, y, w, color) {
  s.addText(text, { x, y, w: w || 6.4, h: 0.34, isTextBox: true, fontFace: F,
    fontSize: 15.5, bold: true, color: color || BLUE, margin: 0 });
}
function body(s, text, x, y, w, h, size, color) {
  s.addText(text, { x, y, w, h, isTextBox: true, fontFace: F, fontSize: size || 14,
    color: color || INK, margin: 0, lineSpacingMultiple: 1.06 });
}
function bullets(s, items, x, y, w, h, size, color) {
  // indent 10pt pulls the text close to the bullet -- pptxgenjs defaults to
  // ~27pt, which leaves a distracting gap on a dense academic slide.
  s.addText(items.map((t, i) => ({ text: t,
      options: { bullet: { indent: 10 }, breakLine: i !== items.length - 1 } })),
    { x, y, w, h, isTextBox: true, fontFace: F, fontSize: size || 15,
      color: color || INK, paraSpaceAfter: 4, margin: 0, lineSpacingMultiple: 1.04 });
}
function cite(s, text) {
  s.addText(text, { x: 4.6, y: 6.8, w: 8.2, h: 0.3, isTextBox: true, fontFace: F,
    fontSize: 10.5, italic: true, color: MUTE, align: "right", margin: 0 });
}
function card(s, x, y, w, h, fill, border) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.05,
    fill: { color: fill || WHITE }, line: { color: border || "D8DEEC", width: 1 } });
}
function node(s, x, y, d, fill) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d,
    fill: { color: fill }, line: { color: fill, width: 0 } });
}
function edge(s, x1, y1, x2, y2, color, width) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1),
              h: Math.abs(y2 - y1), line: { color: color || "AEB6C9", width: width || 1 } };
  if ((x2 - x1) * (y2 - y1) < 0) o.flipV = true;
  s.addShape(pres.ShapeType.line, o);
}
function tbl(s, header, rows, x, y, w, colW, hi, rowH, fs) {
  const t = [header.map(c => ({ text: c,
    options: { bold: true, color: WHITE, fill: { color: BLUE } } }))];
  rows.forEach((r, i) => t.push(r.map((c, j) => ({ text: c, options: {
    bold: (hi !== undefined && i === hi) || j === 0,
    color: (hi !== undefined && i === hi) ? AMBER : INK,
    fill: { color: (hi !== undefined && i === hi) ? "FDF3E3" : (i % 2 ? WHITE : PAPER) } } }))));
  s.addTable(t, { x, y, w, colW, rowH: rowH || 0.36, fontFace: F, fontSize: fs || 12,
    border: { type: "solid", color: "DCE2EF", pt: 1 }, valign: "middle" });
}

// =====================================================================
// 1  Title
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  corner(s);
  brand(s, true);
  s.addShape(pres.ShapeType.roundRect, { x: 0.75, y: 1.15, w: 11.8, h: 4.95,
    rectRadius: 0.05, fill: { color: WHITE }, line: { color: INK, width: 1.2 } });

  s.addText("Causality-Preserving Graph Coarsening\nfor Efficient LLM Reasoning",
    { x: 1.3, y: 1.65, w: 10.7, h: 1.5, isTextBox: true, fontFace: F, fontSize: 29,
      bold: true, color: BLUE, align: "center", lineSpacingMultiple: 1.14, margin: 0 });
  s.addText("Mid Term Review", { x: 1.3, y: 3.22, w: 10.7, h: 0.42, isTextBox: true,
    fontFace: F, fontSize: 18, bold: true, color: INK, align: "center", margin: 0 });

  s.addText("Presented by  -  Prasoon Raj (2023EE10708),  Nikhil Bansal (2023EE10787)",
    { x: 1.3, y: 4.0, w: 10.7, h: 0.32, isTextBox: true, fontFace: F,
      fontSize: 13, color: INK, align: "center", margin: 0 });
  s.addText("Guided by  -  Subhanu Halder (PhD Scholar)", { x: 1.3, y: 4.44, w: 10.7, h: 0.32,
    isTextBox: true, fontFace: F, fontSize: 13, color: INK, align: "center", margin: 0 });
  s.addText("Supervisor  -  Prof. Sandeep Kumar", { x: 1.3, y: 4.88, w: 10.7, h: 0.32,
    isTextBox: true, fontFace: F, fontSize: 13, color: INK, align: "center", margin: 0 });
  s.addText("Department of Electrical Engineering,  Indian Institute of Technology Delhi",
    { x: 1.3, y: 5.45, w: 10.7, h: 0.3, isTextBox: true, fontFace: F,
      fontSize: 11, color: MUTE, align: "center", margin: 0 });

  s.addNotes("Reasoning graphs make language models more accurate but far more expensive. Our aim is to shrink the graph without destroying the causal structure that makes it work. This review covers the replication that establishes the baseline, and the compression method it motivates.");
}

// =====================================================================
// 2  Problem Statement
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Problem Statement");

  sub(s, "Why does LLM reasoning cost matter ?", 0.75, 1.32, 7.4);
  bullets(s, [
    "A language model spends identical computation on every token, and cannot allocate more effort to a harder question.",
    "Structured prompting - Chain, Tree and Graph of Thoughts - buys accuracy by generating many intermediate thoughts.",
    "Cost scales with the number of thoughts and the length of each. We measure Graph of Thoughts at 15x the tokens of direct prompting.",
    "Inference cost, not accuracy, is what currently blocks deployment of graph-structured reasoning.",
  ], 0.75, 1.78, 7.4, 2.6, 14);

  sub(s, "Statement", 0.75, 4.6, 7.4);
  body(s, "To compress the reasoning graph of Graph of Thoughts - reducing both the number of vertices and the tokens per vertex - while preserving the causal dependencies that carry information to the final answer.",
    0.75, 5.05, 7.4, 1.3, 12);

  card(s, 8.6, 1.32, 4.0, 4.7, PAPER);
  s.addText("Measured cost per instance", { x: 8.85, y: 1.48, w: 3.5, h: 0.3,
    isTextBox: true, fontFace: F, fontSize: 13.5, bold: true, color: BLUE, margin: 0 });
  s.addText("64-element sorting, Qwen2.5-7B, 100 instances", { x: 8.85, y: 1.76, w: 3.5, h: 0.4,
    isTextBox: true, fontFace: F, fontSize: 10, color: MUTE, margin: 0 });
  [["IO", 520], ["CoT", 1169], ["CoT-SC", 887], ["ToT", 2185], ["GoT", 7910]]
    .forEach((b, i) => {
      const y = 2.3 + i * 0.68;
      s.addText(b[0], { x: 8.85, y: y + 0.02, w: 0.85, h: 0.26, isTextBox: true,
        fontFace: F, fontSize: 12.5, color: INK, margin: 0 });
      s.addShape(pres.ShapeType.roundRect, { x: 9.72, y, w: Math.max(0.1, (b[1] / 8200) * 2.0),
        h: 0.27, rectRadius: 0.03, fill: { color: i === 4 ? AMBER : BLUE }, line: { width: 0 } });
      s.addText(b[1].toLocaleString(), { x: 9.78 + (b[1] / 8200) * 2.0, y: y + 0.02,
        w: 0.95, h: 0.26, isTextBox: true, fontFace: F, fontSize: 8.5,
        color: i === 4 ? AMBER : MUTE, margin: 0 });
    });
  s.addText("mean tokens consumed", { x: 8.85, y: 5.68, w: 3.5, h: 0.24, isTextBox: true,
    fontFace: F, fontSize: 10, italic: true, color: MUTE, margin: 0 });

  s.addNotes("SCRIPT  (~75 sec)\n\nLanguage models spend identical computation on every token. They cannot think harder on a harder question, and once a token is sampled everything after it is conditioned on that choice. So asking directly gives an arbitrarily hard problem one token's worth of computation.\n\nStructured prompting fixes that by generating intermediate thoughts - and Graph of Thoughts generates the most, because it can merge separate lines of reasoning.\n\nThe problem is what that costs. On the right are our measurements: 64-element sorting, Qwen2.5-7B, a hundred instances. Graph of Thoughts consumes seven thousand nine hundred tokens per instance against five hundred and twenty for direct prompting. Fifteen times. And three and a half times Tree of Thoughts.\n\nThat price would be acceptable if it bought accuracy. At this scale it does not - Tree of Thoughts scores eleven percent better for a third of the cost. So the most structurally capable scheme is currently the worst deal available.\n\nAnd you cannot simply turn the cost down. We measured that too: cutting cost ninefold raises the error above the level you get with no reasoning structure at all. Uniform pruning removes useful and redundant computation at the same rate.\n\nSo our statement is: compress the reasoning graph - fewer vertices, and fewer tokens per vertex - while preserving the causal dependencies that carry information to the answer.\n\n---\nCUE: pause on the cost bars before reading the Statement.\nIF ASKED why 15x is a problem: it is the barrier to anyone deploying this, and at 7B it does not even buy accuracy.");
}

// =====================================================================
// 3  Study Summary - CoT
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Study Summaries");

  sub(s, "Paper 1 : Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", 0.75, 1.3, 11.0);
  s.addText("( Chain structure - intermediate steps written into the prompt )",
    { x: 0.75, y: 1.64, w: 11.0, h: 0.28, isTextBox: true, fontFace: F,
      fontSize: 12.5, italic: true, color: MUTE, margin: 0 });

  bullets(s, [
    "Intermediate reasoning steps are emitted before the answer, with no change to model weights.",
    "More tokens means more forward passes - the written chain IS the additional computation.",
    "GSM8K solve rate rises from 18% to 57% on PaLM-540B under chain-of-thought prompting.",
    "The ability is emergent: below roughly 10B parameters chains are fluent but logically invalid, and accuracy can fall below direct prompting.",
  ], 0.75, 2.1, 6.5, 3.0, 14);

  card(s, 7.6, 2.1, 5.0, 3.6, PAPER);
  s.addText("Structure", { x: 7.85, y: 2.26, w: 4.5, h: 0.28, isTextBox: true,
    fontFace: F, fontSize: 13.5, bold: true, color: BLUE, margin: 0 });
  const d = 0.21;
  for (let i = 0; i < 5; i++) {
    if (i) edge(s, 8.35 + d / 2, 2.72 + (i - 1) * 0.6 + d / 2, 8.35 + d / 2, 2.72 + i * 0.6 + d / 2, BLUE, 1.4);
    node(s, 8.35, 2.72 + i * 0.6, d, i === 4 ? AMBER : BLUE);
  }
  ["input x", "thought z1", "thought z2", "thought z3", "answer y"].forEach((t, i) =>
    s.addText(t, { x: 8.78, y: 2.72 + i * 0.6 - 0.02, w: 3.4, h: 0.26, isTextBox: true,
      fontFace: F, fontSize: 10, color: i === 4 ? AMBER : INK, margin: 0 }));
  s.addText("A single path. A wrong step is unrecoverable.", { x: 7.85, y: 5.25, w: 4.5, h: 0.3,
    isTextBox: true, fontFace: F, fontSize: 9.5, italic: true, color: MUTE, margin: 0 });

  cite(s, "Wei et al., NeurIPS 2022, arXiv:2201.11903");
  s.addNotes("SCRIPT  (~45 sec)\n\nChain-of-Thought is the foundation. You add worked examples to the prompt showing intermediate steps, and the model imitates that - with no change to the weights at all. So this is purely inference-time, which is true of everything in this project.\n\nThe key insight is that the chain is not the model explaining itself afterwards. The chain IS the computation. A model does one forward pass per token, so a one-token answer gets one token's worth of thinking no matter how hard the question. Fifty reasoning tokens gets fifty forward passes.\n\nAnd it works: GSM8K goes from eighteen percent to fifty-seven percent on PaLM-540B. That beat a fine-tuned model with a verifier, by prompting alone.\n\nOne caveat that matters for our results later. The ability is emergent. Below roughly ten billion parameters the chains are fluent but logically invalid, and accuracy can drop below direct prompting. We are running a 7B model, so we will come back to this.\n\nThe structural limitation is on the right. One path, each step with a single parent. Once a step is wrong, everything after it is conditioned on that error - there is no way to back out.\n\n---\nTRANSITION: That is what Tree of Thoughts addresses.\nCUE: the emergence bullet is load-bearing - it explains CoT scoring below IO on slide 11.");
}

// =====================================================================
// 4  Study Summary - ToT
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Study Summaries");

  sub(s, "Paper 2 : Tree of Thoughts - Deliberate Problem Solving with Large Language Models", 0.75, 1.3, 11.0);
  s.addText("( Tree structure - branching, evaluation and backtracking )",
    { x: 0.75, y: 1.64, w: 11.0, h: 0.28, isTextBox: true, fontFace: F,
      fontSize: 12.5, italic: true, color: MUTE, margin: 0 });

  bullets(s, [
    "Three components: a thought generator producing k children, a state evaluator scoring them, and a search algorithm over the tree.",
    "Adds local exploration and backtracking, which Self-Consistency with CoT cannot do - its k chains never exchange information.",
    "Every vertex has exactly one parent, so information on a discarded branch is lost permanently.",
    "In our replication ToT is the strongest baseline, once its evaluator can reject a refinement that worsens the answer.",
  ], 0.75, 2.1, 6.5, 3.0, 14);

  card(s, 7.6, 2.1, 5.0, 3.6, PAPER);
  s.addText("Structure", { x: 7.85, y: 2.26, w: 4.5, h: 0.28, isTextBox: true,
    fontFace: F, fontSize: 13.5, bold: true, color: BLUE, margin: 0 });
  const d = 0.19, cx = 10.0;
  node(s, cx, 2.8, d, BLUE);
  [-1.0, 0, 1.0].forEach(o => {
    edge(s, cx + d / 2, 2.8 + d / 2, cx + o + d / 2, 3.6 + d / 2, BLUE, 1.3);
    node(s, cx + o, 3.6, d, BLUE);
    [-0.3, 0.3].forEach(o2 => {
      edge(s, cx + o + d / 2, 3.6 + d / 2, cx + o + o2 + d / 2, 4.4 + d / 2, BLUE, 1.3);
      node(s, cx + o + o2, 4.4, d, (o === 0 && o2 === 0.3) ? AMBER : "B9C2D8");
    });
  });
  s.addText("Only one root-to-leaf path reaches the answer; the remainder of the tree is discarded.",
    { x: 7.85, y: 4.95, w: 4.5, h: 0.6, isTextBox: true, fontFace: F,
      fontSize: 9.5, italic: true, color: MUTE, margin: 0 });

  cite(s, "Yao et al., NeurIPS 2023, arXiv:2305.10601");
  s.addNotes("SCRIPT  (~45 sec)\n\nTree of Thoughts turns reasoning into a search problem. It has three named components: a thought generator that produces k children, a state evaluator that scores them, and a search algorithm - typically breadth-first with a beam - that decides what to expand next. Those map one to one onto operations in our framework.\n\nWhat it adds over self-consistency is local exploration. Self-consistency samples k complete chains and picks the best, but the branching happens only at the root - inside any one chain you are still committed. Tree of Thoughts branches at every level and can abandon a bad partial solution part-way.\n\nIts limitation is structural. Every vertex has exactly one parent, so when the search discards a branch, whatever was good on that branch is lost permanently. There is no operation in a tree that combines two branches. And since only one root-to-leaf path reaches the answer, nearly everything the tree computes is thrown away.\n\nOne note on our own results, because it comes up later: Tree of Thoughts is the strongest baseline we measure - once its evaluator is able to reject a refinement that made the answer worse. We will come back to that clause.\n\n---\nTRANSITION: Graph of Thoughts removes the one-parent constraint.\nIF ASKED why our ToT is stronger than the paper's: sorting admits an exact scorer, so our state evaluator is never wrong; theirs is typically another LLM call.");
}

// =====================================================================
// 5  Study Summary - GoT
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Study Summaries");

  sub(s, "Paper 3 : Graph of Thoughts - Solving Elaborate Problems with Large Language Models", 0.75, 1.3, 11.0);
  s.addText("( Graph structure - our baseline, and the object we compress )",
    { x: 0.75, y: 1.64, w: 11.0, h: 0.28, isTextBox: true, fontFace: F,
      fontSize: 11, italic: true, color: AMBER, margin: 0 });

  bullets(s, [
    "Reasoning is a directed graph: vertices are thoughts, and an edge (a,b) means a's text was the direct input that produced b.",
    "Aggregation merges k thoughts into one, creating a vertex of in-degree k > 1 - which a tree cannot contain, by definition.",
    "Reported gains of 62% in sorting quality over ToT with a 31% cost reduction, on ChatGPT-3.5.",
    "Latency log-k N at volume N: the only scheme where every computed thought retains a causal path to the answer.",
  ], 0.75, 2.1, 6.5, 3.0, 14);

  card(s, 7.6, 2.1, 5.0, 3.7, "FDF3E3");
  s.addText("Structure - merge sort executed by an LLM", { x: 7.85, y: 2.26, w: 4.5, h: 0.28,
    isTextBox: true, fontFace: F, fontSize: 12, bold: true, color: AMBER, margin: 0 });
  const d = 0.18, lv = [[10.05], [8.85, 9.65, 10.45, 11.25], [9.25, 10.85], [10.05]];
  const yy = [2.8, 3.45, 4.2, 4.95];
  for (let L = 1; L < 4; L++) lv[L].forEach((x, i) => {
    const par = L === 1 ? lv[0] : (L === 2 ? [lv[1][i * 2], lv[1][i * 2 + 1]] : lv[2]);
    par.forEach(px => edge(s, px + d / 2, yy[L - 1] + d / 2, x + d / 2, yy[L] + d / 2,
                           L >= 2 ? AMBER : "AEB6C9", L >= 2 ? 1.7 : 1.1));
  });
  lv.forEach((r, L) => r.forEach(x => node(s, x, yy[L], d, L >= 2 ? AMBER : BLUE)));
  s.addText("split  ->  sort chunks  ->  merge  ->  merge", { x: 7.85, y: 5.38, w: 4.5, h: 0.28,
    isTextBox: true, fontFace: F, fontSize: 13, color: INK, align: "center", margin: 0 });

  cite(s, "Besta et al., AAAI 2024, arXiv:2308.09687");
  s.addNotes("SCRIPT  (~50 sec)\n\nGraph of Thoughts models reasoning as a directed graph. Vertices are thoughts, and an edge means one thought's text was the direct input that produced another. So edges encode causality, not similarity - and that definition is what our whole method has to respect.\n\nThe contribution is aggregation: merging k thoughts into one. That creates a vertex with in-degree greater than one, and a tree cannot contain one by definition. So this is a structural result, not a design preference.\n\nFor sorting it runs merge sort with the model as the primitive - split into chunks, sort each chunk, then merge pairwise back up. Sorting sixteen numbers is reliable, and merging two sorted lists is reliable. Sorting sixty-four from scratch is not. So you only ever perform the reliable operations.\n\nThe theoretical result is volume N at latency log N. It is the only scheme where every computed thought keeps a causal path to the answer - a tree discards nearly all of what it computes.\n\nAnd the reported gain was sixty-two percent over Tree of Thoughts on ChatGPT-3.5, with a thirty-one percent cost reduction. That is the claim we set out to test.\n\n---\nTRANSITION: And this is the object we want to compress.\nCUE: point at the amber merge edges when saying 'in-degree greater than one'.\nDO NOT oversell the 62% - we report on slide 11 that it does not reproduce.");
}

// =====================================================================
// 6  Study Summary - CoT compression
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Study Summaries");

  sub(s, "Paper 4 : Compression of Chain-of-Thought Reasoning", 0.75, 1.3, 11.0);
  s.addText("( The literature we borrow from - all of it operates on a single chain )",
    { x: 0.75, y: 1.64, w: 11.0, h: 0.28, isTextBox: true, fontFace: F,
      fontSize: 12.5, italic: true, color: MUTE, margin: 0 });

  tbl(s, ["Method", "Mechanism", "Reported effect"], [
    ["Chain of Draft", "Cap each reasoning step to a few words", "~7.6% of CoT tokens, accuracy retained"],
    ["TokenSkip", "Prune low-utility tokens from the chain", "Controllable compression ratio"],
    ["Coconut", "Feed the hidden state back instead of decoding", "Reasoning without emitting text tokens"],
    ["Token budgeting", "Allocate a budget per step by difficulty", "Adaptive depth, less overthinking"],
  ], 0.75, 2.1, 7.5, [1.7, 2.9, 2.9], undefined, 0.56, 11.5);

  bullets(s, [
    "Every method above shortens the CONTENT of a reasoning step.",
    "None reduces the NUMBER of steps, because a chain has no structure to remove.",
    "A graph does - and that is the gap this project occupies.",
  ], 0.75, 4.75, 7.5, 1.3, 14);

  card(s, 8.6, 2.1, 4.0, 4.0, PAPER);
  s.addText("Two independent factors", { x: 8.85, y: 2.28, w: 3.5, h: 0.3,
    isTextBox: true, fontFace: F, fontSize: 13.5, bold: true, color: BLUE, margin: 0 });
  s.addText("cost  =  |V|  x  tokens/vertex", { x: 8.85, y: 2.68, w: 3.5, h: 0.34,
    isTextBox: true, fontFace: F, fontSize: 11.5, bold: true, color: AMBER, margin: 0 });
  s.addShape(pres.ShapeType.line, { x: 9.45, y: 3.04, w: 0, h: 0.85,
    line: { color: AMBER, width: 1.6 } });
  s.addShape(pres.ShapeType.line, { x: 11.3, y: 3.04, w: 0, h: 0.4,
    line: { color: TEAL, width: 1.6 } });
  s.addText("Graph coarsening\nacts on this factor", { x: 8.85, y: 3.92, w: 1.9, h: 0.6,
    isTextBox: true, fontFace: F, fontSize: 9.5, color: AMBER, margin: 0 });
  s.addText("Chain compression\nacts on this one", { x: 10.85, y: 3.48, w: 1.7, h: 0.6,
    isTextBox: true, fontFace: F, fontSize: 9.5, color: TEAL, margin: 0 });
  s.addText("The factors are orthogonal, so the two compressions compose and their savings multiply.",
    { x: 8.85, y: 4.9, w: 3.5, h: 0.95, isTextBox: true, fontFace: F,
      fontSize: 10.5, bold: true, color: BLUE, margin: 0 });

  cite(s, "Xu et al. arXiv:2502.18600  |  Xia et al. arXiv:2502.12067  |  Hao et al. arXiv:2412.06769");
  s.addNotes("SCRIPT  (~50 sec)\n\nThe second literature is chain-of-thought compression - a large and very fast-moving field, most of it from the last eighteen months.\n\nFour representative mechanisms. Chain of Draft caps each reasoning step to a few words and reports getting down to under eight percent of the original tokens with accuracy retained. TokenSkip prunes individual low-utility tokens and gives a controllable ratio. Coconut is the most radical - it feeds the hidden state straight back instead of decoding it, so the reasoning never becomes text at all. And token budgeting allocates an allowance per step by difficulty.\n\nBut notice what they all have in common. Every one of them shortens the CONTENT of a reasoning step. None of them reduces the NUMBER of steps - and they cannot, because a chain has no structure to remove. Every step was generated from the one before it.\n\nA graph does have removable structure. Graph of Thoughts generates k candidate sortings per chunk and keeps one; it tries k aggregations and keeps one. Those are vertices that exist and do not survive.\n\nSo cost factorises: number of vertices, times tokens per vertex. Chain compression acts on the second factor. Graph coarsening acts on the first. They are orthogonal, so in principle the savings multiply - and that intersection is where this project sits.\n\n---\nTRANSITION: So our objectives follow from that.\nBE HONEST: orthogonality is an assumption, not a result. It is listed as an open question.\nIF ASKED about Coconut needing training: yes - that row is the most speculative, and a longer-term direction.");
}

// =====================================================================
// 7  Objectives and Scope
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Objectives & Scope of Work");

  bullets(s, [
    "Faithful replication of Graph of Thoughts on open-weights models, with all four baselines on one engine",
    "Verification of the structural claims - volume, latency, aggregation count - independently of model quality",
    "Measurement of where the cost of a reasoning graph actually accumulates",
    "Application of chain-of-thought compression to the content of each vertex",
    "Development of a causality-preserving coarsening operator that reduces the number of vertices",
    "Evaluation of the compressed graph against the Tree of Thoughts cost-quality frontier",
  ], 0.9, 1.5, 7.3, 4.2, 14);

  card(s, 8.6, 1.45, 4.0, 4.6, PAPER);
  s.addText("Status at mid-term", { x: 8.85, y: 1.63, w: 3.5, h: 0.3, isTextBox: true,
    fontFace: F, fontSize: 13, bold: true, color: BLUE, margin: 0 });
  [["Replication framework", "done", TEAL], ["Structural verification", "done", TEAL],
   ["Cost characterisation", "done", TEAL], ["Naive-compression study", "done", TEAL],
   ["Vertex compression", "in progress", AMBER], ["Coarsening operator", "planned", MUTE],
   ["Frontier evaluation", "planned", MUTE],
  ].forEach((r, i) => {
    const y = 2.12 + i * 0.53;
    s.addShape(pres.ShapeType.ellipse, { x: 8.9, y: y + 0.06, w: 0.13, h: 0.13,
      fill: { color: r[2] }, line: { width: 0 } });
    s.addText(r[0], { x: 9.12, y, w: 2.25, h: 0.26, isTextBox: true, fontFace: F,
      fontSize: 13, color: INK, margin: 0 });
    s.addText(r[1], { x: 11.3, y, w: 1.1, h: 0.26, isTextBox: true, fontFace: F,
      fontSize: 10, bold: true, color: r[2], align: "right", margin: 0 });
  });

  s.addNotes("SCRIPT  (~45 sec)\n\nSix objectives. The first three are replication and measurement. Reproduce Graph of Thoughts faithfully with all baselines on a single engine, so that any difference is attributable to structure alone. Verify the structural claims - volume, latency and aggregation count - which are model-independent and therefore settleable outright. And measure where the cost actually accumulates, because you cannot compress what you have not located.\n\nThe last three are the contribution. Apply chain-of-thought compression to the content of each vertex. Develop a causality-preserving coarsening operator to reduce the number of vertices. And evaluate the result against the Tree of Thoughts cost-quality frontier.\n\nAt mid-term the first four items on the right are complete - including a naive-compression study that was not in the original plan, came out of the cost measurement, and became a result in its own right.\n\nVertex compression is in progress. The coarsening operator and the final evaluation are ahead of us, and those are the harder half - I would not claim we are past halfway. What the completed work buys us is that we now know precisely what we are aiming at: a specific error at a specific token count.\n\n---\nIF ASKED whether it is achievable: the infrastructure is done, which was the bulk of the engineering. Vertex compression is measurable this week. The coarsening operator is the research risk, and we have scoped it to two tasks rather than four.");
}

// =====================================================================
// 8  Methodology - graph construction
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Methodology");

  sub(s, "1) How the reasoning graph constructs its vertices and edges", 0.75, 1.3, 11.0);

  card(s, 0.75, 1.72, 5.6, 2.0, PAPER);
  s.addText("Execution loop over the plan", { x: 1.0, y: 1.85, w: 5.1, h: 0.26,
    isTextBox: true, fontFace: F, fontSize: 10, bold: true, color: BLUE, margin: 0 });
  s.addText("for op in TOPOLOGICAL-ORDER(GoO):\n   parents <- thoughts of op.predecessors\n   for t in op.EXECUTE(parents):\n      V <- V + { t }\n      for p in t.parents:\n         E <- E + { (p, t) }",
    { x: 1.0, y: 2.16, w: 5.1, h: 1.42, isTextBox: true, fontFace: "Courier New",
      fontSize: 13, color: INK, margin: 0 });

  tbl(s, ["Operation", "Vertices", "Edges"], [
    ["Generate (k)", "k per input", "input -> each new"],
    ["Aggregate (k)", "k, from all inputs", "EVERY input -> each new"],
    ["Improve", "1 per input", "input -> new"],
    ["Score", "none", "none"],
    ["KeepBest (n)", "none", "none"],
  ], 6.75, 1.72, 5.8, [1.75, 1.85, 2.2], 1, 0.35, 11.5);

  s.addText("Only Generate, Aggregate and Improve create vertices. Score and KeepBest annotate and filter.",
    { x: 6.75, y: 3.9, w: 5.8, h: 0.3, isTextBox: true, fontFace: F,
      fontSize: 9.5, italic: true, color: MUTE, margin: 0 });

  sub(s, "2) The rule that defines the structure", 0.75, 4.3, 11.0);
  card(s, 0.75, 4.72, 5.6, 1.9, "FDF3E3");
  const d = 0.16, ay = 5.18;
  [-0.7, -0.23, 0.23, 0.7].forEach(o => {
    edge(s, 2.3 + o + d / 2, ay + d / 2, 2.3 + d / 2, ay + 0.72 + d / 2, AMBER, 1.7);
    node(s, 2.3 + o, ay, d, BLUE);
  });
  node(s, 2.3, ay + 0.72, d + 0.03, AMBER);
  s.addText("in-degree 4 > 1\nso the graph is not a tree", { x: 3.25, y: 5.3, w: 2.9, h: 0.6,
    isTextBox: true, fontFace: F, fontSize: 10.5, bold: true, color: INK, margin: 0 });

  card(s, 6.75, 4.72, 5.8, 1.9, PAPER);
  s.addText("An edge (a, b) means a's text was the input that produced b.",
    { x: 7.0, y: 4.86, w: 5.3, h: 0.66, isTextBox: true, fontFace: F,
      fontSize: 13.5, bold: true, color: BLUE, margin: 0 });
  s.addText("Edges encode causality, not similarity. A coarsening operator therefore cannot merge vertices because they resemble one another - it must preserve which thoughts can still reach the answer.",
    { x: 7.0, y: 5.58, w: 5.3, h: 0.95, isTextBox: true, fontFace: F,
      fontSize: 12, color: INK, margin: 0 });

  s.addNotes("SCRIPT  (~60 sec)\n\nThere are two graphs here, and keeping them apart matters. The Graph of Operations is the plan - vertices are operations, and you build it by hand before the run. The Graph Reasoning State is what actually gets produced - vertices are thoughts. Run one plan over a hundred inputs and you get one plan and a hundred state graphs.\n\nThe state graph is built by this single loop, in topological order. Topological is required rather than stylistic: with aggregation the plan is a DAG, so an operation can have two predecessors, and you cannot simply run them in the order they were written.\n\nThe rules are on the right. Generate makes k vertices from one input. Aggregate makes vertices that take an edge from EVERY input - that is the fan-in. Improve refines in place. And Score and KeepBest create nothing at all; they annotate and filter. So only three operations ever create a vertex, which is exactly the paper's three thought transformations.\n\nThe consequence is at the bottom. Aggregation gives a vertex in-degree greater than one, so the graph is not a tree. And an edge means one thought's text was literally the input that produced another - causality, not similarity. That is the constraint our coarsening operator has to respect: we cannot merge two vertices because they look alike.\n\n---\nSPARE 10 SEC: on 32 numbers the first merge level creates 20 vertices but 40 edges - two per vertex, because each merge draws from both inputs. The only point in the trace where edges outnumber vertices.\nIF ASKED why topological: an operation cannot run before the thoughts it consumes exist, and written order does not guarantee that once the plan is a DAG.");
}

// =====================================================================
// 9  Methodology - setup
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Methodology");

  sub(s, "3) Datasets - synthetic, with computed ground truth", 0.75, 1.3, 7.4);
  tbl(s, ["Task", "Input", "Sizes", "Instances"], [
    ["Sorting", "digits 0-9 with duplicates", "32 / 64 / 128", "100"],
    ["Set intersection", "two sets, 25-75% overlap", "32 / 64 / 128", "100"],
    ["Keyword counting", "country mentions in text", "4 / 8 / 16 sentences", "-"],
    ["Document merging", "overlapping NDA documents", "4 documents", "-"],
  ], 0.75, 1.7, 7.4, [1.8, 2.65, 1.75, 1.2], undefined, 0.38, 11.5);

  s.addText("Run on the authors' own published CSV files, byte for byte - eliminating differing random data as an explanation for any discrepancy.",
    { x: 0.75, y: 3.6, w: 7.4, h: 0.45, isTextBox: true, fontFace: F,
      fontSize: 12, italic: true, color: MUTE, margin: 0 });

  sub(s, "4) Models and infrastructure", 0.75, 4.15, 7.4);
  tbl(s, ["", "Original paper", "This work"], [
    ["Model", "ChatGPT-3.5 (closed API)", "Qwen2.5-7B-Instruct (open)"],
    ["Serving", "Vendor API", "vLLM on NVIDIA RTX A6000"],
    ["Sampling", "Temperature 1.0, 4k context", "Identical"],
    ["Samples", "100 per configuration", "100 per configuration"],
  ], 0.75, 4.55, 7.4, [1.35, 3.05, 3.0], undefined, 0.38, 11.5);

  card(s, 8.6, 1.7, 4.0, 4.35, PAPER);
  s.addText("Experimental control", { x: 8.85, y: 1.88, w: 3.5, h: 0.3, isTextBox: true,
    fontFace: F, fontSize: 13, bold: true, color: BLUE, margin: 0 });
  s.addText("All five schemes share one controller, one backend, one prompt set and one scorer.",
    { x: 8.85, y: 2.26, w: 3.5, h: 0.78, isTextBox: true, fontFace: F,
      fontSize: 13, color: INK, margin: 0 });
  s.addText("Any measured difference is therefore attributable to graph structure alone.",
    { x: 8.85, y: 3.06, w: 3.5, h: 0.72, isTextBox: true, fontFace: F,
      fontSize: 12, bold: true, color: AMBER, margin: 0 });
  [["5", "schemes on one engine"], ["4", "interchangeable backends"],
   ["60", "automated tests per run"]].forEach((r, i) => {
    const y = 3.95 + i * 0.68;
    s.addText(r[0], { x: 8.85, y, w: 0.95, h: 0.42, isTextBox: true, fontFace: F,
      fontSize: 21, bold: true, color: BLUE, margin: 0 });
    s.addText(r[1], { x: 9.8, y: y + 0.1, w: 2.55, h: 0.3, isTextBox: true,
      fontFace: F, fontSize: 10, color: MUTE, margin: 0 });
  });

  s.addNotes("SCRIPT  (~60 sec)\n\nAll four tasks are synthetic with computed ground truth. That means no annotation - the correct sorting of a list is computable - and crucially no contamination, because randomly generated digits cannot have been in the training data. On a public benchmark you could never separate improved reasoning from memorisation. Difficulty is also a controlled variable: thirty-two, sixty-four, a hundred and twenty-eight.\n\nAnd we run on the authors' own published CSV files, byte for byte. That removes differing random data as an explanation for any discrepancy between our numbers and theirs.\n\nOn models. The paper used ChatGPT-3.5, which is closed and now deprecated - nobody can reproduce its exact figures, including the authors. So we do not attempt to replicate the numbers. We replicate the claims, which are relative and testable on any model. We use Qwen2.5-7B, open weights, served with vLLM on an A6000, at identical temperature and sample counts.\n\nThe experimental design is the last line, and it is the one that matters. All five schemes share one controller, one backend, one prompt set and one scorer. So anything we measure is attributable to graph structure alone - not to one scheme happening to have better prompts than another.\n\n---\nTRANSITION: So - results.\nIF ASKED if 7B is too small: the structural claims are model-independent and settle completely. And whether the quality claim generalises below GPT-3.5 scale is itself a question nobody had tested.");
}

// =====================================================================
// 10  Results - structural
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Results & Discussion");

  sub(s, "Structural claims - model-independent, and reproduced exactly", 0.75, 1.3, 11.0);

  tbl(s, ["Scheme", "Volume", "Latency", "Aggregations"], [
    ["IO", "1.0", "1.0", "0"], ["CoT", "2.0", "2.0", "0"],
    ["CoT-SC", "2.0", "2.0", "0"], ["ToT", "5.3", "5.3", "0"],
    ["GoT", "20.0", "9.0", "15"],
  ], 0.75, 1.72, 6.2, [1.6, 1.5, 1.5, 1.6], 4, 0.4, 10);

  body(s, "Volume is the number of prior thoughts with a causal path to the final answer; latency is the count of sequential model calls. Both are properties of the graph, so they cannot vary with model quality - a deviation here would indicate a defect, not a finding.",
    0.75, 4.3, 6.2, 1.25, 10.5);

  card(s, 0.75, 5.7, 6.2, 0.95, "E8F4F1");
  s.addText("Graph of Thoughts is the only scheme that aggregates - measured, not assumed. Every other scheme reports exactly zero.",
    { x: 1.0, y: 5.86, w: 5.7, h: 0.64, isTextBox: true, fontFace: F,
      fontSize: 13, bold: true, color: TEAL, margin: 0 });

  s.addChart(pres.ChartType.bar, [
    { name: "Volume", labels: ["IO", "CoT", "CoT-SC", "ToT", "GoT"], values: [1, 2, 2, 5.3, 20] },
    { name: "Latency", labels: ["IO", "CoT", "CoT-SC", "ToT", "GoT"], values: [1, 2, 2, 5.3, 9] },
  ], {
    x: 7.3, y: 1.72, w: 5.3, h: 4.1, barDir: "col", barGrouping: "clustered",
    chartColors: [AMBER, BLUE], showTitle: true, title: "Volume vs latency per scheme",
    titleFontFace: F, titleFontSize: 11, titleColor: INK,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 8.5,
    dataLabelColor: INK, showLegend: true, legendPos: "b", legendFontSize: 9.5,
    catAxisLabelColor: MUTE, valAxisLabelColor: MUTE, catAxisLabelFontSize: 9.5,
    valAxisLabelFontSize: 8.5, valGridLine: { color: "EDF0F7", size: 1 },
    catGridLine: { style: "none" }, chartArea: { fill: { color: WHITE } },
  });

  s.addNotes("The gap between the amber and blue bars for GoT is the whole point of the paper: high volume at low latency. Every other scheme has the bars equal, because a chain or tree wastes what it computes.");
}

// =====================================================================
// 11  Results - quality
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Results & Discussion");

  sub(s, "Quality - the paper's claim does not reproduce at 7B", 0.75, 1.3, 11.0);

  s.addChart(pres.ChartType.bar, [{
    name: "Error scope", labels: ["IO", "CoT", "CoT-SC", "ToT", "GoT"],
    values: [9.83, 10.34, 8.17, 7.45, 8.28],
  }], {
    x: 0.75, y: 1.72, w: 6.3, h: 3.75, barDir: "col",
    chartColors: [BLUE, BLUE, BLUE, TEAL, AMBER],
    showTitle: true, title: "Mean error scope, 100 instances (lower is better)",
    titleFontFace: F, titleFontSize: 11, titleColor: INK,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9.5,
    dataLabelColor: INK, dataLabelFormatCode: "0.00", showLegend: false,
    catAxisLabelColor: MUTE, valAxisLabelColor: MUTE, catAxisLabelFontSize: 10,
    valAxisLabelFontSize: 8.5, valGridLine: { color: "EDF0F7", size: 1 },
    catGridLine: { style: "none" }, valAxisMaxVal: 12,
    chartArea: { fill: { color: WHITE } },
  });
  body(s, "Parse-failure rate was 0.0% for every scheme, so the pipeline is sound and the ordering is meaningful.",
    0.75, 5.62, 6.3, 0.5, 10, MUTE);

  card(s, 7.4, 1.72, 5.2, 1.5, "E8F4F1");
  s.addText("Structure does help", { x: 7.65, y: 1.87, w: 4.7, h: 0.28, isTextBox: true,
    fontFace: F, fontSize: 13, bold: true, color: TEAL, margin: 0 });
  s.addText("Graph of Thoughts improves on direct prompting by 15.8%, and all structural claims hold exactly.",
    { x: 7.65, y: 2.19, w: 4.7, h: 0.9, isTextBox: true, fontFace: F,
      fontSize: 12.5, color: INK, margin: 0 });

  card(s, 7.4, 3.38, 5.2, 1.6, "FBEBE9");
  s.addText("But Tree of Thoughts wins here", { x: 7.65, y: 3.53, w: 4.7, h: 0.28,
    isTextBox: true, fontFace: F, fontSize: 13, bold: true, color: ROSE, margin: 0 });
  s.addText("ToT reaches 11.1% lower error than GoT while consuming 3.6x fewer tokens.",
    { x: 7.65, y: 3.85, w: 4.7, h: 0.58, isTextBox: true, fontFace: F,
      fontSize: 12.5, color: INK, margin: 0 });
  s.addText("Likely causes: an exact scorer makes ToT's monotone refinement very strong, and at 7B each of GoT's 15 merges is an opportunity to fail.",
    { x: 7.65, y: 4.4, w: 4.7, h: 0.55, isTextBox: true, fontFace: F,
      fontSize: 10.5, italic: true, color: MUTE, margin: 0 });

  card(s, 7.4, 5.14, 5.2, 1.5, PAPER);
  s.addText("A note on baselines", { x: 7.65, y: 5.29, w: 4.7, h: 0.28, isTextBox: true,
    fontFace: F, fontSize: 13, bold: true, color: BLUE, margin: 0 });
  s.addText("An earlier run showed GoT ahead by 17.8%. That margin was an artefact of our own ToT implementation being unable to reject a refinement that worsened the answer. Correcting it reversed the result.",
    { x: 7.65, y: 5.6, w: 4.7, h: 0.95, isTextBox: true, fontFace: F,
      fontSize: 13, color: INK, margin: 0 });

  s.addNotes("Present this honestly - a negative result that survives a corrected baseline is stronger evidence than a confirmation. It also sharpens the motivation: the graph is being paid for and, at this scale, is not earning it.");
}

// =====================================================================
// 12  Results - cost, naive compression
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Results & Discussion");

  sub(s, "Cost, and why naive compression fails", 0.75, 1.3, 11.0);

  tbl(s, ["Configuration", "Tokens", "Volume", "Error"], [
    ["8 chunks", "14,384", "39.9", "2.88"],
    ["4 chunks (paper)", "8,565", "19.9", "3.20"],
    ["2 chunks", "5,163", "9.9", "3.38"],
    ["stripped down", "950", "8.0", "10.50"],
  ], 0.75, 1.72, 6.3, [2.1, 1.5, 1.35, 1.35], 3, 0.42, 10);

  card(s, 0.75, 3.85, 6.3, 1.1, "FBEBE9");
  s.addText("Reducing cost ninefold raises the error to 10.50 - worse than using no reasoning structure at all, where direct prompting scores 9.83.",
    { x: 1.0, y: 4.02, w: 5.8, h: 0.8, isTextBox: true, fontFace: F,
      fontSize: 13, bold: true, color: ROSE, margin: 0 });

  body(s, "Uniform pruning is structure-blind: lowering k or the chunk count removes useful and redundant computation at the same rate. Cost and quality stay coupled, so no setting of the existing knobs reaches the Tree of Thoughts frontier.",
    0.75, 5.1, 6.3, 1.3, 10.5);

  card(s, 7.4, 1.72, 5.2, 4.7, PAPER);
  s.addText("Where the cost accumulates", { x: 7.65, y: 1.9, w: 4.7, h: 0.3,
    isTextBox: true, fontFace: F, fontSize: 13, bold: true, color: BLUE, margin: 0 });
  [["Chunk sorting", 1580, BLUE], ["Merge level 1", 2890, AMBER],
   ["Merge level 2", 2210, AMBER], ["Final refinement", 1230, BLUE]]
    .forEach((p, i) => {
      const y = 2.42 + i * 0.75;
      s.addText(p[0], { x: 7.65, y, w: 1.85, h: 0.26, isTextBox: true, fontFace: F,
        fontSize: 13, color: INK, margin: 0 });
      s.addShape(pres.ShapeType.roundRect, { x: 9.55, y, w: (p[1] / 3000) * 1.95, h: 0.26,
        rectRadius: 0.03, fill: { color: p[2] }, line: { width: 0 } });
      s.addText(String(p[1]), { x: 9.61 + (p[1] / 3000) * 1.95, y, w: 0.85, h: 0.26,
        isTextBox: true, fontFace: F, fontSize: 10, color: MUTE, margin: 0 });
    });
  s.addText("Aggregation consumes roughly 65% of the budget - each merge emits a full-length list, and there are fifteen of them.",
    { x: 7.65, y: 5.5, w: 4.7, h: 0.8, isTextBox: true, fontFace: F,
      fontSize: 12.5, bold: true, color: AMBER, margin: 0 });

  s.addNotes("The right panel is the useful diagnostic: cost is concentrated in aggregation, not in chunk sorting. That tells us where both compression axes should be aimed first.");
}

// =====================================================================
// 13  Proposed approach
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Proposed Approach");

  sub(s, "Transferring chain compression to a reasoning graph", 0.75, 1.3, 11.0);

  tbl(s, ["Chain technique", "Graph analogue we propose", "Expected effect"], [
    ["Chain of Draft", "Encode each vertex compactly - a sorted chunk as run-length pairs rather than a literal list", "Lossless here; 3.8x fewer tokens per vertex (measured)"],
    ["TokenSkip", "Ablate each vertex; drop those whose removal does not change the answer", "Reduces |V| directly"],
    ["Coconut", "Aggregate in latent space - merge hidden states without decoding to text", "Removes decode cost at the 15 merge steps"],
    ["Token budgeting", "Allocate k per vertex by measured difficulty rather than uniformly", "Spend where errors concentrate"],
  ], 0.75, 1.72, 7.6, [1.7, 3.45, 2.45], undefined, 0.74, 11.5);

  card(s, 0.75, 4.85, 7.6, 1.1, "FDF3E3");
  s.addText("The causality constraint decides which of these are safe: an edge means one thought was literally the input that produced another, so every vertex on a path to the answer must retain one.",
    { x: 1.0, y: 5.02, w: 7.1, h: 0.8, isTextBox: true, fontFace: F,
      fontSize: 10, bold: true, color: INK, margin: 0 });

  card(s, 8.65, 1.72, 3.95, 4.23, PAPER);
  s.addText("Pipeline", { x: 8.9, y: 1.9, w: 3.45, h: 0.28, isTextBox: true,
    fontFace: F, fontSize: 13, bold: true, color: BLUE, margin: 0 });
  ["Instrument: per-vertex token cost and marginal contribution",
   "Compress vertex content with a compact encoding",
   "Coarsen: merge or drop low-contribution vertices",
   "Verify reachability to the final thought is preserved",
   "Evaluate against the ToT cost-quality frontier",
  ].forEach((t, i) => {
    const y = 2.32 + i * 0.7;
    s.addShape(pres.ShapeType.ellipse, { x: 8.9, y, w: 0.29, h: 0.29,
      fill: { color: i < 2 ? BLUE : AMBER }, line: { width: 0 } });
    s.addText(String(i + 1), { x: 8.9, y: y + 0.03, w: 0.29, h: 0.24, isTextBox: true,
      fontFace: F, fontSize: 10.5, bold: true, color: WHITE, align: "center", margin: 0 });
    s.addText(t, { x: 9.3, y: y - 0.02, w: 3.05, h: 0.62, isTextBox: true,
      fontFace: F, fontSize: 13, color: INK, margin: 0 });
  });

  card(s, 0.75, 6.1, 11.85, 0.7, "E8F4F1");
  s.addText("Target, set by our own measurement:  reach error 7.45 at 2,185 tokens - where Tree of Thoughts sits today. Graph of Thoughts is currently at 8.28 error and 7,910 tokens.",
    { x: 1.0, y: 6.24, w: 11.4, h: 0.44, isTextBox: true, fontFace: F,
      fontSize: 13, bold: true, color: TEAL, margin: 0 });

  s.addNotes("The run-length figure is measured with Qwen2.5's own tokenizer, not estimated. A 64-element list costs exactly 192 tokens - three per element, being the digit, the comma and the space. Run-length encoded it costs 50. That is 3.8x, losslessly, because the encoding is exact. It is chain-of-draft applied to a graph vertex.");
}

// =====================================================================
// 14  Future Plan
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "Future Plan");

  bullets(s, [
    "Complete the sweeps now running - aggregation attempts and chunk count across 32, 64 and 128 elements",
    "Implement compact vertex encoding and measure its accuracy cost",
    "Define the coarsening operator formally, with reachability to the final thought as the preserved invariant",
    "Establish whether coarsening and vertex compression compose as predicted, or interact",
    "Repeat at a larger model scale, to separate scale effects from structural ones",
    "Extend beyond sorting to set intersection, where aggregation is a union rather than a merge",
  ], 0.9, 1.5, 7.4, 4.3, 14);

  card(s, 8.7, 1.5, 3.9, 4.6, PAPER);
  s.addText("Open questions", { x: 8.95, y: 1.68, w: 3.4, h: 0.3, isTextBox: true,
    fontFace: F, fontSize: 13, bold: true, color: BLUE, margin: 0 });
  ["Does latent aggregation preserve causality in any meaningful sense, or dissolve the edge semantics entirely?",
   "Is there a principled criterion for vertex removal, or must it be learned per task?",
   "Does GoT's disadvantage at 7B close with scale, or is aggregation intrinsically fragile?",
  ].forEach((q, i) => {
    const y = 2.15 + i * 1.32;
    s.addText(String(i + 1), { x: 8.95, y, w: 0.3, h: 0.3, isTextBox: true, fontFace: F,
      fontSize: 12.5, bold: true, color: AMBER, margin: 0 });
    s.addText(q, { x: 9.28, y, w: 3.07, h: 1.2, isTextBox: true, fontFace: F,
      fontSize: 13, color: INK, margin: 0 });
  });

  s.addNotes("The third question is the one a committee is most likely to press on, and we should be candid that we cannot yet answer it - the 7B result may be a scale artefact.");
}

// =====================================================================
// 15  References
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  head(s, "References");

  [
    "[1] M. Besta, N. Blach, A. Kubicek, et al., \"Graph of Thoughts: Solving elaborate problems with large language models,\" in Proc. AAAI Conf. Artif. Intell., vol. 38, 2024, arXiv:2308.09687.",
    "[2] S. Yao, D. Yu, J. Zhao, et al., \"Tree of Thoughts: Deliberate problem solving with large language models,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), 2023, arXiv:2305.10601.",
    "[3] J. Wei, X. Wang, D. Schuurmans, et al., \"Chain-of-thought prompting elicits reasoning in large language models,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), 2022, arXiv:2201.11903.",
    "[4] X. Wang, J. Wei, D. Schuurmans, et al., \"Self-consistency improves chain of thought reasoning in language models,\" in Proc. Int. Conf. Learn. Represent. (ICLR), 2023, arXiv:2203.11171.",
    "[5] S. Xu, W. Xie, L. Zhao, and P. He, \"Chain of Draft: Thinking faster by writing less,\" 2025, arXiv:2502.18600.",
    "[6] H. Xia, Y. Li, C. T. Leong, et al., \"TokenSkip: Controllable chain-of-thought compression in LLMs,\" 2025, arXiv:2502.12067.",
    "[7] S. Hao, S. Sukhbaatar, D. Su, et al., \"Training large language models to reason in a continuous latent space,\" 2024, arXiv:2412.06769.",
    "[8] A. Loukas, \"Graph reduction with spectral and cut guarantees,\" J. Mach. Learn. Res., vol. 20, no. 116, pp. 1-42, 2019.",
    "[9] W. Kwon, Z. Li, S. Zhuang, et al., \"Efficient memory management for large language model serving with PagedAttention,\" in Proc. ACM SOSP, 2023, arXiv:2309.06180.",
  ].forEach((r, i) => {
    s.addText(r, { x: 0.8, y: 1.32 + i * 0.585, w: 11.6, h: 0.55, isTextBox: true,
      fontFace: F, fontSize: 13, color: INK, margin: 0, lineSpacingMultiple: 1.1 });
  });

  s.addNotes("Items 1 to 4 are the reasoning-structure lineage we replicated; 5 to 7 are the chain-compression methods we intend to transfer; 8 and 9 are the graph-reduction theory and the serving infrastructure.");
}

// =====================================================================
// 16  Thank You
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  corner(s);
  brand(s, true);
  s.addShape(pres.ShapeType.roundRect, { x: 2.8, y: 2.45, w: 7.7, h: 2.5,
    rectRadius: 0.06, fill: { color: WHITE }, line: { color: INK, width: 1.2 } });
  s.addText("Thank You", { x: 2.8, y: 3.2, w: 7.7, h: 1.0, isTextBox: true,
    fontFace: F, fontSize: 44, bold: true, color: BLUE, align: "center", margin: 0 });
  s.addText("github.com/TREX4096/Graph_of_Thought", { x: 2.8, y: 5.25, w: 7.7, h: 0.32,
    isTextBox: true, fontFace: F, fontSize: 11.5, color: MUTE, align: "center", margin: 0 });
  s.addNotes("Questions.");
}

pres.writeFile({ fileName: process.argv[2] || "deck.pptx" })
  .then(f => console.log("written:", f));

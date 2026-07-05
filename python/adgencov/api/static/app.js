/* ADGENCOV dashboard (Phase D) — a zero-build single-page client for the
 * FastAPI service.  It submits an analysis (uploaded matrix or GEO accession),
 * polls the async job to completion, and renders the recommendation, estimator
 * ranking, gene blocks, a covariance heatmap, and the covariance network.
 *
 * No framework, no build step: the API serves this file directly.  Same-origin
 * requests, so the API base is empty. */
(function () {
  "use strict";

  var API = ""; // same origin as the served dashboard
  var POLL_MS = 800;
  var POLL_MAX = 300; // ~4 min ceiling

  // Distinct, colour-blind-friendly-ish palette for block colouring.
  var PALETTE = [
    "#5db2ff", "#7ee0b8", "#ffcf7a", "#ff8fa3", "#b79bff",
    "#8fd0ff", "#ffb26b", "#6be7d8", "#e0a3ff", "#c3e88d",
    "#f78c6c", "#89ddff", "#f07178", "#c792ea", "#addb67"
  ];
  function blockColor(i) { return PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length]; }

  function $(id) { return document.getElementById(id); }

  // -- shared render state -------------------------------------------------
  // The most recent analysis result + its block ordering + heatmap geometry,
  // so the edge/cell click handlers and the async symbol relabel can all reach
  // the same data without re-plumbing it through every function.
  var LAST = null;            // { result, bo }
  var HEATMAP = null;         // { perm, p, cell } — pixel→cell mapping for clicks
  var SYMBOLS = {};           // original gene id -> {symbol, name, rna_type, ...}

  // Display symbol for a gene id (falls back to the raw id until resolved).
  function displayName(gene) {
    var s = SYMBOLS[gene];
    return (s && s.symbol) ? s.symbol : gene;
  }
  function geneMeta(gene) { return SYMBOLS[gene] || null; }
  // Coarse RNA class for node shaping / labels (unknown → protein_coding).
  function rnaType(gene) {
    var s = SYMBOLS[gene];
    if (s && s.rna_type) { return s.rna_type; }
    // Heuristic fallback before symbols resolve, mirroring the backend.
    if (/^(hsa-)?(mir|let-?7)/i.test(gene)) { return "miRNA"; }
    if (/^(snord|snora|scarna)/i.test(gene)) { return "snoRNA"; }
    return "protein_coding";
  }

  // -- source tabs ---------------------------------------------------------
  var currentSource = "geo";
  function selectSource(src) {
    currentSource = src;
    var tabs = document.querySelectorAll(".tab");
    for (var i = 0; i < tabs.length; i++) {
      var active = tabs[i].getAttribute("data-source") === src;
      tabs[i].classList.toggle("active", active);
      tabs[i].setAttribute("aria-selected", active ? "true" : "false");
    }
    $("fields-geo").classList.toggle("hidden", src !== "geo");
    $("fields-upload").classList.toggle("hidden", src !== "upload");
  }

  // -- status helpers ------------------------------------------------------
  function setStatus(msg, cls) {
    var el = $("status");
    el.textContent = msg || "";
    el.className = "status" + (cls ? " " + cls : "");
  }
  function busy(on) { $("run-btn").disabled = on; }

  // -- progress bar --------------------------------------------------------
  var _progressStart = 0;
  function showProgress(on) {
    var w = $("progress-wrap");
    if (!w) { return; }
    w.classList.toggle("hidden", !on);
    w.setAttribute("aria-hidden", on ? "false" : "true");
    if (on) { _progressStart = Date.now(); setProgress(0, "Queued…", "running"); }
  }
  function setProgress(fraction, phase, cls) {
    var bar = $("progress-bar"), ph = $("progress-phase"), pct = $("progress-pct");
    if (!bar) { return; }
    var f = fraction == null ? 0 : Math.max(0, Math.min(1, fraction));
    var pctVal = Math.round(f * 100);
    bar.style.width = pctVal + "%";
    bar.className = "progress-bar" + (cls ? " " + cls : "");
    // Unknown fraction (0 while running) → indeterminate stripe animation.
    bar.classList.toggle("indeterminate", cls === "running" && f === 0);
    var secs = _progressStart ? Math.round((Date.now() - _progressStart) / 1000) : 0;
    if (ph) { ph.textContent = (phase || "Working…") + (secs ? "  ·  " + secs + "s" : ""); }
    if (pct) { pct.textContent = pctVal + "%"; }
  }

  // -- request assembly ----------------------------------------------------
  function commonParams() {
    return {
      n_genes: parseInt($("n_genes").value, 10),
      min_mean: parseFloat($("min_mean").value),
      log_transform: $("log_transform").checked,
      group: $("group").value,
      n_blocks: parseInt($("n_blocks").value, 10),
      top_fraction: parseFloat($("top_fraction").value),
      // Fast mode scores the estimator grid with 10-fold CV instead of exact
      // leave-one-out — much faster on large sample counts. null = exact LOO.
      cv_folds: $("fast_cv").checked ? 10 : null
    };
  }

  function submitGeo() {
    var body = commonParams();
    body.accession = $("accession").value.trim();
    body.force = $("force").checked;
    if (!body.accession) { throw new Error("Enter a GEO accession (e.g. GSE52778)."); }
    return fetch(API + "/analyze/geo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  }

  function submitUpload() {
    var f = $("file").files[0];
    if (!f) { throw new Error("Choose a matrix file to upload."); }
    var p = commonParams();
    var fd = new FormData();
    fd.append("file", f);
    fd.append("n_genes", p.n_genes);
    fd.append("min_mean", p.min_mean);
    fd.append("log_transform", p.log_transform);
    fd.append("group", p.group);
    fd.append("n_blocks", p.n_blocks);
    fd.append("top_fraction", p.top_fraction);
    if (p.cv_folds != null) { fd.append("cv_folds", p.cv_folds); }
    fd.append("sample_regex", $("sample_regex").value);
    fd.append("gene_col", $("gene_col").value);
    return fetch(API + "/analyze/upload", { method: "POST", body: fd });
  }

  // -- job polling ---------------------------------------------------------
  function pollJob(id, tries) {
    tries = tries || 0;
    return fetch(API + "/jobs/" + encodeURIComponent(id))
      .then(function (r) {
        if (!r.ok) { throw new Error("job lookup failed (" + r.status + ")"); }
        return r.json();
      })
      .then(function (job) {
        if (job.state === "succeeded") { setProgress(1, "Complete", "ok"); return job.result; }
        if (job.state === "failed") { throw new Error(job.error || "analysis failed"); }
        if (tries >= POLL_MAX) { throw new Error("timed out waiting for the job"); }
        setProgress(job.progress, job.phase || (job.state === "pending" ? "Queued…" : "Working…"), "running");
        setStatus(job.phase ? job.phase : "Running… (" + job.state + ")", null);
        return new Promise(function (res) { setTimeout(res, POLL_MS); })
          .then(function () { return pollJob(id, tries + 1); });
      });
  }

  function onSubmit(ev) {
    ev.preventDefault();
    var req;
    try { req = currentSource === "geo" ? submitGeo() : submitUpload(); }
    catch (e) { setStatus(e.message, "err"); return; }

    busy(true);
    showProgress(true);
    setStatus("Submitting…", null);
    window.ADGENCOV._lastSource = currentSource === "geo"
      ? ("GEO " + $("accession").value.trim())
      : "uploaded matrix";
    req.then(function (r) {
      return r.json().then(function (body) {
        if (r.status !== 202) {
          var d = body && body.detail ? JSON.stringify(body.detail) : ("HTTP " + r.status);
          throw new Error(d);
        }
        return body;
      });
    })
    .then(function (summary) { setStatus("Queued job " + summary.id.slice(0, 8) + "…", null); return pollJob(summary.id); })
    .then(function (result) { setStatus("Done.", "ok"); setProgress(1, "Complete", "ok"); render(result); })
    .catch(function (e) { setStatus(e.message || String(e), "err"); setProgress(0, "Failed", "err"); })
    .then(function () { busy(false); setTimeout(function () { showProgress(false); }, 1200); });
  }

  // -- GEO keyword search --------------------------------------------------
  function runGeoSearch() {
    var term = $("geo-search-term").value.trim();
    var status = $("geo-search-status");
    var list = $("geo-search-results");
    if (!term) { status.textContent = "Enter one or more keywords."; status.className = "status err"; return; }
    status.textContent = "Searching GEO…"; status.className = "status";
    list.classList.add("hidden"); list.innerHTML = "";
    $("geo-search-btn").disabled = true;
    fetch(API + "/search/geo?term=" + encodeURIComponent(term) + "&retmax=20")
      .then(function (r) { return r.json().then(function (b) {
        if (!r.ok) { throw new Error((b && b.detail) || ("HTTP " + r.status)); } return b; }); })
      .then(function (body) {
        var hits = body.hits || [];
        if (!hits.length) { status.textContent = "No GEO series matched."; return; }
        status.textContent = hits.length + " series found — click one to use it.";
        status.className = "status ok";
        renderGeoHits(hits);
      })
      .catch(function (e) { status.textContent = e.message || String(e); status.className = "status err"; })
      .then(function () { $("geo-search-btn").disabled = false; });
  }

  function renderGeoHits(hits) {
    var list = $("geo-search-results");
    list.innerHTML = "";
    for (var i = 0; i < hits.length; i++) {
      var h = hits[i];
      var li = document.createElement("li");
      li.className = "hit";
      li.setAttribute("data-acc", h.accession);
      var meta = [h.taxon, (h.n_samples ? h.n_samples + " samples" : ""), h.gds_type]
        .filter(function (x) { return x; }).join(" · ");
      li.innerHTML =
        '<div class="hit-head"><span class="acc">' + esc(h.accession) + "</span>" +
        '<span class="hit-title">' + esc(h.title) + "</span></div>" +
        '<div class="hit-meta">' + esc(meta) + "</div>";
      li.addEventListener("click", (function (acc) {
        return function () {
          $("accession").value = acc;
          $("geo-search-status").textContent = "Selected " + acc + " — press Run analysis.";
          $("geo-search-status").className = "status ok";
          var items = document.querySelectorAll("#geo-search-results .hit");
          for (var j = 0; j < items.length; j++) {
            items[j].classList.toggle("selected", items[j].getAttribute("data-acc") === acc);
          }
        };
      })(h.accession));
      list.appendChild(li);
    }
    list.classList.remove("hidden");
  }

  // -- protein id translation ----------------------------------------------
  function runTranslate() {
    var raw = $("protein-ids").value || "";
    var ids = raw.split(/[\s,;]+/).map(function (s) { return s.trim(); })
      .filter(function (s) { return s; });
    var status = $("protein-status");
    var table = $("protein-table");
    if (!ids.length) { status.textContent = "Enter one or more identifiers."; status.className = "status err"; return; }
    if (ids.length > 500) { status.textContent = "Please limit to 500 ids per request."; status.className = "status err"; return; }
    status.textContent = "Translating " + ids.length + " id(s)…"; status.className = "status";
    table.classList.add("hidden");
    $("protein-btn").disabled = true;
    fetch(API + "/translate/proteins", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: ids, source: $("protein-source").value })
    })
      .then(function (r) { return r.json().then(function (b) {
        if (!r.ok) { throw new Error((b && b.detail) || ("HTTP " + r.status)); } return b; }); })
      .then(function (body) {
        renderProteins(body.results || []);
        status.textContent = body.matched + " of " + body.count + " resolved.";
        status.className = "status " + (body.matched ? "ok" : "err");
      })
      .catch(function (e) { status.textContent = e.message || String(e); status.className = "status err"; })
      .then(function () { $("protein-btn").disabled = false; });
  }

  function renderProteins(results) {
    var table = $("protein-table");
    var tb = table.getElementsByTagName("tbody")[0];
    tb.innerHTML = "";
    for (var i = 0; i < results.length; i++) {
      var p = results[i];
      var tr = document.createElement("tr");
      if (!p.matched) { tr.className = "unmatched"; }
      var link = p.url
        ? '<a href="' + esc(p.url) + '" target="_blank" rel="noopener">' + esc(p.accession) + "</a>"
        : "—";
      tr.innerHTML =
        "<td>" + esc(p.query) + "</td>" +
        "<td>" + (p.matched ? esc(p.name) : '<span class="hint">not found</span>') + "</td>" +
        "<td>" + esc(p.gene || "") + "</td>" +
        "<td><em>" + esc(p.organism || "") + "</em></td>" +
        "<td>" + link + "</td>";
      tb.appendChild(tr);
    }
    table.classList.remove("hidden");
  }

  // -- block ordering ------------------------------------------------------
  // Returns display order of gene indices grouped by block label, plus the
  // block groups themselves (label -> ordered member indices).
  function blockOrder(labels) {
    var groups = {}, order = [];
    for (var i = 0; i < labels.length; i++) {
      var L = labels[i];
      if (!groups.hasOwnProperty(L)) { groups[L] = []; order.push(L); }
      groups[L].push(i);
    }
    order.sort(function (a, b) { return a - b; });
    var perm = [];
    for (var k = 0; k < order.length; k++) { perm = perm.concat(groups[order[k]]); }
    return { perm: perm, order: order, groups: groups };
  }

  // -- render dispatch -----------------------------------------------------
  function render(result) {
    $("results").classList.remove("hidden");
    SYMBOLS = {};                       // reset relabel state for the new run
    LAST = { result: result, bo: blockOrder(result.labels) };
    renderRecommendation(result);
    renderRanking(result.ranking);
    renderBlocks(result, LAST.bo);
    renderHeatmap(result, LAST.bo);
    renderNetwork(result, LAST.bo);
    // Clear any stale click-detail panels from a previous analysis.
    hide($("edge-detail")); hide($("cell-detail"));
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
    resolveSymbols(result.genes);       // async: relabels once mygene answers
  }

  function hide(el) { if (el) { el.classList.add("hidden"); el.innerHTML = ""; } }

  // -- gene id -> symbol relabel (async, non-blocking) ---------------------
  function resolveSymbols(genes) {
    var status = $("symbol-status");
    if (!genes || !genes.length) { return; }
    status.textContent = "Resolving " + genes.length + " gene symbols…";
    fetch(API + "/translate/symbols", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: genes, species: $("species").value })
    })
      .then(function (r) { return r.json().then(function (b) {
        if (!r.ok) { throw new Error((b && b.detail) || ("HTTP " + r.status)); } return b; }); })
      .then(function (body) {
        var results = body.results || [];
        for (var i = 0; i < results.length; i++) {
          var s = results[i];
          if (s.matched) { SYMBOLS[s.query] = s; }
        }
        status.textContent = body.matched + " of " + body.count + " symbols resolved.";
        // Re-render the textual/label surfaces now that symbols are known;
        // the canvas/svg geometry is unchanged so we only refresh labels.
        if (LAST) {
          renderBlocks(LAST.result, LAST.bo);
          renderNetwork(LAST.result, LAST.bo);
        }
      })
      .catch(function (e) { status.textContent = "symbol lookup failed: " + (e.message || e); });
  }

  function renderRecommendation(result) {
    var best = result.ranking[0] || {};
    // The API payload (AnalysisResult.to_dict) carries no source field, so we
    // stash the submitted job's kind/label on the last render for an honest
    // "GEO GSE…" vs "uploaded matrix" label instead of guessing.
    var srcTxt = (window.ADGENCOV && window.ADGENCOV._lastSource) || "analysis";
    $("recommendation").innerHTML =
      '<span class="badge">' + esc(result.recommended) + "</span>" +
      '<span class="meta">recommended estimator &middot; LOO&nbsp;NLL ' +
        fmt(best.loo_nll) + " &middot; " + result.n_genes + " genes &middot; " +
        (result.labels ? new Set(result.labels).size : 0) + " blocks &middot; " +
        esc(srcTxt) + "</span>";
  }

  function renderRanking(ranking) {
    var tb = $("ranking-table").getElementsByTagName("tbody")[0];
    tb.innerHTML = "";
    for (var i = 0; i < ranking.length; i++) {
      var r = ranking[i];
      var tr = document.createElement("tr");
      if (i === 0) { tr.className = "best"; }
      tr.innerHTML =
        "<td>" + (i + 1) + "</td>" +
        "<td>" + esc(r.method) + "</td>" +
        "<td>" + esc(paramStr(r.params)) + "</td>" +
        '<td class="num">' + fmt(r.loo_nll) + "</td>" +
        '<td class="num">' + fmt(r.condition_number, 1) + "</td>";
      tb.appendChild(tr);
    }
  }

  function renderBlocks(result, bo) {
    var host = $("blocks");
    host.innerHTML = "";
    $("block-count").textContent = "(" + bo.order.length + ")";
    for (var k = 0; k < bo.order.length; k++) {
      var L = bo.order[k];
      var members = bo.groups[L].map(function (idx) { return displayName(result.genes[idx]); });
      var div = document.createElement("div");
      div.className = "block";
      div.innerHTML =
        '<div class="block-head"><span class="swatch" style="background:' + blockColor(k) + '"></span>' +
        "<strong>Block " + L + "</strong> <span class=\"hint\">(" + members.length + " genes)</span></div>" +
        '<div class="genes">' + esc(members.join(", ")) + "</div>";
      host.appendChild(div);
    }
  }

  // -- covariance heatmap --------------------------------------------------
  function renderHeatmap(result, bo) {
    var canvas = $("heatmap");
    var ctx = canvas.getContext("2d");
    var note = $("heatmap-note");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    var cov = result.covariance;
    if (!cov || !cov.length) {
      HEATMAP = null;
      hide($("cell-detail"));
      note.textContent = "Covariance matrix omitted for large gene sets (> payload cap); network view still reflects the top edges.";
      ctx.fillStyle = "#93a1b5";
      ctx.font = "13px sans-serif";
      ctx.fillText("heatmap unavailable for this size", 90, canvas.height / 2);
      return;
    }
    note.textContent = "";

    var perm = bo.perm;
    var p = perm.length;
    // Largest absolute off-diagonal drives the colour scale (diagonal often
    // dominates and would wash out the structure).
    var maxAbs = 1e-12;
    for (var a = 0; a < p; a++) {
      for (var b = 0; b < p; b++) {
        if (a === b) { continue; }
        var v = Math.abs(cov[perm[a]][perm[b]]);
        if (v > maxAbs) { maxAbs = v; }
      }
    }
    var cell = Math.max(1, Math.floor(canvas.width / p));
    var dim = cell * p;
    // Keep the canvas crisp at the drawn size.
    canvas.width = dim; canvas.height = dim;
    // Remember the pixel→cell mapping so a click can recover (row, col) genes.
    HEATMAP = { perm: perm, p: p, cell: cell };
    for (var i = 0; i < p; i++) {
      for (var j = 0; j < p; j++) {
        var val = cov[perm[i]][perm[j]];
        ctx.fillStyle = divergingColor(val / maxAbs);
        ctx.fillRect(j * cell, i * cell, cell, cell);
      }
    }
    // Block boundary lines.
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1;
    var acc = 0;
    for (var g = 0; g < bo.order.length; g++) {
      acc += bo.groups[bo.order[g]].length;
      var pos = acc * cell + 0.5;
      ctx.beginPath(); ctx.moveTo(pos, 0); ctx.lineTo(pos, dim); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(dim, pos); ctx.stroke();
    }
    $("heatmap-legend").innerHTML =
      '<span>&minus;' + fmt(maxAbs, 2) + '</span><span class="bar"></span><span>+' + fmt(maxAbs, 2) + "</span>";
  }

  // Diverging blue-(dark)-red colour map for t in [-1, 1].
  function divergingColor(t) {
    if (t > 1) { t = 1; } if (t < -1) { t = -1; }
    var lo = [59, 111, 224], mid = [13, 20, 32], hi = [224, 86, 59];
    var c0, c1, m;
    if (t >= 0) { c0 = mid; c1 = hi; m = t; } else { c0 = mid; c1 = lo; m = -t; }
    var r = Math.round(c0[0] + (c1[0] - c0[0]) * m);
    var g = Math.round(c0[1] + (c1[1] - c0[1]) * m);
    var b = Math.round(c0[2] + (c1[2] - c0[2]) * m);
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  // -- covariance network --------------------------------------------------
  // Nodes placed on a circle, grouped into contiguous arcs by block, so block
  // structure is visible without a physics simulation.  Edges from the payload
  // are drawn with width/opacity by |covariance|.
  function renderNetwork(result, bo) {
    var svg = $("network");
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    $("edge-count").textContent = "(" + (result.edges ? result.edges.length : 0) + " edges)";

    var W = 440, H = 440, cx = W / 2, cy = H / 2, R = 180;
    var perm = bo.perm, p = perm.length;
    if (!p) { return; }

    // gene index -> position and colour (colour by its block's palette slot).
    var blockSlot = {};
    for (var s = 0; s < bo.order.length; s++) { blockSlot[bo.order[s]] = s; }
    var pos = {}, color = {};
    for (var i = 0; i < p; i++) {
      var gi = perm[i];
      var ang = (i / p) * 2 * Math.PI - Math.PI / 2;
      pos[gi] = { x: cx + R * Math.cos(ang), y: cy + R * Math.sin(ang) };
      color[gi] = blockColor(blockSlot[result.labels[gi]]);
    }

    var geneIndex = {};
    for (var gidx = 0; gidx < result.genes.length; gidx++) { geneIndex[result.genes[gidx]] = gidx; }

    var edges = result.edges || [];
    var maxAbs = 1e-12;
    for (var e = 0; e < edges.length; e++) { if (edges[e].abs_covariance > maxAbs) { maxAbs = edges[e].abs_covariance; } }

    // Node degree (from the shown edges) drives which nodes get a text label —
    // just the top hubs, so the view stays legible like the reference figure.
    var degree = {};
    for (var d = 0; d < edges.length; d++) {
      degree[edges[d].gene_a] = (degree[edges[d].gene_a] || 0) + 1;
      degree[edges[d].gene_b] = (degree[edges[d].gene_b] || 0) + 1;
    }
    var byDeg = result.genes.slice().sort(function (a, b) { return (degree[b] || 0) - (degree[a] || 0); });
    var labelSet = {};
    var nLabels = Math.min(12, p);
    for (var t = 0; t < nLabels; t++) { if (degree[byDeg[t]]) { labelSet[byDeg[t]] = true; } }

    var NS = "http://www.w3.org/2000/svg";
    var tip = $("net-tooltip");

    // edges first (under the nodes), each clickable to open the detail panel.
    for (var k = 0; k < edges.length; k++) {
      var ed = edges[k];
      var ia = geneIndex[ed.gene_a], ib = geneIndex[ed.gene_b];
      if (ia === undefined || ib === undefined || !pos[ia] || !pos[ib]) { continue; }
      var w = ed.abs_covariance / maxAbs;
      var line = document.createElementNS(NS, "line");
      line.setAttribute("x1", pos[ia].x.toFixed(1)); line.setAttribute("y1", pos[ia].y.toFixed(1));
      line.setAttribute("x2", pos[ib].x.toFixed(1)); line.setAttribute("y2", pos[ib].y.toFixed(1));
      line.setAttribute("stroke", ed.covariance >= 0 ? "#e0563b" : "#3b6fe0");
      line.setAttribute("stroke-width", (0.4 + 2.4 * w).toFixed(2));
      line.setAttribute("stroke-opacity", (0.15 + 0.6 * w).toFixed(2));
      (function (edge, ln) {
        ln.addEventListener("click", function () { showEdgeDetail(edge); });
        ln.addEventListener("mouseover", function () { ln.classList.add("hot"); });
        ln.addEventListener("mouseout", function () { ln.classList.remove("hot"); });
      })(ed, line);
      svg.appendChild(line);
    }

    // nodes — shaped by RNA class (circle=gene, square=miRNA, diamond=snoRNA).
    for (var n = 0; n < p; n++) {
      var g = perm[n];
      var gene = result.genes[g];
      var r = p > 120 ? 2.4 : 4.2;
      // Hubs render a touch larger, echoing the size-by-degree reference figure.
      if (labelSet[gene]) { r += 2.2; }
      var shape = nodeShape(NS, rnaType(gene), pos[g].x, pos[g].y, r, color[g]);
      shape.setAttribute("class", "node");
      shape.setAttribute("data-gene", gene);
      shape.setAttribute("data-block", result.labels[g]);
      (function (gid) {
        shape.addEventListener("mousemove", function (ev) {
          var m = geneMeta(gid);
          tip.textContent = displayName(gid) +
            (m && m.rna_type ? "  ·  " + m.rna_type : "") +
            "  ·  block " + result.labels[geneIndex[gid]];
          tip.style.left = (ev.clientX + 12) + "px";
          tip.style.top = (ev.clientY + 12) + "px";
          tip.classList.remove("hidden");
        });
        shape.addEventListener("mouseout", function () { tip.classList.add("hidden"); });
      })(gene);
      svg.appendChild(shape);

      if (labelSet[gene]) {
        var txt = document.createElementNS(NS, "text");
        txt.setAttribute("class", "node-label");
        txt.setAttribute("x", (pos[g].x + r + 1).toFixed(1));
        txt.setAttribute("y", (pos[g].y + 3).toFixed(1));
        txt.textContent = displayName(gene);
        svg.appendChild(txt);
      }
    }
  }

  // Build an SVG node marker shaped by RNA class, centred at (x, y).
  function nodeShape(NS, type, x, y, r, fill) {
    var el;
    if (type === "miRNA") {
      el = document.createElementNS(NS, "rect");
      el.setAttribute("x", (x - r).toFixed(1)); el.setAttribute("y", (y - r).toFixed(1));
      el.setAttribute("width", (2 * r).toFixed(1)); el.setAttribute("height", (2 * r).toFixed(1));
      el.setAttribute("rx", "0.8");
    } else if (type === "snoRNA") {
      el = document.createElementNS(NS, "polygon");
      var q = (r * 1.25).toFixed(1);
      el.setAttribute("points",
        x + "," + (y - q) + " " + (x + q) + "," + y + " " + x + "," + (+y + +q) + " " + (x - q) + "," + y);
    } else {
      el = document.createElementNS(NS, "circle");
      el.setAttribute("cx", x.toFixed(1)); el.setAttribute("cy", y.toFixed(1));
      el.setAttribute("r", r.toFixed(1));
    }
    el.setAttribute("fill", fill);
    return el;
  }

  // -- edge detail (click an edge) -----------------------------------------
  // Renders one endpoint of an edge/cell: symbol, RNA tag, full name, raw id.
  function endpointHtml(gene) {
    var m = geneMeta(gene);
    var sym = displayName(gene);
    var name = m && m.name ? m.name : "";
    var sub = [name, gene !== sym ? gene : ""].filter(function (x) { return x; }).join(" · ");
    return '<div class="endpoint">' +
      '<div class="sym">' + esc(sym) +
        '<span class="rna-tag">' + esc(rnaType(gene)) + "</span></div>" +
      '<div class="sub">' + esc(sub || "—") + "</div></div>";
  }

  function showEdgeDetail(ed) {
    var panel = $("edge-detail");
    var sign = ed.covariance >= 0 ? "pos" : "neg";
    panel.innerHTML =
      '<h3>Edge &middot; covariance <span class="cov ' + sign + '">' + fmt(ed.covariance, 3) + "</span></h3>" +
      '<div class="pair">' + endpointHtml(ed.gene_a) +
        '<div class="link">&mdash;</div>' + endpointHtml(ed.gene_b) + "</div>";
    panel.classList.remove("hidden");
  }

  // -- heatmap cell detail (click a cell) → STRING interactions ------------
  function onHeatmapClick(ev) {
    if (!HEATMAP || !LAST) { return; }
    var canvas = $("heatmap");
    var rect = canvas.getBoundingClientRect();
    if (!rect.width) { return; }
    var scale = canvas.width / rect.width;          // CSS px → canvas px
    var col = Math.floor((ev.clientX - rect.left) * scale / HEATMAP.cell);
    var row = Math.floor((ev.clientY - rect.top) * scale / HEATMAP.cell);
    if (row < 0 || col < 0 || row >= HEATMAP.p || col >= HEATMAP.p) { return; }
    showCellDetail(HEATMAP.perm[row], HEATMAP.perm[col]);
  }

  function showCellDetail(gi, gj) {
    var result = LAST.result;
    var geneA = result.genes[gi], geneB = result.genes[gj];
    var cov = result.covariance[gi][gj];
    var same = gi === gj;
    var sign = cov >= 0 ? "pos" : "neg";
    var panel = $("cell-detail");
    panel.innerHTML =
      "<h3>" + (same ? "Diagonal &middot; variance " : "Cell &middot; covariance ") +
        '<span class="cov ' + sign + '">' + fmt(cov, 3) + "</span></h3>" +
      '<div class="pair">' + endpointHtml(geneA) +
        (same ? "" : '<div class="link">&mdash;</div>' + endpointHtml(geneB)) + "</div>" +
      '<button type="button" class="mini" id="cell-string-btn">Search STRING interactions</button>' +
      '<div class="interactions" id="cell-interactions"></div>';
    panel.classList.remove("hidden");
    var genes = same ? [displayName(geneA)] : [displayName(geneA), displayName(geneB)];
    $("cell-string-btn").addEventListener("click", function () { fetchInteractions(genes); });
  }

  function fetchInteractions(genes) {
    var host = $("cell-interactions");
    var btn = $("cell-string-btn");
    host.innerHTML = '<p class="hint">Searching STRING-db…</p>';
    if (btn) { btn.disabled = true; }
    fetch(API + "/interactions?genes=" + encodeURIComponent(genes.join(",")) +
          "&species=" + encodeURIComponent($("species").value))
      .then(function (r) { return r.json().then(function (b) {
        if (!r.ok) { throw new Error((b && b.detail) || ("HTTP " + r.status)); } return b; }); })
      .then(function (data) { renderInteractions(host, data); })
      .catch(function (e) { host.innerHTML = '<p class="status err">' + esc(e.message || String(e)) + "</p>"; })
      .then(function () { if (btn) { btn.disabled = false; } });
  }

  function renderInteractions(host, data) {
    var partners = data.partners || [];
    if (!partners.length) {
      host.innerHTML = '<p class="hint">No STRING interactions above the medium-confidence threshold.</p>';
      return;
    }
    var html = "";
    if (data.direct != null) {
      html += '<div class="direct">Direct interaction &middot; STRING score ' + fmt(data.direct, 3) + "</div>";
    }
    html += "<ul>";
    for (var i = 0; i < partners.length; i++) {
      var p = partners[i];
      var pct = Math.max(0, Math.min(100, Math.round((p.score || 0) * 100)));
      html +=
        '<li><span class="pn">' + esc(p.query) + '</span><span class="arrow">&rarr;</span>' +
        '<span class="pn">' + esc(p.partner) + "</span>" +
        '<span class="meter"><span style="width:' + pct + '%"></span></span>' +
        '<span class="sc">' + fmt(p.score, 2) + "</span></li>";
    }
    html += "</ul>";
    host.innerHTML = html;
  }

  // -- formatting helpers --------------------------------------------------
  function fmt(x, digits) {
    if (x === null || x === undefined || (typeof x === "number" && !isFinite(x))) { return "—"; }
    digits = digits === undefined ? 4 : digits;
    var n = Number(x);
    if (Math.abs(n) >= 1e5 || (n !== 0 && Math.abs(n) < 1e-4)) { return n.toExponential(2); }
    return n.toFixed(digits);
  }
  function paramStr(params) {
    if (!params) { return "—"; }
    var keys = Object.keys(params);
    if (!keys.length) { return "—"; }
    return keys.map(function (k) { return k + "=" + fmt(params[k], 3); }).join(", ");
  }
  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // -- version footer ------------------------------------------------------
  function loadVersion() {
    fetch(API + "/health").then(function (r) { return r.json(); })
      .then(function (h) { $("version").textContent = "v" + h.version; })
      .catch(function () {});
  }

  // -- wire up -------------------------------------------------------------
  function init() {
    var tabs = document.querySelectorAll(".tab");
    for (var i = 0; i < tabs.length; i++) {
      tabs[i].addEventListener("click", function (ev) { selectSource(ev.target.getAttribute("data-source")); });
    }
    $("analyze-form").addEventListener("submit", onSubmit);
    // GEO keyword search
    $("geo-search-btn").addEventListener("click", runGeoSearch);
    $("geo-search-term").addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") { ev.preventDefault(); runGeoSearch(); }
    });
    // Protein id translator
    $("protein-btn").addEventListener("click", runTranslate);
    // Interactive results: click a heatmap cell; re-resolve on organism change.
    $("heatmap").addEventListener("click", onHeatmapClick);
    $("species").addEventListener("change", function () {
      hide($("edge-detail")); hide($("cell-detail"));
      if (LAST) { resolveSymbols(LAST.result.genes); }
    });
    selectSource("geo");
    loadVersion();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }

  // Exported for tests / debugging (harmless in the browser).
  window.ADGENCOV = { blockOrder: blockOrder, divergingColor: divergingColor, fmt: fmt, _lastSource: null };
})();

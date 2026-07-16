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

  // -- network view mode ---------------------------------------------------
  // The covariance network renders either as the 2D SVG (renderNetwork) or a
  // WebGL 3D force graph (renderNetwork3D, via the vendored 3d-force-graph).
  var NET_MODE = "3d";        // "3d" | "2d" (falls back to 2d without WebGL)
  var GRAPH3D = null;         // the lazily-created ForceGraph3D instance
  var NET_FROZEN = false;     // whether 3D node positions are pinned

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
    $("fields-multi").classList.toggle("hidden", src !== "multi");
  }

  // Accessions typed into the multi-dataset box, split on comma/space/newline.
  function multiAccessions() {
    return ($("multi-accessions").value || "")
      .split(/[\s,;]+/).map(function (s) { return s.trim().toUpperCase(); })
      .filter(function (s) { return s; });
  }

  function submitMulti() {
    var accs = multiAccessions();
    if (accs.length < 2) { throw new Error("Enter at least two GEO accessions."); }
    if (accs.length > 8) { throw new Error("At most 8 accessions per run."); }
    var body = commonParams();
    body.accessions = accs;
    body.force = false;
    var mode = $("multi-mode").value;           // combine | compare
    return fetch(API + "/analyze/" + mode, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
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
    // The selection criterion picks how the estimator grid is ranked, entirely
    // server-side: loo = exact leave-one-out, kfold = 10-fold CV (fast), ebic =
    // one-pass Extended BIC (fastest). cv_folds is only meaningful for kfold.
    var criterion = $("criterion").value;
    return {
      n_genes: parseInt($("n_genes").value, 10),
      min_mean: parseFloat($("min_mean").value),
      log_transform: $("log_transform").checked,
      group: $("group").value,
      n_blocks: parseInt($("n_blocks").value, 10),
      top_fraction: parseFloat($("top_fraction").value),
      criterion: criterion,
      ebic_gamma: parseFloat($("ebic_gamma").value),
      cv_folds: criterion === "kfold" ? 10 : null
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
    fd.append("criterion", p.criterion);
    fd.append("ebic_gamma", p.ebic_gamma);
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
    try {
      req = currentSource === "geo" ? submitGeo()
          : currentSource === "multi" ? submitMulti()
          : submitUpload();
    }
    catch (e) { setStatus(e.message, "err"); return; }

    busy(true);
    showProgress(true);
    setStatus("Submitting…", null);
    LAST_JOB_ID = null; hideEl($("export-panel"));
    window.ADGENCOV._lastSource = currentSource === "geo"
      ? ("GEO " + $("accession").value.trim())
      : currentSource === "multi"
        ? ($("multi-mode").value + ": " + multiAccessions().join(" + "))
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
    .then(function (summary) { LAST_JOB_ID = summary.id; setStatus("Queued job " + summary.id.slice(0, 8) + "…", null); return pollJob(summary.id); })
    .then(function (result) {
      setStatus("Done.", "ok"); setProgress(1, "Complete", "ok");
      // A compare run returns {datasets, comparison} rather than one analysis.
      if (result && result.comparison) { renderCompare(result); renderExports(true); }
      else { render(result); renderExports(false); }
    })
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
  // -- exports -------------------------------------------------------------
  // The server renders these with adgencov.export — the same module the
  // manuscript scripts use — so a downloaded table matches the paper exactly.
  var LAST_JOB_ID = null;

  var EXPORTS_SINGLE = [
    ["figure.pdf", "⬇ Download figure (PDF)"],
    ["figure.png", "⬇ Download figure (PNG)"],
    ["table.tex", "Estimator table (LaTeX)"],
    ["table.csv", "Estimator table (CSV)"],
    ["edges.csv", "Covariance edges (CSV)"],
    ["blocks.csv", "Gene blocks (CSV)"],
    ["covariance.csv", "Covariance matrix (CSV)"]
  ];
  var EXPORTS_COMPARE = [
    ["compare.tex", "Comparison table (LaTeX)"],
    ["compare.csv", "Comparison (CSV)"]
  ];

  // Rebuilds the whole panel, including its heading and note. It must not rely
  // on child nodes surviving: hide() clears innerHTML, so any child cached from
  // index.html can be gone by the time we render.
  function renderExports(isCompare) {
    var host = $("export-panel");
    if (!host || !LAST_JOB_ID) { return; }
    var list = isCompare ? EXPORTS_COMPARE : EXPORTS_SINGLE;
    var btns = list.map(function (x) {
      return '<a class="mini export-btn" download href="' + API + "/jobs/" +
        encodeURIComponent(LAST_JOB_ID) + "/export/" + x[0] + '">' + esc(x[1]) + "</a>";
    }).join("");
    host.innerHTML =
      '<h2>Export <span class="hint">publication-ready, generated by this run</span></h2>' +
      '<div class="export-row" id="export-buttons">' + btns + "</div>" +
      '<p class="hint" id="export-note">Rendered server-side by the same code the ' +
      "manuscript scripts use, so what you download is what the paper reports.</p>";
    host.classList.remove("hidden");
  }

  // -- multi-dataset views -------------------------------------------------
  // Banner shown above a pooled (combine) analysis: what was merged.
  function renderCombined(result) {
    var host = $("combined-panel");
    if (!host) { return; }
    var c = result.combined;
    if (!c) { hide(host); return; }
    var rows = c.datasets.map(function (d) {
      return "<tr><td>" + esc(d.accession) + "</td><td>" + esc(d.title || "") +
        '</td><td class="num">' + d.n_samples + "</td></tr>";
    }).join("");
    host.innerHTML =
      "<h2>Combined datasets <span class=\"hint\">(" + c.n_datasets + " series pooled)</span></h2>" +
      '<table class="hub-table"><thead><tr><th>Accession</th><th>Title</th><th>Samples</th></tr></thead>' +
      "<tbody>" + rows + "</tbody></table>" +
      '<p class="hint">Pooled <strong>' + c.n_samples_total + "</strong> samples · " +
      c.n_shared_genes + " genes shared by all series · " + c.n_genes_analyzed +
      " analyzed. Batch control: " + esc(c.batch_control) + ".</p>";
    host.classList.remove("hidden");
  }

  // Compare view: per-dataset recommendation + pairwise agreement.
  function renderCompare(result) {
    $("results").classList.remove("hidden");
    hide($("combined-panel"));
    // The per-analysis panels below don't apply to a compare run.
    ["ranking-table", "blocks"].forEach(function (id) {
      var t = $(id); if (t) { t.innerHTML = ""; }
    });
    $("recommendation").innerHTML =
      '<span class="badge">compare</span><span class="meta">' +
      result.datasets.length + " datasets · common panel of " +
      result.comparison.gene_panel_size + " genes drawn from " +
      result.comparison.n_shared_genes + " shared</span>";

    var c = result.comparison;
    var dsRows = result.datasets.map(function (d) {
      return "<tr><td>" + esc(d.accession) + "</td><td>" + esc(d.recommended) +
        '</td><td class="num">' + fmt(d.loo_nll, 3) + '</td><td class="num">' +
        d.n_samples + '</td><td class="num">' + d.n_edges + "</td></tr>";
    }).join("");
    var pairRows = c.pairs.map(function (p) {
      return "<tr><td>" + esc(p.a) + " vs " + esc(p.b) + '</td><td class="num">' +
        p.shared_edges + '</td><td class="num">' + fmt(p.edge_jaccard, 3) +
        '</td><td class="num">' + (p.sign_agreement == null ? "—" : fmt(p.sign_agreement, 2)) +
        "</td><td>" + (p.same_recommendation ? "yes" : "no") + "</td></tr>";
    }).join("");
    var rec = (c.recurrent_edges || []).slice(0, 12).map(function (e) {
      return '<li class="pair-row"><span class="pn">' + esc(e.gene_a) + "</span>" +
        '<span class="arrow">&harr;</span><span class="pn">' + esc(e.gene_b) +
        '</span><span class="cov">' + e.n_datasets + " datasets</span></li>";
    }).join("");

    $("compare-panel").innerHTML =
      "<h2>Dataset comparison</h2>" +
      '<table class="hub-table"><thead><tr><th>Dataset</th><th>Recommended</th>' +
      "<th>Best score</th><th>Samples</th><th>Edges</th></tr></thead><tbody>" + dsRows + "</tbody></table>" +
      '<p class="hint">Consensus recommendation: <strong>' +
      (c.consensus_recommendation ? esc(c.consensus_recommendation) : "none — datasets disagree") +
      "</strong></p>" +
      "<h3>Pairwise agreement</h3>" +
      '<table class="hub-table"><thead><tr><th>Pair</th><th>Shared edges</th><th>Jaccard</th>' +
      "<th>Sign agree</th><th>Same rec.</th></tr></thead><tbody>" + pairRows + "</tbody></table>" +
      "<h3>Edges recovered in more than one dataset <span class=\"hint\">(" +
      c.n_recurrent_edges + ")</span></h3>" +
      (rec ? '<ul class="pairs-list">' + rec + "</ul>"
           : '<p class="hint">No edge was recovered by more than one dataset at this threshold. ' +
             "Raising the edge fraction widens the comparison.</p>");
    $("compare-panel").classList.remove("hidden");
    hide($("edge-detail")); hide($("cell-detail"));
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function render(result) {
    $("results").classList.remove("hidden");
    hide($("compare-panel"));
    renderCombined(result);
    SYMBOLS = {};                       // reset relabel state for the new run
    LAST = { result: result, bo: blockOrder(result.labels) };
    renderRecommendation(result);
    renderRanking(result.ranking);
    renderBlocks(result, LAST.bo);
    renderHeatmap(result, LAST.bo);
    renderNetworkActive(result, LAST.bo);
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
          refreshNetworkLabels(LAST.result, LAST.bo);
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
    resetHeatmapZoom();                 // each new analysis starts unzoomed
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
    // Size cells from a fixed base, not canvas.width — the previous render
    // overwrote canvas.width with its (smaller) dim, so reading it back here
    // shrank the bitmap a little more on every run. CSS pins the display size
    // (max-width: 440px) regardless, so a larger base just buys sharper cells.
    var BASE = 880;
    var cell = Math.max(1, Math.floor(BASE / p));
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
  // Force-directed layout (Fruchterman-Reingold) with Louvain community
  // detection, echoing the reference figure: nodes coloured by community,
  // shaped by RNA class, sized by degree; edges tinted/scaled by |covariance|.
  // The expensive layout + community solve runs once per analysis and is cached
  // on the result object so the async symbol relabel only refreshes text.

  // Community palette — saturated, distinct, colour-blind-leaning.
  var COMMUNITY_COLORS = [
    "#4aa3ff", "#ff6b7d", "#5ed19b", "#f5c451", "#b98cff",
    "#ff9f56", "#5fd0e0", "#f78fb3", "#a3d65c", "#ff7ac0"
  ];
  function communityColor(i) {
    return COMMUNITY_COLORS[((i % COMMUNITY_COLORS.length) + COMMUNITY_COLORS.length) % COMMUNITY_COLORS.length];
  }
  // Purple (positive) / blue (negative) strength ramps, weak → very strong.
  var EDGE_RAMP_POS = ["#cbb8ef", "#a98fe0", "#8b5fd6", "#6a30b8"];
  var EDGE_RAMP_NEG = ["#b6cdf0", "#7ea3e2", "#5478d6", "#2f52b8"];
  var EDGE_WIDTH = [0.5, 1.1, 1.9, 2.9];
  var EDGE_OPACITY = [0.30, 0.48, 0.66, 0.88];
  var STRENGTH_LABELS = ["Weak", "Moderate", "Strong", "Very strong"];

  // Node marker sizing, shared between the collision layout and the renderer
  // so hub nodes never settle closer together than their drawn radius allows
  // (previously a flat 17px separation let big markers visibly overlap into
  // jagged, triangle-like blobs in dense hub clusters).
  var NODE_R_MIN = 3.0, NODE_R_SPAN = 5.5;
  // The snoRNA "diamond" marker's centre-to-vertex reach relative to its base
  // radius; was 1.25 (25% oversized vs. the circle/square markers, and the
  // main driver of the overlap), trimmed to keep it visually distinct but
  // no longer dominate the plot.
  var DIAMOND_SCALE = 1.05;
  function nodeRadius(deg, maxDeg) {
    return NODE_R_MIN + NODE_R_SPAN * Math.sqrt((deg || 0) / (maxDeg || 1));
  }
  // Effective on-screen reach used for collision separation (bigger for the
  // pointed diamond marker than for circles/squares of the same radius).
  function nodeReach(r, type) {
    return type === "snoRNA" ? r * DIAMOND_SCALE : r;
  }

  // Deterministic small hash of a gene id (stable layout seed, no Math.random).
  function geneHash(s) {
    var h = 2166136261;
    for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = (h * 16777619) >>> 0; }
    return h >>> 0;
  }

  // Greedy modularity community detection (Louvain local-moving level).
  // nodeIds: string[]; links: {a,b,w}[].  Returns id -> community index,
  // with communities relabelled 0..K-1 in descending size order.
  function louvain(nodeIds, links) {
    var n = nodeIds.length;
    var idx = {}; for (var i = 0; i < n; i++) { idx[nodeIds[i]] = i; }
    var adj = []; for (i = 0; i < n; i++) { adj[i] = {}; }
    var deg = new Array(n).fill(0);
    var m2 = 0;
    for (var e = 0; e < links.length; e++) {
      var a = idx[links[e].a], b = idx[links[e].b];
      if (a === undefined || b === undefined || a === b) { continue; }
      var w = links[e].w || 1e-6;
      adj[a][b] = (adj[a][b] || 0) + w;
      adj[b][a] = (adj[b][a] || 0) + w;
      deg[a] += w; deg[b] += w; m2 += 2 * w;
    }
    var comm = new Array(n);
    for (i = 0; i < n; i++) { comm[i] = i; }
    if (m2 === 0) { var solo = {}; for (i = 0; i < n; i++) { solo[nodeIds[i]] = i; } return solo; }
    var sigTot = deg.slice();
    var improved = true, passes = 0;
    while (improved && passes < 30) {
      improved = false; passes++;
      for (var v = 0; v < n; v++) {
        var cv = comm[v];
        sigTot[cv] -= deg[v];
        var neigh = {};
        for (var u in adj[v]) { var cu = comm[u]; neigh[cu] = (neigh[cu] || 0) + adj[v][u]; }
        var best = cv, bestGain = 0, ki = deg[v];
        // Staying put (contribution of v's own community minus itself) is the
        // baseline; only move for a strictly positive modularity gain.
        var stayKin = neigh[cv] || 0;
        for (var c in neigh) {
          var gain = neigh[c] - sigTot[c] * ki / m2;
          var rel = gain - (stayKin - sigTot[cv] * ki / m2);
          if (rel > bestGain + 1e-12) { bestGain = rel; best = +c; }
        }
        comm[v] = best; sigTot[best] += deg[v];
        if (best !== cv) { improved = true; }
      }
    }
    // Relabel by descending community size for stable, size-ordered colours.
    var counts = {};
    for (i = 0; i < n; i++) { counts[comm[i]] = (counts[comm[i]] || 0) + 1; }
    var order = Object.keys(counts).sort(function (x, y) { return counts[y] - counts[x]; });
    var relabel = {}; for (i = 0; i < order.length; i++) { relabel[order[i]] = i; }
    var out = {};
    for (i = 0; i < n; i++) { out[nodeIds[i]] = relabel[comm[i]]; }
    return out;
  }

  // Spring-electrical layout (D3-style): linear Hooke springs on edges +
  // inverse-square repulsion between all pairs, integrated with velocity
  // damping and a decaying alpha, centroid pinned each tick, then fit to the
  // viewport.  Linear springs (vs. FR's quadratic pull) are far more stable —
  // dense cliques settle at ~L spacing instead of collapsing to a point.
  // Deterministic: seeded from the per-node hash, no Math.random.
  function forceLayout(nodes, links, comm, W, H, degree) {
    var n = nodes.length;
    if (!n) { return; }
    var idOf = {}; for (var i = 0; i < n; i++) { idOf[nodes[i].id] = i; }
    var L = 0.90 * Math.sqrt((W * H) / n);          // ideal edge length
    var REP = L * L * 1.7;                            // repulsion charge
    var SPRING = 0.055;                               // Hooke stiffness
    var minD2 = (L * 0.35) * (L * 0.35);             // soften near-zero distance
    // Seed on a golden-angle spiral with a hash jitter (no coincident nodes).
    for (i = 0; i < n; i++) {
      var h = nodes[i].hash;
      var ang = i * 2.399963229;
      var rad = L * 0.6 * Math.sqrt(i + 1);
      nodes[i].x = rad * Math.cos(ang) + (((h % 1000) / 1000) - 0.5) * L;
      nodes[i].y = rad * Math.sin(ang) + ((((h >> 9) % 1000) / 1000) - 0.5) * L;
      nodes[i].vx = 0; nodes[i].vy = 0;
    }
    var iters = n > 400 ? 260 : 460;
    var alpha = 1.0, velDecay = 0.82;
    for (var it = 0; it < iters; it++) {
      // Repulsion — inverse-square, softened, scaled by alpha.
      for (i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
          var dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y;
          var d2 = dx * dx + dy * dy; if (d2 < minD2) { d2 = minD2; }
          var dist = Math.sqrt(d2);
          var rep = REP / d2 * alpha;                // push magnitude
          var ux = dx / dist, uy = dy / dist;
          nodes[i].vx -= ux * rep; nodes[i].vy -= uy * rep;
          nodes[j].vx += ux * rep; nodes[j].vy += uy * rep;
        }
      }
      // Springs — linear pull toward length L, stiffer for strong covariances.
      for (var e = 0; e < links.length; e++) {
        var a = idOf[links[e].a], b = idOf[links[e].b];
        if (a === undefined || b === undefined) { continue; }
        var lx = nodes[b].x - nodes[a].x, ly = nodes[b].y - nodes[a].y;
        var ld = Math.sqrt(lx * lx + ly * ly) || 0.01;
        var att = (ld - L) * SPRING * (0.5 + 0.9 * links[e].w) * alpha;
        var vx = lx / ld, vy = ly / ld;
        nodes[a].vx += vx * att; nodes[a].vy += vy * att;
        nodes[b].vx -= vx * att; nodes[b].vy -= vy * att;
      }
      // Integrate with damping.
      for (i = 0; i < n; i++) {
        nodes[i].x += nodes[i].vx; nodes[i].y += nodes[i].vy;
        nodes[i].vx *= velDecay; nodes[i].vy *= velDecay;
      }
      // Pin centroid so the graph cannot drift.
      var mx = 0, my = 0;
      for (i = 0; i < n; i++) { mx += nodes[i].x; my += nodes[i].y; }
      mx /= n; my /= n;
      for (i = 0; i < n; i++) { nodes[i].x -= mx; nodes[i].y -= my; }
      alpha *= 0.9925;
    }
    // Fit-to-viewport: rescale the settled layout into the canvas with margins.
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (i = 0; i < n; i++) {
      if (nodes[i].x < minX) { minX = nodes[i].x; } if (nodes[i].x > maxX) { maxX = nodes[i].x; }
      if (nodes[i].y < minY) { minY = nodes[i].y; } if (nodes[i].y > maxY) { maxY = nodes[i].y; }
    }
    var mar = 26;
    var spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
    var scale = Math.min((W - 2 * mar) / spanX, (H - 2 * mar) / spanY);
    var offX = (W - spanX * scale) / 2, offY = (H - spanY * scale) / 2;
    for (i = 0; i < n; i++) {
      nodes[i].x = offX + (nodes[i].x - minX) * scale;
      nodes[i].y = offY + (nodes[i].y - minY) * scale;
    }
    // Collision relaxation — guarantee legibility by pushing apart any nodes
    // closer than their drawn radii allow (dense cliques otherwise pack into
    // an unreadable ball, and — since this used to be a flat 17px regardless
    // of marker size — big hub markers, especially the pointed snoRNA
    // diamond, could overlap into jagged blobs). Separation now scales with
    // each node's actual on-screen reach, with a floor so small nodes still
    // get a comfortable gap.
    var maxDeg2 = 1;
    for (i = 0; i < n; i++) { var d0 = (degree && degree[nodes[i].id]) || 0; if (d0 > maxDeg2) { maxDeg2 = d0; } }
    var reach = new Array(n);
    for (i = 0; i < n; i++) {
      var dgI = (degree && degree[nodes[i].id]) || 0;
      reach[i] = nodeReach(nodeRadius(dgI, maxDeg2), rnaType(nodes[i].id));
    }
    var SEP_PAD = 6, SEP_MIN = 17;
    for (var pass = 0; pass < 60; pass++) {
      var moved = false;
      for (i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
          var SEP = Math.max(SEP_MIN, reach[i] + reach[j] + SEP_PAD), SEP2 = SEP * SEP;
          var cx = nodes[j].x - nodes[i].x, cy = nodes[j].y - nodes[i].y;
          var cd2 = cx * cx + cy * cy;
          if (cd2 >= SEP2) { continue; }
          var cd = Math.sqrt(cd2) || 0.01;
          // Deterministic separation direction when exactly coincident.
          var ndx = cd2 > 1e-6 ? cx / cd : Math.cos(nodes[i].hash);
          var ndy = cd2 > 1e-6 ? cy / cd : Math.sin(nodes[i].hash);
          var shove = (SEP - cd) / 2 + 0.5;
          nodes[i].x -= ndx * shove; nodes[i].y -= ndy * shove;
          nodes[j].x += ndx * shove; nodes[j].y += ndy * shove;
          moved = true;
        }
      }
      // Keep everything inside the frame.
      for (i = 0; i < n; i++) {
        nodes[i].x = Math.max(mar, Math.min(W - mar, nodes[i].x));
        nodes[i].y = Math.max(mar, Math.min(H - mar, nodes[i].y));
      }
      if (!moved) { break; }
    }
  }

  // Compute (or reuse cached) network geometry: connected nodes, degree,
  // communities, and force-directed positions.  Cached on the result object.
  function computeNetwork(result) {
    if (result.__net) { return result.__net; }
    var edges = result.edges || [];
    var W = 640, H = 480;
    var degree = {}, maxAbs = 1e-12, links = [];
    for (var e = 0; e < edges.length; e++) {
      var ed = edges[e];
      degree[ed.gene_a] = (degree[ed.gene_a] || 0) + 1;
      degree[ed.gene_b] = (degree[ed.gene_b] || 0) + 1;
      if (ed.abs_covariance > maxAbs) { maxAbs = ed.abs_covariance; }
    }
    for (e = 0; e < edges.length; e++) {
      links.push({ a: edges[e].gene_a, b: edges[e].gene_b, w: edges[e].abs_covariance / maxAbs });
    }
    // Only nodes that participate in a shown edge are drawn (matches figure).
    var nodeIds = [];
    for (var g in degree) { if (degree[g] > 0) { nodeIds.push(g); } }
    var comm = louvain(nodeIds, links);
    var nodes = nodeIds.map(function (id) { return { id: id, hash: geneHash(id) }; });
    forceLayout(nodes, links, comm, W, H, degree);
    var pos = {};
    for (var i = 0; i < nodes.length; i++) { pos[nodes[i].id] = { x: nodes[i].x, y: nodes[i].y }; }
    // Community sizes (for the legend) and community count.
    var commSize = {}, nComm = 0;
    for (i = 0; i < nodeIds.length; i++) {
      var c = comm[nodeIds[i]]; commSize[c] = (commSize[c] || 0) + 1;
      if (c + 1 > nComm) { nComm = c + 1; }
    }
    result.__net = {
      W: W, H: H, pos: pos, comm: comm, degree: degree, maxAbs: maxAbs,
      nodeIds: nodeIds, commSize: commSize, nComm: nComm, nEdges: edges.length
    };
    return result.__net;
  }

  function strengthBucket(w) {
    if (w >= 0.66) { return 3; }
    if (w >= 0.40) { return 2; }
    if (w >= 0.18) { return 1; }
    return 0;
  }

  function renderNetwork(result, bo) {
    var svg = $("network");
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    var edges = result.edges || [];
    $("edge-count").textContent = "(" + edges.length + " edges)";
    if (!edges.length) { renderNetLegend(null); renderHubTable(null, result); renderPairsPanel(null, result); return; }

    var net = computeNetwork(result);
    var pos = net.pos, comm = net.comm, degree = net.degree, maxAbs = net.maxAbs;
    var W = net.W, H = net.H;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    // Top hubs (by degree) get a text label so the view stays legible.
    var byDeg = net.nodeIds.slice().sort(function (a, b) { return (degree[b] || 0) - (degree[a] || 0); });
    var labelSet = {};
    for (var t = 0; t < Math.min(14, byDeg.length); t++) { labelSet[byDeg[t]] = true; }
    var maxDeg = degree[byDeg[0]] || 1;

    var NS = "http://www.w3.org/2000/svg";
    var tip = $("net-tooltip");
    var geneIndex = {};
    for (var gi = 0; gi < result.genes.length; gi++) { geneIndex[result.genes[gi]] = gi; }

    // Edges first (under the nodes), each clickable to open the detail panel.
    for (var k = 0; k < edges.length; k++) {
      var ed = edges[k];
      var pa = pos[ed.gene_a], pb = pos[ed.gene_b];
      if (!pa || !pb) { continue; }
      var w = ed.abs_covariance / maxAbs;
      var bkt = strengthBucket(w);
      var ramp = ed.covariance >= 0 ? EDGE_RAMP_POS : EDGE_RAMP_NEG;
      var line = document.createElementNS(NS, "line");
      line.setAttribute("x1", pa.x.toFixed(1)); line.setAttribute("y1", pa.y.toFixed(1));
      line.setAttribute("x2", pb.x.toFixed(1)); line.setAttribute("y2", pb.y.toFixed(1));
      line.setAttribute("stroke", ramp[bkt]);
      line.setAttribute("stroke-width", EDGE_WIDTH[bkt].toFixed(2));
      line.setAttribute("stroke-opacity", EDGE_OPACITY[bkt].toFixed(2));
      (function (edge, ln) {
        ln.addEventListener("click", function () { showEdgeDetail(edge); });
        ln.addEventListener("mouseover", function () { ln.classList.add("hot"); });
        ln.addEventListener("mouseout", function () { ln.classList.remove("hot"); });
      })(ed, line);
      svg.appendChild(line);
    }

    // Nodes — coloured by community, shaped by RNA class, sized by degree.
    for (var ni = 0; ni < net.nodeIds.length; ni++) {
      var gene = net.nodeIds[ni];
      var p = pos[gene];
      var deg = degree[gene] || 0;
      var r = nodeRadius(deg, maxDeg);
      var fill = communityColor(comm[gene]);
      var shape = nodeShape(NS, rnaType(gene), p.x, p.y, r, fill);
      shape.setAttribute("class", "node");
      shape.setAttribute("data-gene", gene);
      (function (gid) {
        shape.addEventListener("mousemove", function (ev) {
          var m = geneMeta(gid);
          tip.textContent = displayName(gid) +
            (m && m.rna_type ? "  ·  " + m.rna_type : "") +
            "  ·  degree " + (degree[gid] || 0) + "  ·  community " + (comm[gid] + 1);
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
        txt.setAttribute("x", (p.x + r + 1).toFixed(1));
        txt.setAttribute("y", (p.y + 3).toFixed(1));
        txt.textContent = displayName(gene);
        svg.appendChild(txt);
      }
    }

    renderNetLegend(net);
    renderHubTable(net, result);
    renderPairsPanel(net, result);
  }

  // -- network mode dispatch -----------------------------------------------
  // True when the browser can create a WebGL context (else 3D is unavailable).
  function webglSupported() {
    try {
      var c = document.createElement("canvas");
      return !!(window.WebGLRenderingContext &&
        (c.getContext("webgl") || c.getContext("experimental-webgl")));
    } catch (e) { return false; }
  }
  // 3D is only possible when both the vendored library loaded and WebGL exists.
  function canRender3D() { return !!window.ForceGraph3D && webglSupported(); }

  // Render the covariance network in whichever mode is active, toggling the
  // 2D SVG and the 3D canvas containers. Falls back to 2D when 3D is not
  // available (missing WebGL or the vendored bundle failed to load).
  function renderNetworkActive(result, bo) {
    var svg = $("network"), host3d = $("network-3d");
    var use3d = NET_MODE === "3d" && canRender3D();
    if (use3d) {
      hideEl(svg); showEl(host3d);
      renderNetwork3D(result);
    } else {
      hideEl(host3d); showEl(svg);
      if (NET_MODE === "3d") {
        setMode("2d");            // reflect the fallback in the toggle UI
        setNet3dNote("3D needs WebGL, which this browser/GPU didn't provide — showing the 2D view.");
      }
      renderNetwork(result, bo);
    }
  }

  // Refresh only the text surfaces (labels, legend, hub/pairs panels) after an
  // async symbol relabel, without disturbing the settled 3D layout.
  function refreshNetworkLabels(result, bo) {
    if (NET_MODE === "2d" || !canRender3D()) { renderNetwork(result, bo); return; }
    // 3D: node tooltips read displayName() live, so only the HTML panels + the
    // pinned camera-independent legend need refreshing here.
    var net = result.__net ? computeNetwork(result) : null;
    renderNetLegend(net); renderHubTable(net, result); renderPairsPanel(net, result);
  }

  function hideEl(el) { if (el) { el.classList.add("hidden"); } }
  function showEl(el) { if (el) { el.classList.remove("hidden"); } }
  function setNet3dNote(msg) { var n = $("net-3d-note"); if (n) { n.textContent = msg || ""; } }

  // Reflect the active mode in the toggle buttons.
  function setMode(mode) {
    NET_MODE = mode;
    var btns = document.querySelectorAll(".nmode");
    for (var i = 0; i < btns.length; i++) {
      var on = btns[i].getAttribute("data-mode") === mode;
      btns[i].classList.toggle("active", on);
      btns[i].setAttribute("aria-pressed", on ? "true" : "false");
    }
  }

  // -- 3D covariance network (vendored 3d-force-graph / three.js) ----------
  // Nodes coloured by Louvain community, sized by degree; edges coloured by the
  // sign of the covariance (purple positive / blue negative) and widened by its
  // strength, with animated particles flowing along the strongest edges. Orbit
  // with drag, zoom with the wheel, drag a node to reposition (and pin) it.
  function sizeGraph3D() {
    if (!GRAPH3D) { return; }
    var host = $("network-3d");
    var w = host.clientWidth || 640;
    GRAPH3D.width(w).height(480);
  }

  function renderNetwork3D(result) {
    var host = $("network-3d");
    var edges = result.edges || [];
    $("edge-count").textContent = "(" + edges.length + " edges)";
    setNet3dNote("Drag to orbit · scroll to zoom · drag a node to move it · Freeze to pin the layout.");
    if (!edges.length) {
      if (GRAPH3D) { GRAPH3D.graphData({ nodes: [], links: [] }); }
      renderNetLegend(null); renderHubTable(null, result); renderPairsPanel(null, result);
      return;
    }

    var net = computeNetwork(result);
    var comm = net.comm, degree = net.degree, maxAbs = net.maxAbs;
    var maxDeg = 1;
    for (var g in degree) { if (degree[g] > maxDeg) { maxDeg = degree[g]; } }

    var nodes = net.nodeIds.map(function (id) {
      var deg = degree[id] || 0;
      return {
        id: id, comm: comm[id], deg: deg, rtype: rnaType(id),
        color: communityColor(comm[id]),
        val: 0.6 + 5.0 * Math.sqrt(deg / maxDeg)   // marker volume ~ degree
      };
    });
    var links = edges.map(function (ed) {
      var w = ed.abs_covariance / maxAbs;
      var bkt = strengthBucket(w);
      var ramp = ed.covariance >= 0 ? EDGE_RAMP_POS : EDGE_RAMP_NEG;
      return {
        source: ed.gene_a, target: ed.gene_b, ga: ed.gene_a, gb: ed.gene_b,
        covariance: ed.covariance, bkt: bkt,
        color: ramp[bkt], width: EDGE_WIDTH[bkt], particles: bkt >= 2 ? 2 : 0
      };
    });

    if (!GRAPH3D) {
      try {
      GRAPH3D = window.ForceGraph3D()(host)
        .backgroundColor("#0b0f18")
        .showNavInfo(false)
        .nodeRelSize(4)
        .nodeVal(function (n) { return n.val; })
        .nodeColor(function (n) { return n.color; })
        .nodeOpacity(0.95)
        .nodeResolution(12)
        .nodeLabel(function (n) {
          return displayName(n.id) + "  ·  " + rnaType(n.id) +
            "  ·  degree " + n.deg + "  ·  community " + (n.comm + 1);
        })
        .linkColor(function (l) { return l.color; })
        .linkWidth(function (l) { return l.width * 0.4; })
        .linkOpacity(0.55)
        .linkDirectionalParticles(function (l) { return l.particles; })
        .linkDirectionalParticleWidth(1.1)
        .linkDirectionalParticleSpeed(0.006)
        .onNodeClick(focusNode3D)
        .onNodeDragEnd(function (n) { n.fx = n.x; n.fy = n.y; n.fz = n.z; })
        .onLinkClick(function (l) {
          showEdgeDetail({ gene_a: l.ga, gene_b: l.gb, covariance: l.covariance });
        });
      sizeGraph3D();
      window.addEventListener("resize", sizeGraph3D);
      } catch (err) {
        // WebGL/3D failed to initialise (some tablets, blocked GPU): fall back
        // to the 2D SVG so the network is never blank.
        GRAPH3D = null;
        setMode("2d");
        hideEl(host); showEl($("network"));
        setNet3dNote("3D view isn't available on this device — showing the 2D graph.");
        renderNetwork(result, blockOrder(result.labels));
        return;
      }
    }
    NET_FROZEN = false;
    setFreezeLabel();
    GRAPH3D.graphData({ nodes: nodes, links: links });
    // Fit the fresh layout into view once it has had a moment to expand.
    GRAPH3D.onEngineStop(function () {
      GRAPH3D.zoomToFit(600, 24);
      GRAPH3D.onEngineStop(function () {});   // fit only once per analysis
    });

    renderNetLegend(net);
    renderHubTable(net, result);
    renderPairsPanel(net, result);
  }

  // Fly the camera to look at a clicked node from a short distance.
  function focusNode3D(node) {
    if (!GRAPH3D) { return; }
    var d = 120;
    var r = Math.hypot(node.x, node.y, node.z) || 1;
    var k = 1 + d / r;
    GRAPH3D.cameraPosition(
      { x: node.x * k, y: node.y * k, z: node.z * k }, node, 1200);
  }

  // Pin (or release) every node so the structure can be orbited as a fixed
  // object ("fix it in place and zoom around on it").
  function toggleFreeze() {
    if (!GRAPH3D) { return; }
    NET_FROZEN = !NET_FROZEN;
    var data = GRAPH3D.graphData();
    for (var i = 0; i < data.nodes.length; i++) {
      var n = data.nodes[i];
      if (NET_FROZEN) { n.fx = n.x; n.fy = n.y; n.fz = n.z; }
      else { n.fx = null; n.fy = null; n.fz = null; }
    }
    if (!NET_FROZEN && GRAPH3D.d3ReheatSimulation) { GRAPH3D.d3ReheatSimulation(); }
    setFreezeLabel();
  }
  function setFreezeLabel() {
    var b = $("net-freeze");
    if (b) { b.textContent = NET_FROZEN ? "Unfreeze" : "Freeze"; b.classList.toggle("active", NET_FROZEN); }
  }

  // Legend: node shapes, edge-strength ramp, and Louvain communities + sizes.
  function renderNetLegend(net) {
    var host = $("net-legend");
    if (!host) { return; }
    if (!net) { host.innerHTML = ""; return; }
    var html = "";
    if (NET_MODE === "3d" && canRender3D()) {
      // In 3D every node is a sphere; RNA class shows on hover, so the legend
      // describes the colour/size encoding instead of the 2D marker shapes.
      html += '<div class="lg-group"><div class="lg-title">Nodes</div>' +
        '<div class="lg-row"><span class="lg-dot" style="background:var(--muted)"></span>Sphere per gene</div>' +
        '<div class="lg-row">Colour = community</div>' +
        '<div class="lg-row">Size = degree</div>' +
        '<div class="lg-row"><span class="hint">RNA type on hover</span></div></div>';
    } else {
      html += '<div class="lg-group"><div class="lg-title">Node type</div>' +
        '<div class="lg-row"><span class="lg-shape circle"></span>Protein-coding gene</div>' +
        '<div class="lg-row"><span class="lg-shape square"></span>miRNA</div>' +
        '<div class="lg-row"><span class="lg-shape diamond"></span>snoRNA</div></div>';
    }
    html += '<div class="lg-group"><div class="lg-title">Edge strength</div>';
    for (var s = 0; s < 4; s++) {
      html += '<div class="lg-row"><span class="lg-line" style="background:' + EDGE_RAMP_POS[s] +
        ';height:' + (EDGE_WIDTH[s] + 0.6).toFixed(1) + 'px"></span>' + STRENGTH_LABELS[s] + "</div>";
    }
    html += "</div>";
    html += '<div class="lg-group"><div class="lg-title">Community (Louvain)</div>';
    for (var c = 0; c < net.nComm; c++) {
      html += '<div class="lg-row"><span class="lg-dot" style="background:' + communityColor(c) +
        '"></span>Community ' + (c + 1) + '<span class="lg-n">n = ' + (net.commSize[c] || 0) + "</span></div>";
    }
    html += '<div class="lg-total">Total nodes = ' + net.nodeIds.length +
      ' &middot; Edges shown = ' + net.nEdges + "</div>";
    html += "</div>";
    host.innerHTML = html;
  }

  // "Top hub nodes by degree" table, mirroring the reference figure.
  function renderHubTable(net, result) {
    var host = $("hub-panel");
    if (!host) { return; }
    if (!net) { host.classList.add("hidden"); return; }
    var byDeg = net.nodeIds.slice().sort(function (a, b) { return (net.degree[b] || 0) - (net.degree[a] || 0); });
    var rows = "";
    var typeLabel = { protein_coding: "Protein-coding", miRNA: "miRNA", snoRNA: "snoRNA" };
    for (var i = 0; i < Math.min(8, byDeg.length); i++) {
      var g = byDeg[i];
      var tp = rnaType(g);
      rows += "<tr>" +
        '<td><span class="lg-dot" style="background:' + communityColor(net.comm[g]) + '"></span>' +
          esc(displayName(g)) + "</td>" +
        "<td>" + esc(typeLabel[tp] || tp) + "</td>" +
        '<td class="num">' + (net.degree[g] || 0) + "</td></tr>";
    }
    host.innerHTML =
      "<h3>Top hub nodes by degree</h3>" +
      '<table class="hub-table"><thead><tr><th>Node</th><th>Type</th><th>Degree</th></tr></thead>' +
      "<tbody>" + rows + "</tbody></table>";
    host.classList.remove("hidden");
  }

  // "Top gene pairs" — the strongest edges by |covariance|, as a plain
  // clickable list (not a graph) so a pair can be searched against STRING-db
  // in one click instead of having to find and click its edge in the plot.
  function renderPairsPanel(net, result) {
    var host = $("pairs-panel");
    if (!host) { return; }
    if (!net) { host.classList.add("hidden"); return; }
    var edges = (result.edges || []).slice()
      .sort(function (a, b) { return b.abs_covariance - a.abs_covariance; });
    var top = edges.slice(0, Math.min(12, edges.length));
    var rows = "";
    for (var i = 0; i < top.length; i++) {
      var ed = top[i];
      var sign = ed.covariance >= 0 ? "pos" : "neg";
      rows += '<li class="pair-row" data-i="' + i + '" tabindex="0" role="button">' +
        '<span class="pn">' + esc(displayName(ed.gene_a)) + "</span>" +
        '<span class="arrow">&harr;</span>' +
        '<span class="pn">' + esc(displayName(ed.gene_b)) + "</span>" +
        '<span class="cov ' + sign + '">' + fmt(ed.covariance, 3) + "</span></li>";
    }
    host.innerHTML =
      '<h3>Top gene pairs <span class="hint">click a pair to search STRING</span></h3>' +
      '<ul class="pairs-list" id="pairs-list">' + rows + "</ul>" +
      '<div class="interactions" id="pairs-interactions"></div>';
    host.classList.remove("hidden");
    var list = $("pairs-list");
    var items = list.querySelectorAll(".pair-row");
    var select = function (idx) {
      var all = list.querySelectorAll(".pair-row");
      for (var k = 0; k < all.length; k++) { all[k].classList.remove("active"); }
      items[idx].classList.add("active");
      var ed = top[idx];
      fetchInteractionsInto($("pairs-interactions"), [displayName(ed.gene_a), displayName(ed.gene_b)]);
    };
    for (var ii = 0; ii < items.length; ii++) {
      (function (idx, li) {
        li.addEventListener("click", function () { select(idx); });
        li.addEventListener("keydown", function (ev) {
          if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(idx); }
        });
      })(ii, items[ii]);
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
      // Keep q numeric: toFixed() returns a string, and the right vertex uses
      // (x + q), which would string-concatenate rather than add.
      var q = r * DIAMOND_SCALE;
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
    if (HM.moved) { HM.moved = false; return; }     // was a pan, not a cell click
    var canvas = $("heatmap");
    var rect = canvas.getBoundingClientRect();       // includes the zoom transform
    if (!rect.width) { return; }
    var scale = canvas.width / rect.width;          // CSS px → canvas px
    var col = Math.floor((ev.clientX - rect.left) * scale / HEATMAP.cell);
    var row = Math.floor((ev.clientY - rect.top) * scale / HEATMAP.cell);
    if (row < 0 || col < 0 || row >= HEATMAP.p || col >= HEATMAP.p) { return; }
    showCellDetail(HEATMAP.perm[row], HEATMAP.perm[col]);
  }

  // -- heatmap zoom / pan (wheel, pinch, drag) -----------------------------
  var HM = { s: 1, tx: 0, ty: 0, dragging: false, moved: false,
             lastX: 0, lastY: 0, pinchDist: 0, pinchCx: 0, pinchCy: 0 };
  function applyHeatmapTransform() {
    var c = $("heatmap");
    if (!c) { return; }
    c.style.transformOrigin = "0 0";
    c.style.transform = "translate(" + HM.tx.toFixed(1) + "px," + HM.ty.toFixed(1) +
      "px) scale(" + HM.s.toFixed(3) + ")";
  }
  function resetHeatmapZoom() { HM.s = 1; HM.tx = 0; HM.ty = 0; applyHeatmapTransform(); }
  // Keep the scaled canvas covering the wrapper (no empty gutters).
  function clampHeatmap() {
    var wrap = $("heatmap-wrap");
    if (!wrap) { return; }
    if (HM.s <= 1) { HM.tx = 0; HM.ty = 0; return; }
    var minTx = wrap.clientWidth * (1 - HM.s), minTy = wrap.clientHeight * (1 - HM.s);
    if (HM.tx > 0) { HM.tx = 0; } if (HM.tx < minTx) { HM.tx = minTx; }
    if (HM.ty > 0) { HM.ty = 0; } if (HM.ty < minTy) { HM.ty = minTy; }
  }
  // Zoom by *factor* keeping the point (cx, cy) (wrapper-relative px) fixed.
  function zoomHeatmapAt(cx, cy, factor) {
    var ns = Math.max(1, Math.min(10, HM.s * factor));
    var k = ns / HM.s;
    HM.tx = cx - k * (cx - HM.tx);
    HM.ty = cy - k * (cy - HM.ty);
    HM.s = ns;
    clampHeatmap();
    applyHeatmapTransform();
  }
  function initHeatmapZoom() {
    var wrap = $("heatmap-wrap");
    if (!wrap) { return; }
    wrap.addEventListener("wheel", function (e) {
      e.preventDefault();
      var r = wrap.getBoundingClientRect();
      zoomHeatmapAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.15 : 1 / 1.15);
    }, { passive: false });
    wrap.addEventListener("dblclick", function () { resetHeatmapZoom(); });
    // Mouse/trackpad drag to pan (only when zoomed in).
    wrap.addEventListener("pointerdown", function (e) {
      if (e.pointerType === "touch") { return; }
      HM.dragging = true; HM.moved = false; HM.lastX = e.clientX; HM.lastY = e.clientY;
    });
    wrap.addEventListener("pointermove", function (e) {
      if (!HM.dragging) { return; }
      var dx = e.clientX - HM.lastX, dy = e.clientY - HM.lastY;
      if (Math.abs(dx) + Math.abs(dy) > 2) { HM.moved = true; }
      if (HM.s > 1) { HM.tx += dx; HM.ty += dy; clampHeatmap(); applyHeatmapTransform(); }
      HM.lastX = e.clientX; HM.lastY = e.clientY;
    });
    window.addEventListener("pointerup", function () { HM.dragging = false; });
    // Touch: one finger pans (when zoomed), two fingers pinch-zoom.
    wrap.addEventListener("touchstart", function (e) {
      if (e.touches.length === 2) {
        var a = e.touches[0], b = e.touches[1], r = wrap.getBoundingClientRect();
        HM.pinchDist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
        HM.pinchCx = (a.clientX + b.clientX) / 2 - r.left;
        HM.pinchCy = (a.clientY + b.clientY) / 2 - r.top;
      } else if (e.touches.length === 1) {
        HM.lastX = e.touches[0].clientX; HM.lastY = e.touches[0].clientY; HM.moved = false;
      }
    }, { passive: false });
    wrap.addEventListener("touchmove", function (e) {
      if (e.touches.length === 2) {
        e.preventDefault();
        var a = e.touches[0], b = e.touches[1];
        var d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
        if (HM.pinchDist > 0) { zoomHeatmapAt(HM.pinchCx, HM.pinchCy, d / HM.pinchDist); }
        HM.pinchDist = d;
      } else if (e.touches.length === 1 && HM.s > 1) {
        e.preventDefault();
        var t = e.touches[0];
        HM.tx += t.clientX - HM.lastX; HM.ty += t.clientY - HM.lastY; HM.moved = true;
        HM.lastX = t.clientX; HM.lastY = t.clientY;
        clampHeatmap(); applyHeatmapTransform();
      }
    }, { passive: false });
    wrap.addEventListener("touchend", function (e) { if (e.touches.length === 0) { HM.pinchDist = 0; } });
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
    fetchInteractionsInto($("cell-interactions"), genes, $("cell-string-btn"));
  }

  // Shared STRING-lookup runner: renders into any host element, optionally
  // disabling a trigger button (cell-detail's button, or none for the top
  // gene pairs list, where the whole row is the trigger).
  function fetchInteractionsInto(host, genes, btn) {
    if (!host) { return; }
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
    // Network 2D/3D toggle + 3D camera controls.
    var nmodeBtns = document.querySelectorAll(".nmode");
    for (var m = 0; m < nmodeBtns.length; m++) {
      nmodeBtns[m].addEventListener("click", function (ev) {
        var mode = ev.currentTarget.getAttribute("data-mode");
        if (mode === NET_MODE) { return; }
        setMode(mode);
        if (LAST) { renderNetworkActive(LAST.result, LAST.bo); }
      });
    }
    $("net-freeze").addEventListener("click", toggleFreeze);
    $("net-reset").addEventListener("click", function () {
      if (NET_MODE === "3d" && GRAPH3D) { GRAPH3D.zoomToFit(600, 24); }
    });
    // Default to 3D on WebGL desktops; default to the reliable 2D view on
    // touch-primary devices (iPad/phone), where the 3D toggle is still offered.
    var coarsePointer = !!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
    setMode((canRender3D() && !coarsePointer) ? "3d" : "2d");
    initHeatmapZoom();
    // Reveal the EBIC penalty field only when the EBIC criterion is selected.
    function syncCriterion() {
      $("ebic-gamma-field").classList.toggle("hidden", $("criterion").value !== "ebic");
    }
    $("criterion").addEventListener("change", syncCriterion);
    syncCriterion();
    // Explain the active multi-dataset mode.
    function syncMultiMode() {
      var h = $("multi-mode-hint");
      if (!h) { return; }
      h.textContent = $("multi-mode").value === "combine"
        ? "Keeps genes shared by every series and standardizes each gene within its own dataset before pooling, so batch differences don't drive the covariance."
        : "Analyzes each series over one common gene panel, then reports estimator agreement and top-edge overlap.";
    }
    $("multi-mode").addEventListener("change", syncMultiMode);
    syncMultiMode();
    selectSource("geo");
    loadVersion();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }

  // Exported for tests / debugging (harmless in the browser).
  window.ADGENCOV = {
    blockOrder: blockOrder, divergingColor: divergingColor, fmt: fmt,
    louvain: louvain, forceLayout: forceLayout, geneHash: geneHash,
    strengthBucket: strengthBucket, computeNetwork: computeNetwork,
    _lastSource: null
  };
})();

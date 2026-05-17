/* debug_viz — split-pane gallery + OpenSeadragon viewer.
 *
 * Frame model (v3):
 *   frame.mode = "single" | "compare"
 *   frame.panels = [ { id, label, kind, dzi: {...}, render_id, ... }, ... ]
 *
 * Viewer pane shows N OpenSeadragon instances (1 for single, 2-3 for compare),
 * with synced viewports when N > 1.
 */
(() => {
  "use strict";

  // ---------- helpers ----------
  const $ = (id) => document.getElementById(id);
  const fmtNum = (n) => {
    if (n === null || n === undefined) return "?";
    if (typeof n !== "number" || !isFinite(n)) return String(n);
    const a = Math.abs(n);
    if (a !== 0 && (a < 0.01 || a >= 1e5)) return n.toExponential(3);
    return n.toFixed(4).replace(/\.?0+$/, "");
  };
  const shapeStr = (s) => Array.isArray(s) ? `[${s.join("×")}]` : "?";
  const wavelengthToRgb = (nm) => {
    // Mirrors spectral.py — kept compact for JS use (swatch colors only).
    let r=0, g=0, b=0;
    if (nm < 380) { const a=Math.max(0.2,1-(380-nm)/200); return [0.5*a,0,1*a]; }
    if (nm > 780) { const a=Math.max(0.15,1-(nm-780)/400); return [1*a,0,0]; }
    if (nm < 440) { r=-(nm-440)/60; g=0; b=1; }
    else if (nm < 490) { r=0; g=(nm-440)/50; b=1; }
    else if (nm < 510) { r=0; g=1; b=-(nm-510)/20; }
    else if (nm < 580) { r=(nm-510)/70; g=1; b=0; }
    else if (nm < 645) { r=1; g=-(nm-645)/65; b=0; }
    else { r=1; g=0; b=0; }
    let att=1;
    if (nm < 420) att = 0.3+0.7*(nm-380)/40;
    else if (nm > 700) att = 0.3+0.7*(780-nm)/80;
    return [r*att, g*att, b*att];
  };
  const rgbToCss = ([r,g,b]) =>
    `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;

  const debounce = (fn, ms) => {
    let t = null;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  };

  // ---------- DOM refs ----------
  const gallery = $("gallery");
  const statusEl = $("status");
  const countEl = $("count");
  const galleryToggle = $("gallery_toggle");
  const splitEl = $("split");
  const paneGallery = $("pane_gallery");
  const paneViewer = $("pane_viewer");
  const viewerEmpty = $("viewer_empty");
  const viewerRoot = $("viewer_root");
  const vTitle = $("v_title");
  const vKind = $("v_kind");
  const vDelete = $("v_delete");
  const vCanvasRow = $("v_canvas_row");
  const vControls = $("v_controls");
  const vMeta = $("v_meta");
  const tileTmpl = $("tile_tmpl");
  const panelTmpl = $("panel_tmpl");

  // ---------- state ----------
  const tiles = new Map();         // frame.id -> { frame, el }
  let activeFrameId = null;
  // active viewer panels: array of { panel, osd, container, syncSuppress }
  let activeViewers = [];
  let splitInstance = null;

  // ---------- init split ----------
  const initSplit = () => {
    splitInstance = Split([paneGallery, paneViewer], {
      sizes: [22, 78],
      minSize: [0, 240],
      gutterSize: 4,
      gutterAlign: "center",
      onDrag: () => activeViewers.forEach(v => v.osd && v.osd.viewport && v.osd.viewport.resize()),
    });
  };

  // ---------- gallery ----------
  const setCount = () => {
    countEl.textContent = `${tiles.size} frame${tiles.size === 1 ? "" : "s"}`;
  };
  const setBadge = (el, kind) => {
    el.textContent = kind || "";
    el.classList.remove("gray","mono","rgb","raw","multi","mask");
    if (kind) el.classList.add(kind);
  };
  const makeThumbImg = (assetId) => {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.decoding = "async";
    img.draggable = false;
    img.src = `/thumb/${assetId}`;
    img.addEventListener("load", () => img.classList.add("loaded"), { once: true });
    return img;
  };
  const panelStatsLine = (p) => {
    const sh = shapeStr(p.shape);
    const dt = p.dtype || "?";
    const s = p.stats || {};
    const nan = s.nan_pct ? ` nan=${s.nan_pct.toFixed(1)}%` : "";
    if (s.min === undefined || s.min === null) return `${sh} ${dt}`;
    return `${sh} ${dt}  min=${fmtNum(s.min)} max=${fmtNum(s.max)}${nan}`;
  };

  const createTile = (frame) => {
    const node = tileTmpl.content.firstElementChild.cloneNode(true);
    const thumbWrap = node.querySelector(".thumb-wrap");
    const panels = frame.panels || [];
    if (frame.mode === "compare" && panels.length > 1) {
      node.classList.add("compare");
      thumbWrap.style.gridTemplateColumns = `repeat(${panels.length}, 1fr)`;
      panels.forEach(p => thumbWrap.appendChild(makeThumbImg(p.id)));
    } else if (panels.length) {
      thumbWrap.appendChild(makeThumbImg(panels[0].id));
    }
    const labelEl = node.querySelector(".label");
    labelEl.textContent = frame.label || "";
    setBadge(node.querySelector(".tile-kind"), panels[0]?.kind);

    const statsEl = node.querySelector(".stats");
    if (frame.mode === "compare") {
      statsEl.textContent = panels.map(p => panelStatsLine(p)).join(" | ");
    } else {
      statsEl.textContent = panels[0] ? panelStatsLine(panels[0]) : "";
    }

    labelEl.addEventListener("click", e => e.stopPropagation());
    labelEl.addEventListener("keydown", e => {
      if (e.key === "Enter") { e.preventDefault(); labelEl.blur(); }
    });
    labelEl.addEventListener("blur", async () => {
      const nl = labelEl.textContent.trim();
      if (nl !== (frame.label || "")) await patchFrameLabel(frame.id, nl);
    });
    node.querySelector(".del").addEventListener("click", async e => {
      e.stopPropagation();
      if (confirm("Delete this frame?")) await deleteFrame(frame.id);
    });
    node.addEventListener("click", () => openFrame(frame.id));
    return node;
  };

  const addTile = (frame) => {
    if (tiles.has(frame.id)) return updateTileMeta(frame.id, frame, true);
    const el = createTile(frame);
    tiles.set(frame.id, { frame, el });
    gallery.insertBefore(el, gallery.firstChild);
    setCount();
  };
  const updateTileMeta = (id, partial, full=false) => {
    const t = tiles.get(id);
    if (!t) return;
    if (full) Object.assign(t.frame, partial);
    else if (partial.fields) Object.assign(t.frame, partial.fields);
    const labelEl = t.el.querySelector(".label");
    if (document.activeElement !== labelEl) labelEl.textContent = t.frame.label || "";
    if (activeFrameId === id && document.activeElement !== vTitle) {
      vTitle.textContent = t.frame.label || "";
    }
  };
  const removeTile = (id) => {
    const t = tiles.get(id);
    if (!t) return;
    t.el.classList.add("removing");
    setTimeout(() => { t.el.remove(); tiles.delete(id); setCount(); }, 220);
    if (activeFrameId === id) closeViewer();
  };

  // ---------- API ----------
  const patchFrameLabel = (id, label) =>
    fetch(`/frames/${id}`, {
      method: "PATCH", headers: { "content-type": "application/json" },
      body: JSON.stringify({ label }),
    }).catch(e => console.error("patch frame", e));
  const patchPanelLabel = (id, label) =>
    fetch(`/panels/${id}`, {
      method: "PATCH", headers: { "content-type": "application/json" },
      body: JSON.stringify({ label }),
    }).catch(e => console.error("patch panel", e));
  const deleteFrame = (id) =>
    fetch(`/frames/${id}`, { method: "DELETE" }).catch(e => console.error("del", e));
  const postRerender = (panelId, body) =>
    fetch(`/rerender/${panelId}`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json());

  // ---------- viewer pane ----------
  const dziSource = (panel) =>
    `/dzi/${panel.id}/${panel.render_id}.dzi`;

  const closeViewer = () => {
    for (const v of activeViewers) {
      try { v.osd && v.osd.destroy(); } catch (_) {}
    }
    activeViewers = [];
    vCanvasRow.innerHTML = "";
    vControls.innerHTML = "";
    vMeta.textContent = "";
    activeFrameId = null;
    viewerRoot.classList.add("hidden");
    viewerEmpty.classList.remove("hidden");
    // Remove tile active class
    for (const { el } of tiles.values()) el.classList.remove("active");
  };

  const buildOsd = (host, panel) => {
    const osd = OpenSeadragon({
      element: host,
      prefixUrl: "/static/osd-images/",
      tileSources: dziSource(panel),
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      navigatorHeight: 120,
      navigatorWidth: 120,
      navigatorAutoFade: false,
      showRotationControl: false,
      showFullPageControl: false,
      showHomeControl: false,
      showZoomControl: false,
      zoomPerScroll: 1.4,
      animationTime: 0.25,
      springStiffness: 9,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: false,
                              scrollToZoom: true, pinchToZoom: true,
                              flickEnabled: true },
      visibilityRatio: 0.6,
      minZoomImageRatio: 0.4,
      maxZoomPixelRatio: 16,
      imageSmoothingEnabled: false,
      placeholderFillStyle: "#0a0c10",
    });
    // Double-click = reset to home (replaces the default zoom-in behavior)
    osd.addHandler("canvas-double-click", () => {
      try { osd.viewport.goHome(); } catch (_) {}
    });
    return osd;
  };

  // Sync zoom + pan between viewers (compare mode).
  // Uses event payload (e.zoom, e.center) rather than polling viewport state,
  // and snaps slaves with immediately=true so no animation-time feedback loop.
  const wireSync = (viewers) => {
    if (viewers.length < 2) return;
    let suppress = false;
    const onZoom = (src) => (e) => {
      if (suppress) return;
      if (e == null || e.zoom == null) return;
      suppress = true;
      try {
        for (const v of viewers) {
          if (v === src) continue;
          try { v.osd.viewport.zoomTo(e.zoom, e.refPoint || null, true); }
          catch (_) {}
        }
      } finally { suppress = false; }
    };
    const onPan = (src) => (e) => {
      if (suppress) return;
      if (e == null || e.center == null) return;
      suppress = true;
      try {
        for (const v of viewers) {
          if (v === src) continue;
          try { v.osd.viewport.panTo(e.center, true); }
          catch (_) {}
        }
      } finally { suppress = false; }
    };
    for (const v of viewers) {
      v.osd.addHandler("zoom", onZoom(v));
      v.osd.addHandler("pan", onPan(v));
    }
  };

  // Re-open a viewer's DZI source while preserving its viewport bounds.
  const reopenWithViewport = (osd, dziUrl) => {
    let savedBounds = null;
    try { savedBounds = osd.viewport.getBounds(); } catch (_) {}
    osd.addOnceHandler("open", () => {
      if (savedBounds) {
        try { osd.viewport.fitBounds(savedBounds, true); }
        catch (_) {}
      }
    });
    osd.open(dziUrl);
  };

  // ---------- controls (per panel, optional global "apply to all") ----------
  const renderControls = (frame) => {
    vControls.innerHTML = "";
    const panels = frame.panels || [];
    // For compare: per-panel sub-controls inline. For single: one combined block.
    panels.forEach((p, idx) => {
      const block = document.createElement("div");
      block.className = "group";
      const lbl = document.createElement("label");
      lbl.textContent = panels.length > 1 ? `panel ${idx + 1}: ${p.label || ""}` : "render";
      block.appendChild(lbl);

      // Scale toggle + log_dr input (only shown when scale=log)
      const scaleRow = document.createElement("div");
      scaleRow.className = "row";
      const seg = document.createElement("div");
      seg.className = "seg";
      const logDrWrap = document.createElement("span");
      logDrWrap.className = "row";
      logDrWrap.style.gap = "4px";
      const logDrLbl = document.createElement("span");
      logDrLbl.textContent = "DR 10^";
      logDrLbl.style.color = "var(--muted)";
      logDrLbl.style.fontSize = "11px";
      const logDrInp = document.createElement("input");
      logDrInp.type = "number"; logDrInp.min = "1"; logDrInp.max = "12"; logDrInp.step = "0.5";
      logDrInp.placeholder = "off"; logDrInp.title = "Log dynamic range cap — clip values below max / 10^N";
      logDrInp.style.width = "56px";
      if (p.log_dr != null) logDrInp.value = String(p.log_dr);
      logDrInp.addEventListener("change", () => {
        const v = logDrInp.value === "" ? null : +logDrInp.value;
        rerender(p, { log_dr: v });
      });
      logDrWrap.appendChild(logDrLbl); logDrWrap.appendChild(logDrInp);
      const showLogDr = () => { logDrWrap.style.display = (p.scale === "log") ? "" : "none"; };
      ["linear", "log"].forEach(s => {
        const b = document.createElement("button");
        b.textContent = s;
        if ((p.scale || "linear") === s) b.classList.add("on");
        b.addEventListener("click", () => {
          seg.querySelectorAll("button").forEach(x => x.classList.remove("on"));
          b.classList.add("on");
          p.scale = s;
          showLogDr();
          rerender(p, { scale: s });
        });
        seg.appendChild(b);
      });
      scaleRow.appendChild(seg);
      scaleRow.appendChild(logDrWrap);
      showLogDr();
      block.appendChild(scaleRow);

      // Clip percentile — two number inputs (precise, no slider imprecision)
      const cp = p.clip_pct || [1.0, 99.0];
      const clipRow = document.createElement("div");
      clipRow.className = "row";
      const clipLbl = document.createElement("span");
      clipLbl.textContent = "clip %";
      clipLbl.style.color = "var(--muted)"; clipLbl.style.fontSize = "11px";
      const lo = document.createElement("input");
      const hi = document.createElement("input");
      const dash = document.createElement("span"); dash.textContent = "–";
      lo.type = "number"; lo.min = "0"; lo.max = "100"; lo.step = "0.1";
      lo.value = String(cp[0]); lo.style.width = "60px";
      hi.type = "number"; hi.min = "0"; hi.max = "100"; hi.step = "0.1";
      hi.value = String(cp[1]); hi.style.width = "60px";
      const applyClip = () => {
        let l = +lo.value, h = +hi.value;
        if (l < 0) l = 0; if (l > 100) l = 100;
        if (h < 0) h = 0; if (h > 100) h = 100;
        if (l > h) { const t = l; l = h; h = t; }
        lo.value = l; hi.value = h;
        rerender(p, { clip_pct: [l, h] });
      };
      lo.addEventListener("change", applyClip);
      hi.addEventListener("change", applyClip);
      clipRow.appendChild(clipLbl);
      clipRow.appendChild(lo); clipRow.appendChild(dash); clipRow.appendChild(hi);
      block.appendChild(clipRow);

      // Wavelength (mono / gray)
      if (p.kind === "mono" || p.kind === "gray") {
        const wRow = document.createElement("div");
        wRow.className = "row";
        const wl = document.createElement("input");
        wl.type = "number"; wl.min = "200"; wl.max = "1100"; wl.step = "1";
        wl.placeholder = "λ nm";
        if (p.wavelength != null) wl.value = String(p.wavelength);
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        if (p.wavelength != null) swatch.style.background = rgbToCss(wavelengthToRgb(+p.wavelength));
        const clearBtn = document.createElement("button");
        clearBtn.className = "ghost"; clearBtn.textContent = "off";
        clearBtn.style.padding = "2px 6px"; clearBtn.style.fontSize = "10px";
        const apply = () => {
          const v = wl.value === "" ? null : +wl.value;
          if (v != null) swatch.style.background = rgbToCss(wavelengthToRgb(v));
          else swatch.style.background = "transparent";
          rerender(p, { wavelength: v });
        };
        wl.addEventListener("change", apply);
        clearBtn.addEventListener("click", () => { wl.value = ""; apply(); });
        const wlbl = document.createElement("span");
        wlbl.textContent = "λ"; wlbl.style.color = "var(--muted)";
        wRow.appendChild(wlbl); wRow.appendChild(wl);
        wRow.appendChild(swatch); wRow.appendChild(clearBtn);
        block.appendChild(wRow);
      }

      // Multi: per-band wavelengths + band selector + strip
      if (p.kind === "multi" && p.bands_meta) {
        const nb = p.bands_meta.length;
        // Wavelengths editor
        const wlRow = document.createElement("div");
        wlRow.className = "row"; wlRow.style.flexWrap = "wrap";
        const wlInputs = [];
        for (let i = 0; i < nb; i++) {
          const chip = document.createElement("span");
          chip.className = "band-chip";
          const sw = document.createElement("span"); sw.className = "swatch";
          const inp = document.createElement("input");
          inp.type = "number"; inp.placeholder = "nm"; inp.min = "200"; inp.max = "1100";
          const wlv = (p.wavelengths || [])[i];
          if (wlv != null) {
            inp.value = String(wlv);
            sw.style.background = rgbToCss(wavelengthToRgb(+wlv));
          }
          inp.addEventListener("change", () => {
            const nbv = inp.value === "" ? null : +inp.value;
            if (nbv != null) sw.style.background = rgbToCss(wavelengthToRgb(nbv));
            else sw.style.background = "transparent";
            const wls = wlInputs.map(x => x.value === "" ? null : +x.value);
            const anyNull = wls.some(v => v == null);
            rerender(p, { wavelengths: anyNull ? null : wls });
          });
          chip.appendChild(sw); chip.appendChild(inp);
          wlRow.appendChild(chip); wlInputs.push(inp);
        }
        const wlLbl = document.createElement("div");
        wlLbl.style.fontSize = "10px"; wlLbl.style.color = "var(--muted)";
        wlLbl.textContent = "per-band λ (all set → composite by wavelength)";
        block.appendChild(wlLbl); block.appendChild(wlRow);

        // Band RGB picker
        const bRow = document.createElement("div");
        bRow.className = "row";
        const bandsLbl = document.createElement("span");
        bandsLbl.style.color = "var(--muted)"; bandsLbl.style.fontSize = "10px";
        bandsLbl.textContent = "RGB picks";
        const inputs = ["R","G","B"].map((_,k) => {
          const i = document.createElement("input");
          i.type = "number"; i.min = "0"; i.max = String(nb-1); i.step = "1";
          const cur = (p.bands || [0,1,2])[k] ?? (p.bands || [0])[0] ?? 0;
          i.value = String(Math.min(nb-1, cur));
          i.title = `${["Red","Green","Blue"][k]} channel band index`;
          i.style.width = "40px";
          return i;
        });
        const apply3 = debounce(() => {
          rerender(p, { bands: inputs.map(i => +i.value) });
        }, 200);
        inputs.forEach(i => i.addEventListener("change", apply3));
        bRow.appendChild(bandsLbl);
        ["R","G","B"].forEach((c,k) => {
          const w = document.createElement("span");
          w.textContent = c; w.style.color = ["#ff8090","#90ff90","#8090ff"][k];
          bRow.appendChild(w); bRow.appendChild(inputs[k]);
        });
        block.appendChild(bRow);

        // Band thumbnail strip — click to open that single band as new viewer
        const strip = document.createElement("div");
        strip.className = "band-strip-mini";
        p.bands_meta.forEach((bm, i) => {
          const item = document.createElement("div");
          item.className = "band-item";
          const im = document.createElement("img");
          im.src = `/thumb/${bm.asset}`;
          im.loading = "lazy";
          const lb = document.createElement("div");
          lb.className = "lbl"; lb.textContent = bm.label;
          item.appendChild(im); item.appendChild(lb);
          item.title = `band ${i}\n${bm.label}\nclick to view this band only`;
          // Click swaps the panel to single-band view (kind=mono) using bands=[i]
          item.addEventListener("click", () => {
            // Use existing wavelength if known
            const wl = (p.wavelengths || [])[i] ?? bm.wavelength ?? null;
            rerender(p, {
              kind: "mono",
              bands: [i],
              wavelength: wl,
              wavelengths: null,
            });
          });
          strip.appendChild(item);
        });
        block.appendChild(strip);
      }

      vControls.appendChild(block);
    });

    // Footer hint
    const hint = document.createElement("div");
    hint.className = "hint";
    hint.style.flex = "1"; hint.style.textAlign = "right";
    hint.textContent = "scroll = zoom @ cursor · drag = pan · dblclick = reset to home";
    vControls.appendChild(hint);
  };

  // Apply a rerender with given partial update, then swap DZI source live.
  const rerender = async (panel, body) => {
    try {
      const res = await postRerender(panel.id, body);
      if (!res || !res.panel) return;
      Object.assign(panel, res.panel);
      // find matching viewer
      const v = activeViewers.find(av => av.panel.id === panel.id);
      if (v) {
        reopenWithViewport(v.osd, dziSource(panel));
        updateViewerMeta();
      }
    } catch (e) { console.error("rerender failed", e); }
  };

  const updateViewerMeta = () => {
    if (!activeFrameId) return;
    const t = tiles.get(activeFrameId); if (!t) return;
    const f = t.frame;
    const parts = (f.panels || []).map(p => {
      const s = p.stats || {};
      const cu = p.clip || [];
      return `${p.label ? p.label + ": " : ""}${shapeStr(p.shape)} ${p.dtype||"?"} `
        + `min=${fmtNum(s.min)} max=${fmtNum(s.max)} `
        + `clip=[${fmtNum(cu[0])},${fmtNum(cu[1])}] scale=${p.scale}`
        + (p.note ? `  · ${p.note}` : "");
    });
    vMeta.textContent = parts.join("    ");
  };

  const openFrame = (id) => {
    const t = tiles.get(id);
    if (!t) return;
    // Re-click on the already-open frame is a no-op (would otherwise
    // duplicate the viewer columns).
    if (activeFrameId === id && activeViewers.length) return;
    if (activeFrameId && activeFrameId !== id) {
      // tear down previous
      for (const v of activeViewers) {
        try { v.osd.destroy(); } catch (_) {}
      }
      activeViewers = [];
      vCanvasRow.innerHTML = "";
      vControls.innerHTML = "";
    }
    activeFrameId = id;
    const f = t.frame;

    viewerEmpty.classList.add("hidden");
    viewerRoot.classList.remove("hidden");

    // active border
    for (const { el } of tiles.values()) el.classList.remove("active");
    t.el.classList.add("active");

    // Title + kind badge
    vTitle.textContent = f.label || "";
    setBadge(vKind, (f.panels || [])[0]?.kind);

    // Build per-panel viewer columns
    const panels = f.panels || [];
    panels.forEach((p) => {
      const node = panelTmpl.content.firstElementChild.cloneNode(true);
      const sub = node.querySelector(".vp-title");
      sub.value = p.label || "";
      sub.addEventListener("change", () => {
        const nl = sub.value.trim();
        if (nl !== (p.label || "")) {
          p.label = nl;
          patchPanelLabel(p.id, nl);
        }
      });
      const shapeEl = node.querySelector(".vp-shape");
      shapeEl.textContent = `${shapeStr(p.shape)} ${p.dtype || ""}`;
      const statsEl = node.querySelector(".vp-stats");
      const s = p.stats || {};
      statsEl.textContent = `min=${fmtNum(s.min)} max=${fmtNum(s.max)}`
        + (s.nan_pct ? ` nan=${s.nan_pct.toFixed(1)}%` : "");
      vCanvasRow.appendChild(node);
      const host = node.querySelector(".vp-osd");
      // OSD needs the host to have a size; defer to next frame
      requestAnimationFrame(() => {
        const osd = buildOsd(host, p);
        activeViewers.push({ panel: p, osd, container: node });
        if (activeViewers.length === panels.length) wireSync(activeViewers);
      });
    });

    renderControls(f);
    updateViewerMeta();
  };

  // ---- viewer-level editing handlers ----
  vTitle.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); vTitle.blur(); }
  });
  vTitle.addEventListener("blur", async () => {
    if (!activeFrameId) return;
    const t = tiles.get(activeFrameId); if (!t) return;
    const nl = vTitle.textContent.trim();
    if (nl !== (t.frame.label || "")) await patchFrameLabel(activeFrameId, nl);
  });
  vDelete.addEventListener("click", async () => {
    if (!activeFrameId) return;
    if (confirm("Delete this frame?")) await deleteFrame(activeFrameId);
  });
  galleryToggle.addEventListener("click", () => {
    splitEl.classList.toggle("gallery-hidden");
    setTimeout(() => activeViewers.forEach(v => v.osd && v.osd.viewport.resize()), 50);
  });

  document.addEventListener("keydown", e => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.isContentEditable
                     || e.target.tagName === "TEXTAREA")) return;
    if (e.key === "Escape" && activeFrameId) closeViewer();
    else if (e.key === "0" && activeViewers.length) {
      activeViewers[0].osd.viewport.goHome();
    } else if (e.key === "g") {
      splitEl.classList.toggle("gallery-hidden");
      setTimeout(() => activeViewers.forEach(v => v.osd && v.osd.viewport.resize()), 50);
    }
  });

  // ---------- WebSocket ----------
  // After (re)connect, re-fetch /frames and merge — guards against missing a
  // frame_added broadcast that fired before the WS handshake completed (race
  // hit hard on the first view() call, when the browser is still launching).
  const resyncFrames = async () => {
    try {
      const data = await fetch("/frames").then(r => r.json());
      const have = new Set(tiles.keys());
      for (const f of (data.frames || [])) {
        if (have.has(f.id)) {
          // Update in-place so render state (panel.dzi etc) is current
          updateTileMeta(f.id, f, true);
        } else {
          addTile(f);
        }
      }
    } catch (e) { console.warn("resync failed", e); }
  };

  let ws = null;
  const connect = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.addEventListener("open", () => {
      statusEl.textContent = "live"; statusEl.className = "status connected";
      resyncFrames();
    });
    ws.addEventListener("close", () => {
      statusEl.textContent = "offline"; statusEl.className = "status disconnected";
      setTimeout(connect, 1500);
    });
    ws.addEventListener("message", e => {
      let msg; try { msg = JSON.parse(e.data); } catch { return; }
      if (msg.kind === "frame_added") addTile(msg.payload);
      else if (msg.kind === "frame_updated") updateTileMeta(msg.payload.id, msg.payload);
      else if (msg.kind === "frame_deleted") removeTile(msg.payload.id);
      else if (msg.kind === "panel_updated") {
        // sub-title edit from elsewhere
        for (const { frame } of tiles.values()) {
          const p = (frame.panels || []).find(x => x.id === msg.payload.id);
          if (p) Object.assign(p, msg.payload.fields);
        }
      }
      else if (msg.kind === "panel_rerendered") {
        // Update panel state in our tile cache
        for (const { frame } of tiles.values()) {
          const idx = (frame.panels || []).findIndex(x => x.id === msg.payload.id);
          if (idx >= 0) frame.panels[idx] = msg.payload;
        }
      }
    });
  };

  // ---------- font-scale ----------
  const fontScaleEl = document.getElementById("font_scale");
  const applyFontScale = (fs) => {
    document.documentElement.setAttribute("data-fs", fs);
    fontScaleEl.querySelectorAll("button").forEach(b =>
      b.classList.toggle("on", b.dataset.fs === fs));
    try { localStorage.setItem("dbv_fs", fs); } catch (_) {}
  };
  fontScaleEl.addEventListener("click", e => {
    const b = e.target.closest("button[data-fs]");
    if (!b) return;
    applyFontScale(b.dataset.fs);
    // OSD needs to re-measure since header height may have changed
    setTimeout(() => activeViewers.forEach(v => v.osd && v.osd.viewport.resize()), 50);
  });
  applyFontScale(localStorage.getItem("dbv_fs") || "m");

  // ---------- boot ----------
  initSplit();
  fetch("/frames").then(r => r.json()).then(data => {
    for (const f of (data.frames || [])) addTile(f);
    connect();
  }).catch(() => connect());
})();

(() => {
  const gallery = document.getElementById("gallery");
  const statusEl = document.getElementById("status");
  const countEl = document.getElementById("count");
  const modal = document.getElementById("modal");
  const pzTarget = document.getElementById("pz_target");
  const pzHost = document.getElementById("pz_host");
  const mLabel = document.getElementById("m_label");
  const mKind = document.getElementById("m_kind");
  const mStats = document.getElementById("m_stats");
  const mNote = document.getElementById("m_note");
  const mDelete = document.getElementById("m_delete");
  const modalClose = document.getElementById("modal_close");
  const bandStrip = document.getElementById("band_strip");
  const tileTmpl = document.getElementById("tile_tmpl");

  const tiles = new Map();
  let activeFrameId = null;
  let pz = null;

  const fmtNum = (n) => {
    if (n === null || n === undefined) return "?";
    if (typeof n !== "number" || !isFinite(n)) return String(n);
    const a = Math.abs(n);
    if (a !== 0 && (a < 0.01 || a >= 1e5)) return n.toExponential(3);
    return n.toFixed(4).replace(/\.?0+$/, "");
  };
  const setCount = () => { countEl.textContent = `${tiles.size} frame${tiles.size === 1 ? "" : "s"}`; };
  const shapeStr = (s) => Array.isArray(s) ? `[${s.join("×")}]` : "?";

  const statsLine = (stats, shape, dtype) => {
    if (!stats || stats.min === undefined) return `${shapeStr(shape)} ${dtype||"?"}`;
    const nan = stats.nan_pct ? ` nan=${stats.nan_pct.toFixed(1)}%` : "";
    return `${shapeStr(shape)} ${dtype||"?"}  min=${fmtNum(stats.min)} max=${fmtNum(stats.max)}${nan}`;
  };

  const setBadge = (el, kind) => {
    el.textContent = kind || "";
    el.classList.remove("gray", "mono", "rgb", "raw", "multi", "mask");
    if (kind) el.classList.add(kind);
  };

  const makeThumbImg = (assetId, opts={}) => {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.decoding = "async";
    img.draggable = false;
    img.src = `/thumb/${assetId}`;
    img.addEventListener("load", () => img.classList.add("loaded"), { once: true });
    return img;
  };

  const renderTileThumb = (wrap, frame) => {
    wrap.innerHTML = "";
    if (frame.mode === "compare") {
      for (const aid of (frame.compare_assets || [])) wrap.appendChild(makeThumbImg(aid));
    } else if (frame.main_asset) {
      wrap.appendChild(makeThumbImg(frame.main_asset));
    }
  };

  const createTile = (frame) => {
    const node = tileTmpl.content.firstElementChild.cloneNode(true);
    if (frame.mode === "compare") node.classList.add("compare");
    renderTileThumb(node.querySelector(".thumb-wrap"), frame);
    const labelEl = node.querySelector(".label");
    labelEl.textContent = frame.label || "";
    setBadge(node.querySelector(".tile-kind"), frame.kind);

    const statsEl = node.querySelector(".stats");
    if (frame.mode === "compare") {
      const s = frame.stats || [{}, {}];
      const sh = frame.shape || [null, null];
      const dt = frame.dtype || [];
      statsEl.textContent =
        `${statsLine(s[0], sh[0], dt[0])} | ${statsLine(s[1], sh[1], dt[1])}`;
    } else {
      statsEl.textContent = statsLine(frame.stats, frame.shape, frame.dtype);
    }
    if (frame.note) {
      const noteSpan = document.createElement("div");
      noteSpan.textContent = frame.note;
      noteSpan.style.color = "#ffd27a";
      noteSpan.style.fontSize = "11px";
      noteSpan.style.marginTop = "2px";
      statsEl.appendChild(noteSpan);
    }

    labelEl.addEventListener("click", e => e.stopPropagation());
    labelEl.addEventListener("keydown", e => {
      if (e.key === "Enter") { e.preventDefault(); labelEl.blur(); }
    });
    labelEl.addEventListener("blur", async () => {
      const nl = labelEl.textContent.trim();
      if (nl !== (frame.label || "")) await patchLabel(frame.id, nl);
    });
    node.querySelector(".del").addEventListener("click", async e => {
      e.stopPropagation();
      if (confirm("Delete this frame?")) await deleteFrame(frame.id);
    });
    node.addEventListener("click", () => openModal(frame.id));
    return node;
  };

  const addTile = (frame) => {
    if (tiles.has(frame.id)) return updateTile(frame.id, frame, true);
    const el = createTile(frame);
    tiles.set(frame.id, { frame, el });
    gallery.insertBefore(el, gallery.firstChild);
    setCount();
  };

  const updateTile = (id, partial, fullReplace=false) => {
    const t = tiles.get(id);
    if (!t) return;
    if (fullReplace) Object.assign(t.frame, partial);
    else if (partial.fields) Object.assign(t.frame, partial.fields);
    const labelEl = t.el.querySelector(".label");
    if (document.activeElement !== labelEl) labelEl.textContent = t.frame.label || "";
    if (activeFrameId === id) mLabel.textContent = t.frame.label || "";
  };

  const removeTile = (id) => {
    const t = tiles.get(id);
    if (!t) return;
    t.el.classList.add("removing");
    setTimeout(() => { t.el.remove(); tiles.delete(id); setCount(); }, 220);
    if (activeFrameId === id) closeModal();
  };

  const patchLabel = (id, label) =>
    fetch(`/frames/${id}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ label }),
    }).catch(e => console.error("patch", e));

  const deleteFrame = (id) =>
    fetch(`/frames/${id}`, { method: "DELETE" }).catch(e => console.error("del", e));

  // Modal
  const swapMainImage = (assetId, thumbAssetId) => {
    pzTarget.innerHTML = "";
    if (thumbAssetId) {
      const ph = document.createElement("img");
      ph.src = `/thumb/${thumbAssetId}`;
      ph.classList.add("placeholder");
      ph.draggable = false;
      pzTarget.appendChild(ph);
    }
    const img = document.createElement("img");
    img.src = `/image/${assetId}`;
    img.decoding = "async";
    img.draggable = false;
    img.style.opacity = "0";
    img.addEventListener("load", () => {
      // remove placeholder, fade in
      pzTarget.querySelectorAll("img.placeholder").forEach(p => p.remove());
      img.style.opacity = "1";
    }, { once: true });
    pzTarget.appendChild(img);
  };

  const openModal = (id) => {
    const t = tiles.get(id);
    if (!t) return;
    activeFrameId = id;
    const f = t.frame;
    pzTarget.classList.toggle("compare", f.mode === "compare");
    pzTarget.innerHTML = "";

    if (f.mode === "compare" && f.compare_assets) {
      for (const aid of f.compare_assets) {
        const ph = document.createElement("img");
        ph.src = `/thumb/${aid}`;
        ph.classList.add("placeholder");
        ph.draggable = false;
        pzTarget.appendChild(ph);
        const full = document.createElement("img");
        full.src = `/image/${aid}`;
        full.decoding = "async";
        full.draggable = false;
        full.style.opacity = "0";
        full.addEventListener("load", () => {
          full.style.opacity = "1";
          ph.remove();
        }, { once: true });
        pzTarget.appendChild(full);
      }
    } else if (f.main_asset) {
      swapMainImage(f.main_asset, f.main_asset);
    }

    // Sidebar
    setBadge(mKind, f.kind);
    const s = f.stats || {};
    if (f.mode === "compare") {
      const ss = s.length ? s : [{}, {}];
      mStats.innerHTML = `
        <dt>shape A</dt><dd>${shapeStr((f.shape||[])[0])}</dd>
        <dt>shape B</dt><dd>${shapeStr((f.shape||[])[1])}</dd>
        <dt>min/max A</dt><dd>${fmtNum(ss[0].min)} / ${fmtNum(ss[0].max)}</dd>
        <dt>min/max B</dt><dd>${fmtNum(ss[1].min)} / ${fmtNum(ss[1].max)}</dd>
      `;
    } else {
      mStats.innerHTML = `
        <dt>shape</dt><dd>${shapeStr(f.shape)}</dd>
        <dt>dtype</dt><dd>${f.dtype || "?"}</dd>
        <dt>min</dt><dd>${fmtNum(s.min)}</dd>
        <dt>mean</dt><dd>${fmtNum(s.mean)}</dd>
        <dt>max</dt><dd>${fmtNum(s.max)}</dd>
        ${s.nan_pct !== undefined ? `<dt>nan</dt><dd>${s.nan_pct.toFixed(2)}%</dd>` : ""}
        <dt>scale</dt><dd>${f.scale || "linear"}</dd>
      `;
    }
    mLabel.textContent = f.label || "";
    mNote.textContent = f.note || "";

    // Band strip (multi-mode)
    if (f.kind === "multi" && Array.isArray(f.bands) && f.bands.length) {
      bandStrip.classList.remove("hidden");
      bandStrip.innerHTML = "";
      const composite = { asset: f.main_asset, label: "composite", _composite: true };
      const items = [composite, ...f.bands];
      let activeIdx = 0;
      const swap = (idx) => {
        activeIdx = idx;
        bandStrip.querySelectorAll(".band-item").forEach((el, i) =>
          el.classList.toggle("active", i === idx));
        const target = items[idx];
        swapMainImage(target.asset, target.asset);
      };
      items.forEach((b, i) => {
        const el = document.createElement("div");
        el.className = "band-item" + (i === 0 ? " active" : "");
        const im = document.createElement("img");
        im.src = `/thumb/${b.asset}`;
        im.loading = "lazy";
        im.decoding = "async";
        im.draggable = false;
        const lb = document.createElement("div");
        lb.className = "lbl";
        lb.textContent = b.label;
        el.appendChild(im);
        el.appendChild(lb);
        el.addEventListener("click", () => swap(i));
        bandStrip.appendChild(el);
      });
    } else {
      bandStrip.classList.add("hidden");
      bandStrip.innerHTML = "";
    }

    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");

    if (pz) { try { pz.destroy(); } catch(_){} pz = null; }
    if (typeof Panzoom === "function") {
      pz = Panzoom(pzTarget, { maxScale: 40, minScale: 0.1, canvas: true });
      pzHost.addEventListener("wheel", pz.zoomWithWheel, { passive: false });
    }
  };

  const closeModal = () => {
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
    activeFrameId = null;
    if (pz) { try { pz.destroy(); } catch(_){} pz = null; }
    pzTarget.innerHTML = "";
    bandStrip.innerHTML = "";
    bandStrip.classList.add("hidden");
  };

  modalClose.addEventListener("click", closeModal);
  modal.addEventListener("click", e => {
    if (e.target === modal || e.target.classList.contains("modal-stage") ||
        e.target.id === "pz_host") closeModal();
  });
  document.addEventListener("keydown", e => {
    if (modal.classList.contains("hidden")) return;
    if (e.key === "Escape") closeModal();
    else if (e.key === "0" && pz) pz.reset();
  });
  pzHost.addEventListener("dblclick", () => { if (pz) pz.reset(); });

  mLabel.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); mLabel.blur(); }
  });
  mLabel.addEventListener("blur", async () => {
    if (!activeFrameId) return;
    const t = tiles.get(activeFrameId);
    if (!t) return;
    const nl = mLabel.textContent.trim();
    if (nl !== (t.frame.label || "")) await patchLabel(activeFrameId, nl);
  });
  mDelete.addEventListener("click", async () => {
    if (!activeFrameId) return;
    if (confirm("Delete this frame?")) await deleteFrame(activeFrameId);
  });

  let ws = null;
  const connect = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.addEventListener("open", () => {
      statusEl.textContent = "live"; statusEl.className = "status connected";
    });
    ws.addEventListener("close", () => {
      statusEl.textContent = "offline"; statusEl.className = "status disconnected";
      setTimeout(connect, 1500);
    });
    ws.addEventListener("message", e => {
      const msg = JSON.parse(e.data);
      if (msg.kind === "frame_added") addTile(msg.payload);
      else if (msg.kind === "frame_updated") updateTile(msg.payload.id, msg.payload);
      else if (msg.kind === "frame_deleted") removeTile(msg.payload.id);
    });
  };

  fetch("/frames").then(r => r.json()).then(data => {
    for (const f of (data.frames || [])) addTile(f);
    connect();
  }).catch(() => connect());
})();

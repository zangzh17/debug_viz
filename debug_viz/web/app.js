(() => {
  const gallery = document.getElementById("gallery");
  const statusEl = document.getElementById("status");
  const countEl = document.getElementById("count");
  const modal = document.getElementById("modal");
  const pzTarget = document.getElementById("pz_target");
  const pzHost = document.getElementById("pz_host");
  const mLabel = document.getElementById("m_label");
  const mStats = document.getElementById("m_stats");
  const mNote = document.getElementById("m_note");
  const mDelete = document.getElementById("m_delete");
  const modalClose = document.getElementById("modal_close");
  const tileTmpl = document.getElementById("tile_tmpl");

  const tiles = new Map();    // id -> {frame, el}
  let activeFrameId = null;
  let pz = null;

  const fmtNum = (n) => {
    if (n === null || n === undefined) return "?";
    if (typeof n !== "number") return String(n);
    if (!isFinite(n)) return String(n);
    const a = Math.abs(n);
    if (a !== 0 && (a < 0.01 || a >= 1e5)) return n.toExponential(3);
    return n.toFixed(4).replace(/\.?0+$/, "");
  };

  const setCount = () => { countEl.textContent = `${tiles.size} frame${tiles.size === 1 ? "" : "s"}`; };

  const statsLine = (stats, shape, dtype) => {
    const sh = Array.isArray(shape) ? `[${shape.join("×")}]` : "?";
    if (!stats || stats.min === undefined) return `${sh} ${dtype}`;
    const nan = stats.nan_pct ? ` nan=${stats.nan_pct.toFixed(1)}%` : "";
    return `${sh} ${dtype}  min=${fmtNum(stats.min)} max=${fmtNum(stats.max)}${nan}`;
  };

  const renderThumb = (wrap, frame) => {
    wrap.innerHTML = "";
    if (frame.mode === "compare") {
      const t = frame.thumb_b64 || [];
      for (const b64 of t) {
        const img = document.createElement("img");
        img.src = `data:image/png;base64,${b64}`;
        img.draggable = false;
        wrap.appendChild(img);
      }
    } else {
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${frame.thumb_b64}`;
      img.draggable = false;
      wrap.appendChild(img);
    }
  };

  const createTile = (frame) => {
    const node = tileTmpl.content.firstElementChild.cloneNode(true);
    if (frame.mode === "compare") node.classList.add("compare");
    const wrap = node.querySelector(".thumb-wrap");
    renderThumb(wrap, frame);

    const labelEl = node.querySelector(".label");
    labelEl.textContent = frame.label || "";

    const statsEl = node.querySelector(".stats");
    if (frame.mode === "compare") {
      const s = frame.stats || [{}, {}];
      const sh = frame.shape || [null, null];
      statsEl.textContent = `${statsLine(s[0], sh[0], (frame.dtype||[])[0]||"?")} | ${statsLine(s[1], sh[1], (frame.dtype||[])[1]||"?")}`;
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

    labelEl.addEventListener("click", (e) => e.stopPropagation());
    labelEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); labelEl.blur(); }
    });
    labelEl.addEventListener("blur", async () => {
      const newLabel = labelEl.textContent.trim();
      if (newLabel !== (frame.label || "")) {
        await patchLabel(frame.id, newLabel);
      }
    });

    const delBtn = node.querySelector(".del");
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (confirm("Delete this frame?")) {
        await deleteFrame(frame.id);
      }
    });

    node.addEventListener("click", () => openModal(frame.id));
    return node;
  };

  const addTile = (frame) => {
    if (tiles.has(frame.id)) {
      updateTile(frame.id, frame, true);
      return;
    }
    const el = createTile(frame);
    tiles.set(frame.id, { frame, el });
    gallery.insertBefore(el, gallery.firstChild);
    setCount();
  };

  const updateTile = (id, partial, fullReplace=false) => {
    const t = tiles.get(id);
    if (!t) return;
    if (fullReplace) {
      Object.assign(t.frame, partial);
    } else {
      if (partial.fields) Object.assign(t.frame, partial.fields);
    }
    const labelEl = t.el.querySelector(".label");
    if (document.activeElement !== labelEl) {
      labelEl.textContent = t.frame.label || "";
    }
    if (activeFrameId === id) {
      mLabel.textContent = t.frame.label || "";
    }
  };

  const removeTile = (id) => {
    const t = tiles.get(id);
    if (!t) return;
    t.el.classList.add("removing");
    setTimeout(() => { t.el.remove(); tiles.delete(id); setCount(); }, 220);
    if (activeFrameId === id) closeModal();
  };

  const patchLabel = async (id, label) => {
    try {
      await fetch(`/frames/${id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ label }),
      });
    } catch (e) { console.error("patch failed", e); }
  };

  const deleteFrame = async (id) => {
    try {
      await fetch(`/frames/${id}`, { method: "DELETE" });
    } catch (e) { console.error("delete failed", e); }
  };

  // Modal & Panzoom
  const openModal = (id) => {
    const t = tiles.get(id);
    if (!t) return;
    activeFrameId = id;
    pzTarget.innerHTML = "";
    pzTarget.classList.toggle("compare", t.frame.mode === "compare");

    if (t.frame.mode === "compare") {
      for (const sid of (t.frame.sub_ids || [])) {
        const img = document.createElement("img");
        img.src = `/png/${sid}`;
        img.draggable = false;
        pzTarget.appendChild(img);
      }
      const s = t.frame.stats || [{}, {}];
      mStats.innerHTML = `
        <dt>shape A</dt><dd>${(t.frame.shape || [])[0] || "?"}</dd>
        <dt>shape B</dt><dd>${(t.frame.shape || [])[1] || "?"}</dd>
        <dt>min/max A</dt><dd>${fmtNum(s[0].min)} / ${fmtNum(s[0].max)}</dd>
        <dt>min/max B</dt><dd>${fmtNum(s[1].min)} / ${fmtNum(s[1].max)}</dd>
      `;
    } else {
      const img = document.createElement("img");
      img.src = `/png/${id}`;
      img.draggable = false;
      pzTarget.appendChild(img);
      const s = t.frame.stats || {};
      const sh = t.frame.shape ? `[${t.frame.shape.join("×")}]` : "?";
      mStats.innerHTML = `
        <dt>shape</dt><dd>${sh}</dd>
        <dt>dtype</dt><dd>${t.frame.dtype || "?"}</dd>
        <dt>min</dt><dd>${fmtNum(s.min)}</dd>
        <dt>mean</dt><dd>${fmtNum(s.mean)}</dd>
        <dt>max</dt><dd>${fmtNum(s.max)}</dd>
        ${s.nan_pct !== undefined ? `<dt>nan</dt><dd>${s.nan_pct.toFixed(2)}%</dd>` : ""}
        <dt>scale</dt><dd>${t.frame.scale || "linear"}</dd>
      `;
    }
    mLabel.textContent = t.frame.label || "";
    mNote.textContent = t.frame.note || "";
    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");

    if (pz) { try { pz.destroy(); } catch(_){} pz = null; }
    if (typeof Panzoom === "function") {
      pz = Panzoom(pzTarget, {
        maxScale: 40, minScale: 0.1,
        canvas: true,
      });
      pzHost.addEventListener("wheel", pz.zoomWithWheel, { passive: false });
    }
  };

  const closeModal = () => {
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
    activeFrameId = null;
    if (pz) { try { pz.destroy(); } catch(_){} pz = null; }
    pzTarget.innerHTML = "";
  };

  modalClose.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal || e.target.classList.contains("modal-stage") || e.target.id === "pz_host") {
      closeModal();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (modal.classList.contains("hidden")) return;
    if (e.key === "Escape") closeModal();
    else if (e.key === "0" && pz) pz.reset();
  });
  pzHost.addEventListener("dblclick", (e) => { if (pz) pz.reset(); });

  mLabel.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); mLabel.blur(); }
  });
  mLabel.addEventListener("blur", async () => {
    if (!activeFrameId) return;
    const t = tiles.get(activeFrameId);
    if (!t) return;
    const newLabel = mLabel.textContent.trim();
    if (newLabel !== (t.frame.label || "")) {
      await patchLabel(activeFrameId, newLabel);
    }
  });

  mDelete.addEventListener("click", async () => {
    if (!activeFrameId) return;
    if (confirm("Delete this frame?")) {
      await deleteFrame(activeFrameId);
    }
  });

  // WebSocket
  let ws = null;
  const connect = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.addEventListener("open", () => {
      statusEl.textContent = "live";
      statusEl.className = "status connected";
    });
    ws.addEventListener("close", () => {
      statusEl.textContent = "offline";
      statusEl.className = "status disconnected";
      setTimeout(connect, 1500);
    });
    ws.addEventListener("message", (e) => {
      const msg = JSON.parse(e.data);
      if (msg.kind === "frame_added") addTile(msg.payload);
      else if (msg.kind === "frame_updated") updateTile(msg.payload.id, msg.payload);
      else if (msg.kind === "frame_deleted") removeTile(msg.payload.id);
    });
  };

  // Initial load
  fetch("/frames").then(r => r.json()).then(data => {
    const frames = data.frames || [];
    // server returns oldest-first; we insert at front, so iterate forward to end up with newest-first
    for (const f of frames) addTile(f);
    connect();
  }).catch(err => {
    console.error("initial load failed", err);
    connect();
  });
})();

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
  name: "4dhumans.PoseEditor",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "PoseEditor") return;

    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (data) {
      origExecuted?.apply(this, arguments);
      if (!data) return;

      const frames = data.frames || [];
      const fps = (data.fps && data.fps[0]) || 24;
      const nPersons = (data.n_persons && data.n_persons[0]) || 0;
      const personVisibility = (data.person_visibility && data.person_visibility[0]) || [];
      const nodeId = (data.node_id && data.node_id[0]) || "";

      if (frames.length === 0) return;

      initEditorUI(this, frames, fps, nPersons, personVisibility, nodeId);
    };
  },
});

function initEditorUI(node, frames, fps, nPersons, personVisibility, nodeId) {
  // Clean up previous editor
  if (node._poseEditorWidget) {
    const idx = node.widgets?.indexOf(node._poseEditorWidget);
    if (idx >= 0) node.widgets.splice(idx, 1);
    node._poseEditorWidget = null;
  }
  if (node._poseEditorRAF) {
    cancelAnimationFrame(node._poseEditorRAF);
    node._poseEditorRAF = null;
  }

  const totalFrames = frames.length;

  // --- Container ---
  const container = document.createElement("div");
  container.style.cssText =
    "display:flex;flex-direction:column;gap:6px;padding:6px;" +
    "width:100%;height:100%;box-sizing:border-box;overflow:hidden;";

  // --- Playback controls ---
  const controlsRow = document.createElement("div");
  controlsRow.style.cssText =
    "display:flex;align-items:center;gap:6px;width:100%;box-sizing:border-box;overflow:hidden;flex-shrink:0;";

  const playBtn = document.createElement("button");
  playBtn.textContent = "▶";
  playBtn.style.cssText =
    "width:32px;height:26px;font-size:13px;cursor:pointer;border:1px solid #555;" +
    "background:#333;color:#fff;border-radius:4px;flex-shrink:0;";

  const scrubber = document.createElement("input");
  scrubber.type = "range";
  scrubber.min = 0;
  scrubber.max = totalFrames - 1;
  scrubber.value = 0;
  scrubber.style.cssText = "flex:1;min-width:0;cursor:pointer;";

  const frameLabel = document.createElement("span");
  frameLabel.style.cssText =
    "color:#ccc;font-size:11px;white-space:nowrap;flex-shrink:0;";
  frameLabel.textContent = `1/${totalFrames}`;

  controlsRow.appendChild(playBtn);
  controlsRow.appendChild(scrubber);
  controlsRow.appendChild(frameLabel);

  // --- Image display ---
  const imgEl = document.createElement("img");
  imgEl.style.cssText =
    "width:100%;flex:1;min-height:0;object-fit:contain;border-radius:4px;" +
    "background:#000;display:block;";

  // --- Person toggles ---
  const personsRow = document.createElement("div");
  personsRow.style.cssText =
    "display:flex;flex-wrap:wrap;align-items:center;gap:4px;width:100%;box-sizing:border-box;flex-shrink:0;";

  const personsLabel = document.createElement("span");
  personsLabel.textContent = "Persons:";
  personsLabel.style.cssText = "color:#aaa;font-size:11px;flex-shrink:0;";
  personsRow.appendChild(personsLabel);

  const visibility = [...personVisibility];

  for (let p = 0; p < nPersons; p++) {
    const btn = document.createElement("button");
    btn.style.cssText =
      "padding:1px 8px;font-size:11px;cursor:pointer;border:1px solid #555;" +
      "border-radius:3px;min-width:28px;";
    updateToggleBtn(btn, p, visibility[p]);

    btn.addEventListener("click", async () => {
      visibility[p] = !visibility[p];
      updateToggleBtn(btn, p, visibility[p]);

      await api.fetchApi("/pose_editor/update_visibility", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          node_id: nodeId,
          person_id: p,
          visible: visibility[p],
        }),
      });
    });

    personsRow.appendChild(btn);
  }

  // --- Download button ---
  const downloadBtn = document.createElement("button");
  downloadBtn.textContent = "📥 Download NPZ";
  downloadBtn.style.cssText =
    "padding:4px 14px;font-size:12px;cursor:pointer;border:1px solid #555;" +
    "background:#2a5a2a;color:#fff;border-radius:4px;align-self:center;flex-shrink:0;";
  downloadBtn.addEventListener("click", () => {
    window.open(
      `/pose_editor/download_npz?node_id=${encodeURIComponent(nodeId)}`,
      "_blank"
    );
  });

  // Assemble
  container.appendChild(controlsRow);
  container.appendChild(imgEl);
  container.appendChild(personsRow);
  container.appendChild(downloadBtn);

  // --- Preload images ---
  const images = [];
  let loadedCount = 0;

  for (let i = 0; i < totalFrames; i++) {
    const img = new Image();
    const f = frames[i];
    img.src = `/view?filename=${encodeURIComponent(f.filename)}&type=${f.type}&subfolder=${encodeURIComponent(f.subfolder || "")}`;
    img.onload = () => {
      loadedCount++;
      // Show first frame as soon as it loads
      if (i === 0) updateDisplay();
    };
    images.push(img);
  }

  // --- Playback state ---
  let currentFrame = 0;
  let playing = false;
  let lastFrameTime = 0;
  const frameDuration = 1000 / fps;

  function updateDisplay() {
    const img = images[currentFrame];
    if (img && img.complete && img.naturalWidth > 0) {
      // Force browser to treat as new image by resetting src
      imgEl.src = "";
      imgEl.src = img.src;
    }
    scrubber.value = currentFrame;
    frameLabel.textContent = `${currentFrame + 1}/${totalFrames}`;
  }

  function animationLoop(timestamp) {
    if (!playing) return;
    if (timestamp - lastFrameTime >= frameDuration) {
      lastFrameTime = timestamp;
      currentFrame = (currentFrame + 1) % totalFrames;
      updateDisplay();
    }
    node._poseEditorRAF = requestAnimationFrame(animationLoop);
  }

  function play() {
    playing = true;
    playBtn.textContent = "⏸";
    lastFrameTime = performance.now();
    node._poseEditorRAF = requestAnimationFrame(animationLoop);
  }

  function pause() {
    playing = false;
    playBtn.textContent = "▶";
    if (node._poseEditorRAF) {
      cancelAnimationFrame(node._poseEditorRAF);
      node._poseEditorRAF = null;
    }
  }

  playBtn.addEventListener("click", () => {
    if (playing) pause();
    else play();
  });

  scrubber.addEventListener("input", () => {
    pause();
    currentFrame = parseInt(scrubber.value);
    updateDisplay();
  });

  // Show first frame immediately if already loaded
  if (images[0] && images[0].complete) {
    updateDisplay();
  }

  // Add as DOM widget
  const widget = node.addDOMWidget("pose_editor", "customwidget", container, {
    getMinHeight: () => 300,
  });
  node._poseEditorWidget = widget;

  node.setSize([
    Math.max(node.size[0], 380),
    Math.max(node.size[1], 480),
  ]);
}

function updateToggleBtn(btn, personId, visible) {
  if (visible) {
    btn.textContent = `✓ P${personId}`;
    btn.style.background = "#2a4a2a";
    btn.style.color = "#8f8";
  } else {
    btn.textContent = `✗ P${personId}`;
    btn.style.background = "#4a2a2a";
    btn.style.color = "#f88";
  }
}

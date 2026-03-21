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
  // Remove previous editor widget if any
  if (node._poseEditorWidget) {
    const idx = node.widgets?.indexOf(node._poseEditorWidget);
    if (idx >= 0) node.widgets.splice(idx, 1);
    node._poseEditorWidget = null;
  }

  // Preload frame images
  const images = [];
  let loadedCount = 0;
  const totalFrames = frames.length;

  const container = document.createElement("div");
  container.style.cssText =
    "display:flex;flex-direction:column;align-items:center;gap:8px;padding:8px;width:100%;";

  // --- Playback controls ---
  const controlsRow = document.createElement("div");
  controlsRow.style.cssText =
    "display:flex;align-items:center;gap:8px;width:100%;";

  const playBtn = document.createElement("button");
  playBtn.textContent = "▶";
  playBtn.style.cssText =
    "width:36px;height:28px;font-size:14px;cursor:pointer;border:1px solid #555;background:#333;color:#fff;border-radius:4px;";

  const scrubber = document.createElement("input");
  scrubber.type = "range";
  scrubber.min = 0;
  scrubber.max = totalFrames - 1;
  scrubber.value = 0;
  scrubber.style.cssText = "flex:1;cursor:pointer;";

  const frameLabel = document.createElement("span");
  frameLabel.style.cssText = "color:#ccc;font-size:12px;min-width:60px;text-align:right;";
  frameLabel.textContent = `1/${totalFrames}`;

  controlsRow.appendChild(playBtn);
  controlsRow.appendChild(scrubber);
  controlsRow.appendChild(frameLabel);

  // --- Image display ---
  const imgEl = document.createElement("img");
  imgEl.style.cssText = "width:100%;image-rendering:auto;border-radius:4px;background:#000;";

  // --- Person toggles ---
  const personsRow = document.createElement("div");
  personsRow.style.cssText =
    "display:flex;flex-wrap:wrap;gap:6px;width:100%;";

  const personsLabel = document.createElement("span");
  personsLabel.textContent = "Persons:";
  personsLabel.style.cssText = "color:#aaa;font-size:12px;margin-right:4px;";
  personsRow.appendChild(personsLabel);

  const visibility = [...personVisibility];
  const toggleBtns = [];

  for (let p = 0; p < nPersons; p++) {
    const btn = document.createElement("button");
    btn.style.cssText =
      "padding:2px 10px;font-size:12px;cursor:pointer;border:1px solid #555;border-radius:4px;";
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

    toggleBtns.push(btn);
    personsRow.appendChild(btn);
  }

  // --- Download button ---
  const downloadBtn = document.createElement("button");
  downloadBtn.textContent = "📥 Download NPZ";
  downloadBtn.style.cssText =
    "padding:6px 16px;font-size:13px;cursor:pointer;border:1px solid #555;background:#2a5a2a;color:#fff;border-radius:4px;";
  downloadBtn.addEventListener("click", () => {
    window.open(`/pose_editor/download_npz?node_id=${encodeURIComponent(nodeId)}`, "_blank");
  });

  // Assemble
  container.appendChild(controlsRow);
  container.appendChild(imgEl);
  container.appendChild(personsRow);
  container.appendChild(downloadBtn);

  // --- Playback state ---
  let currentFrame = 0;
  let playing = false;
  let timerId = null;

  function updateDisplay() {
    if (images[currentFrame] && images[currentFrame].complete) {
      imgEl.src = images[currentFrame].src;
    }
    scrubber.value = currentFrame;
    frameLabel.textContent = `${currentFrame + 1}/${totalFrames}`;
  }

  function play() {
    playing = true;
    playBtn.textContent = "⏸";
    function tick() {
      if (!playing) return;
      currentFrame = (currentFrame + 1) % totalFrames;
      updateDisplay();
      timerId = setTimeout(tick, 1000 / fps);
    }
    tick();
  }

  function pause() {
    playing = false;
    playBtn.textContent = "▶";
    if (timerId) {
      clearTimeout(timerId);
      timerId = null;
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

  // Load images
  for (let i = 0; i < totalFrames; i++) {
    const img = new Image();
    const f = frames[i];
    img.src = api.apiURL(
      `/view?filename=${encodeURIComponent(f.filename)}&type=${f.type}&subfolder=${encodeURIComponent(f.subfolder || "")}`
    );
    img.onload = () => {
      loadedCount++;
      if (loadedCount === 1) updateDisplay();
    };
    images.push(img);
  }

  // Add as DOM widget
  const widget = node.addDOMWidget("pose_editor", "customwidget", container, {
    getMinHeight: () => 400,
    getMaxHeight: () => 800,
  });
  node._poseEditorWidget = widget;

  // Resize node to fit
  node.setSize([
    Math.max(node.size[0], 400),
    Math.max(node.size[1], 520),
  ]);
}

function updateToggleBtn(btn, personId, visible) {
  if (visible) {
    btn.textContent = `✓ Person ${personId}`;
    btn.style.background = "#2a4a2a";
    btn.style.color = "#8f8";
  } else {
    btn.textContent = `✗ Person ${personId}`;
    btn.style.background = "#4a2a2a";
    btn.style.color = "#f88";
  }
}

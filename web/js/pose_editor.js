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

      const videoInfo = data.video && data.video[0];
      const nPersons = (data.n_persons && data.n_persons[0]) || 0;
      const personVisibility =
        (data.person_visibility && data.person_visibility[0]) || [];
      const nodeId = (data.node_id && data.node_id[0]) || "";
      const velThresh = (data.velocity_threshold && data.velocity_threshold[0]) || 3.0;
      const smoothSigma = (data.smooth_sigma && data.smooth_sigma[0]) || 0.0;

      if (!videoInfo || nPersons === 0) return;

      initEditorUI(this, videoInfo, nPersons, personVisibility, nodeId,
                    velThresh, smoothSigma);
    };
  },
});

function buildVideoUrl(videoInfo) {
  return `/view?filename=${encodeURIComponent(videoInfo.filename)}&type=${videoInfo.type}&subfolder=${encodeURIComponent(videoInfo.subfolder || "")}`;
}

function initEditorUI(node, videoInfo, nPersons, personVisibility, nodeId,
                      velThresh, smoothSigma) {
  // Remove previous widget and clean up old video element
  if (node._poseEditorWidget) {
    // Stop old video to free resources
    const oldContainer = node._poseEditorWidget.element;
    if (oldContainer) {
      const oldVideo = oldContainer.querySelector("video");
      if (oldVideo) {
        oldVideo.pause();
        oldVideo.removeAttribute("src");
        oldVideo.load();
      }
      oldContainer.remove();
    }
    const idx = node.widgets?.indexOf(node._poseEditorWidget);
    if (idx >= 0) node.widgets.splice(idx, 1);
    node._poseEditorWidget = null;
  }

  const container = document.createElement("div");
  container.style.cssText =
    "display:flex;flex-direction:column;gap:6px;padding:6px;" +
    "width:100%;height:100%;box-sizing:border-box;overflow:hidden;" +
    "position:relative;";

  // --- Loading overlay ---
  const overlay = document.createElement("div");
  overlay.style.cssText =
    "position:absolute;top:0;left:0;right:0;bottom:0;z-index:10;" +
    "background:rgba(0,0,0,0.6);display:none;align-items:center;" +
    "justify-content:center;border-radius:4px;";
  const spinner = document.createElement("span");
  spinner.textContent = "⏳ Rendering...";
  spinner.style.cssText = "color:#fff;font-size:14px;";
  overlay.appendChild(spinner);

  // --- Video player ---
  const videoEl = document.createElement("video");
  videoEl.controls = true;
  videoEl.loop = true;
  videoEl.autoplay = true;
  videoEl.muted = true;
  videoEl.style.cssText =
    "width:100%;flex:1;min-height:0;object-fit:contain;" +
    "border-radius:4px;background:#000;display:block;";
  videoEl.src = buildVideoUrl(videoInfo);

  // --- Shared rendering state ---
  const visibility = [...personVisibility];
  let isRendering = false;

  async function refreshVideo(url) {
    const currentTime = videoEl.currentTime;
    videoEl.src = url;
    videoEl.addEventListener(
      "loadeddata",
      () => {
        videoEl.currentTime = currentTime;
        videoEl.play().catch(() => {});
      },
      { once: true }
    );
  }

  // --- Person toggles ---
  const personsRow = document.createElement("div");
  personsRow.style.cssText =
    "display:flex;flex-wrap:wrap;align-items:center;gap:4px;" +
    "width:100%;box-sizing:border-box;flex-shrink:0;";

  const personsLabel = document.createElement("span");
  personsLabel.textContent = "Persons:";
  personsLabel.style.cssText = "color:#aaa;font-size:11px;flex-shrink:0;";
  personsRow.appendChild(personsLabel);

  // Track pending visibility (may differ from committed visibility)
  const pendingVisibility = [...visibility];

  for (let p = 0; p < nPersons; p++) {
    const btn = document.createElement("button");
    btn.style.cssText =
      "padding:2px 0;font-size:11px;cursor:pointer;border:1px solid #555;" +
      "border-radius:3px;width:42px;text-align:center;";
    updateToggleBtn(btn, p, pendingVisibility[p]);

    btn.addEventListener("click", () => {
      if (isRendering) return;
      pendingVisibility[p] = !pendingVisibility[p];
      updateToggleBtn(btn, p, pendingVisibility[p]);
    });

    personsRow.appendChild(btn);
  }

  // Apply visibility button
  const applyVisBtn = document.createElement("button");
  applyVisBtn.textContent = "Apply";
  applyVisBtn.style.cssText =
    "padding:2px 10px;font-size:11px;cursor:pointer;border:1px solid #555;" +
    "background:#2a3a5a;color:#adf;border-radius:3px;flex-shrink:0;";
  applyVisBtn.addEventListener("click", async () => {
    if (isRendering) return;

    // Check if anything changed
    const changed = pendingVisibility.some((v, i) => v !== visibility[i]);
    if (!changed) return;

    isRendering = true;
    overlay.style.display = "flex";

    try {
      // Send all visibility updates
      for (let p = 0; p < nPersons; p++) {
        if (pendingVisibility[p] !== visibility[p]) {
          await api.fetchApi("/pose_editor/toggle_visibility", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              node_id: nodeId,
              person_id: p,
              visible: pendingVisibility[p],
            }),
          });
          visibility[p] = pendingVisibility[p];
        }
      }

      // Trigger re-render after all toggles applied
      const resp = await api.fetchApi("/pose_editor/rerender", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: nodeId }),
      });
      const result = await resp.json();

      if (result.ok && result.video) {
        await refreshVideo(buildVideoUrl(result.video));
      }
    } catch (e) {
      console.error("Pose editor re-render failed:", e);
    } finally {
      overlay.style.display = "none";
      isRendering = false;
    }
  });
  personsRow.appendChild(applyVisBtn);

  // --- Filter controls ---
  const filterRow = document.createElement("div");
  filterRow.style.cssText =
    "display:flex;flex-wrap:wrap;align-items:center;gap:8px;" +
    "width:100%;box-sizing:border-box;flex-shrink:0;";

  const filterLabel = document.createElement("span");
  filterLabel.textContent = "Filter:";
  filterLabel.style.cssText = "color:#aaa;font-size:11px;flex-shrink:0;";
  filterRow.appendChild(filterLabel);

  // Velocity threshold slider
  const velGroup = makeSlider("Vel", velThresh, 0, 20, 0.1);
  filterRow.appendChild(velGroup.container);

  // Smooth sigma slider
  const sigmaGroup = makeSlider("σ", smoothSigma, 0, 5, 0.1);
  filterRow.appendChild(sigmaGroup.container);

  // Apply button
  const applyBtn = document.createElement("button");
  applyBtn.textContent = "Apply";
  applyBtn.style.cssText =
    "padding:2px 10px;font-size:11px;cursor:pointer;border:1px solid #555;" +
    "background:#2a3a5a;color:#adf;border-radius:3px;flex-shrink:0;";
  applyBtn.addEventListener("click", async () => {
    if (isRendering) return;

    isRendering = true;
    overlay.style.display = "flex";

    try {
      const resp = await api.fetchApi("/pose_editor/update_filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          node_id: nodeId,
          velocity_threshold: parseFloat(velGroup.input.value),
          smooth_sigma: parseFloat(sigmaGroup.input.value),
        }),
      });
      const result = await resp.json();

      if (result.ok && result.video) {
        await refreshVideo(buildVideoUrl(result.video));
      }
    } catch (e) {
      console.error("Pose editor filter update failed:", e);
    } finally {
      overlay.style.display = "none";
      isRendering = false;
    }
  });
  filterRow.appendChild(applyBtn);

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
  container.appendChild(overlay);
  container.appendChild(videoEl);
  container.appendChild(personsRow);
  container.appendChild(filterRow);
  container.appendChild(downloadBtn);

  // Add as DOM widget
  const widget = node.addDOMWidget(
    "pose_editor",
    "customwidget",
    container,
    { getMinHeight: () => 340 }
  );
  node._poseEditorWidget = widget;

  node.setSize([
    Math.max(node.size[0], 380),
    Math.max(node.size[1], 500),
  ]);
}

function updateToggleBtn(btn, personId, visible) {
  if (visible) {
    btn.textContent = `✓ P${personId}`;
    btn.style.background = "#2a4a2a";
    btn.style.color = "#8f8";
    btn.style.borderColor = "#555";
  } else {
    btn.textContent = `✗ P${personId}`;
    btn.style.background = "#4a2a2a";
    btn.style.color = "#f88";
    btn.style.borderColor = "#555";
  }
}

function makeSlider(label, value, min, max, step) {
  const container = document.createElement("div");
  container.style.cssText =
    "display:flex;align-items:center;gap:3px;flex-shrink:0;";

  const lbl = document.createElement("span");
  lbl.textContent = label;
  lbl.style.cssText = "color:#aaa;font-size:10px;";

  const input = document.createElement("input");
  input.type = "range";
  input.min = min;
  input.max = max;
  input.step = step;
  input.value = value;
  input.style.cssText = "width:70px;height:14px;";

  const valSpan = document.createElement("span");
  valSpan.textContent = parseFloat(value).toFixed(1);
  valSpan.style.cssText = "color:#ddd;font-size:10px;width:26px;text-align:right;";

  input.addEventListener("input", () => {
    valSpan.textContent = parseFloat(input.value).toFixed(1);
  });

  container.appendChild(lbl);
  container.appendChild(input);
  container.appendChild(valSpan);

  return { container, input, valSpan };
}

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

      if (!videoInfo || nPersons === 0) return;

      initEditorUI(this, videoInfo, nPersons, personVisibility, nodeId);
    };
  },
});

function buildVideoUrl(videoInfo) {
  return `/view?filename=${encodeURIComponent(videoInfo.filename)}&type=${videoInfo.type}&subfolder=${encodeURIComponent(videoInfo.subfolder || "")}`;
}

function initEditorUI(node, videoInfo, nPersons, personVisibility, nodeId) {
  // Remove previous widget
  if (node._poseEditorWidget) {
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

  // --- Person toggles ---
  const personsRow = document.createElement("div");
  personsRow.style.cssText =
    "display:flex;flex-wrap:wrap;align-items:center;gap:4px;" +
    "width:100%;box-sizing:border-box;flex-shrink:0;";

  const personsLabel = document.createElement("span");
  personsLabel.textContent = "Persons:";
  personsLabel.style.cssText = "color:#aaa;font-size:11px;flex-shrink:0;";
  personsRow.appendChild(personsLabel);

  const visibility = [...personVisibility];
  let isRendering = false;

  for (let p = 0; p < nPersons; p++) {
    const btn = document.createElement("button");
    btn.style.cssText =
      "padding:2px 0;font-size:11px;cursor:pointer;border:1px solid #555;" +
      "border-radius:3px;width:42px;text-align:center;";
    updateToggleBtn(btn, p, visibility[p]);

    btn.addEventListener("click", async () => {
      if (isRendering) return;

      visibility[p] = !visibility[p];
      updateToggleBtn(btn, p, visibility[p]);

      // Show overlay and save current playback time
      isRendering = true;
      overlay.style.display = "flex";
      const currentTime = videoEl.currentTime;

      try {
        const resp = await api.fetchApi("/pose_editor/toggle_visibility", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            node_id: nodeId,
            person_id: p,
            visible: visibility[p],
          }),
        });
        const result = await resp.json();

        if (result.ok && result.video) {
          // Update video source and restore playback position
          const newUrl = buildVideoUrl(result.video);
          videoEl.src = newUrl;
          videoEl.addEventListener(
            "loadeddata",
            () => {
              videoEl.currentTime = currentTime;
              videoEl.play().catch(() => {});
            },
            { once: true }
          );
        }
      } catch (e) {
        console.error("Pose editor re-render failed:", e);
      } finally {
        overlay.style.display = "none";
        isRendering = false;
      }
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
  container.appendChild(overlay);
  container.appendChild(videoEl);
  container.appendChild(personsRow);
  container.appendChild(downloadBtn);

  // Add as DOM widget
  const widget = node.addDOMWidget(
    "pose_editor",
    "customwidget",
    container,
    { getMinHeight: () => 300 }
  );
  node._poseEditorWidget = widget;

  node.setSize([
    Math.max(node.size[0], 380),
    Math.max(node.size[1], 450),
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

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

      const nPersons = (data.n_persons && data.n_persons[0]) || 0;
      const personVisibility =
        (data.person_visibility && data.person_visibility[0]) || [];
      const nodeId = (data.node_id && data.node_id[0]) || "";

      if (nPersons === 0) return;

      initEditorControls(this, nPersons, personVisibility, nodeId);
    };
  },
});

function initEditorControls(node, nPersons, personVisibility, nodeId) {
  // Remove previous controls widget if any
  if (node._poseEditorWidget) {
    const idx = node.widgets?.indexOf(node._poseEditorWidget);
    if (idx >= 0) node.widgets.splice(idx, 1);
    node._poseEditorWidget = null;
  }

  const container = document.createElement("div");
  container.style.cssText =
    "display:flex;flex-direction:column;gap:6px;padding:6px;" +
    "width:100%;box-sizing:border-box;";

  // --- Person toggles ---
  const personsRow = document.createElement("div");
  personsRow.style.cssText =
    "display:flex;flex-wrap:wrap;align-items:center;gap:4px;width:100%;box-sizing:border-box;";

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
    "background:#2a5a2a;color:#fff;border-radius:4px;align-self:center;";
  downloadBtn.addEventListener("click", () => {
    window.open(
      `/pose_editor/download_npz?node_id=${encodeURIComponent(nodeId)}`,
      "_blank"
    );
  });

  container.appendChild(personsRow);
  container.appendChild(downloadBtn);

  // Add as DOM widget (below the native animated preview)
  const widget = node.addDOMWidget(
    "pose_editor_controls",
    "customwidget",
    container,
    { getMinHeight: () => 60 }
  );
  node._poseEditorWidget = widget;
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

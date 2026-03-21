import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
  name: "4dhumans.LoadPoseUpload",

  async nodeCreated(node) {
    if (node.comfyClass !== "LoadPoseData") return;

    // Add an upload button widget
    const uploadBtn = node.addWidget("button", "upload_npz", "📁 Upload .npz", async () => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".npz";
      input.style.display = "none";
      document.body.appendChild(input);

      input.addEventListener("change", async () => {
        if (!input.files || input.files.length === 0) return;

        const file = input.files[0];
        const formData = new FormData();
        formData.append("image", file, file.name);  // ComfyUI upload endpoint uses "image" key
        formData.append("overwrite", "true");

        try {
          const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body: formData,
          });
          const result = await resp.json();

          if (result.name) {
            // Update the file selector widget to the uploaded file
            const fileWidget = node.widgets.find(
              (w) => w.name === "file"
            );
            if (fileWidget) {
              // Add the new file to the options if not already present
              if (fileWidget.options && fileWidget.options.values) {
                if (!fileWidget.options.values.includes(result.name)) {
                  fileWidget.options.values.push(result.name);
                }
              }
              fileWidget.value = result.name;
              node.setDirtyCanvas(true);
            }
          }
        } catch (e) {
          console.error("NPZ upload failed:", e);
          alert("Upload failed: " + e.message);
        } finally {
          document.body.removeChild(input);
        }
      });

      input.click();
    });
  },
});

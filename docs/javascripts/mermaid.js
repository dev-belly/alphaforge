// Mermaid 10.9.5 is vendored beside this file; see MERMAID-LICENSE.txt.
// Explicitly render the pre.mermaid nodes emitted by SuperFences.
document.addEventListener("DOMContentLoaded", async function () {
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });
    await window.mermaid.run({ querySelector: ".mermaid" });
  }
});

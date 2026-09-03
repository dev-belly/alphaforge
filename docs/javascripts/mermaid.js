// Initialise Mermaid diagrams after the Material theme has rendered the page.
document.addEventListener("DOMContentLoaded", function () {
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: true, securityLevel: "loose" });
  }
});

"""Render a NetworkX graph to a self-contained HTML string for QWebEngineView."""

from __future__ import annotations

import re

import networkx as nx
from pyvis.network import Network

# Match the GUI dark surfaces.
CANVAS_BG = "#0f172a"
NODE_COLOR = "#38bdf8"
NODE_FONT = "#e2e8f0"
TITLE_COLOR = "#e2e8f0"


def _weight_to_color(norm_weight: float) -> str:
    """Map a normalized weight in [0, 1] to a viridis-like hex color."""
    if norm_weight < 0.5:
        r = 0
        g = int(norm_weight * 2 * 255)
        b = 255
    else:
        r = int((norm_weight - 0.5) * 2 * 255)
        g = 255
        b = int(255 - (norm_weight - 0.5) * 2 * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _theme_pyvis_html(html: str) -> str:
    """Remove duplicate headings and force a full-bleed dark canvas."""
    html = re.sub(
        r"<center>\s*<h1>.*?</h1>\s*</center>\s*",
        "",
        html,
        count=1,
        flags=re.S,
    )
    # Percentage heights + pyvis' float:left collapse the canvas in QWebEngineView.
    dark_css = f"""
    <style>
      html, body {{
        background-color: {CANVAS_BG} !important;
        color: {TITLE_COLOR} !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        height: 100% !important;
        overflow: hidden !important;
      }}
      body {{
        display: flex !important;
        flex-direction: column !important;
      }}
      h1 {{
        color: {TITLE_COLOR} !important;
        font-family: "Segoe UI", Helvetica, sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0.5rem 0.75rem !important;
        flex: 0 0 auto;
      }}
      #mynetwork {{
        background-color: {CANVAS_BG} !important;
        border: none !important;
        float: none !important;
        position: relative !important;
        width: 100% !important;
        height: 100% !important;
        flex: 1 1 auto !important;
        min-height: 0 !important;
      }}
      .card, .card-body, .container, .container-fluid {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        width: 100% !important;
        height: 100% !important;
        max-width: none !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
      }}
    </style>
    """
    # After physics settles, fit the graph to the full canvas.
    fit_script = """
    <script>
      (function () {
        function fitNetwork() {
          try {
            if (typeof network !== "undefined" && network !== null) {
              network.fit({ animation: { duration: 400, easingFunction: "easeInOutQuad" } });
              return true;
            }
          } catch (e) {}
          return false;
        }
        var tries = 0;
        var timer = setInterval(function () {
          tries += 1;
          if (fitNetwork() || tries > 40) {
            clearInterval(timer);
          }
        }, 250);
        window.addEventListener("load", function () {
          setTimeout(fitNetwork, 500);
          setTimeout(fitNetwork, 1500);
        });
      })();
    </script>
    """
    if "</head>" in html:
        html = html.replace("</head>", dark_css + "</head>", 1)
    else:
        html = dark_css + html
    if "</body>" in html:
        html = html.replace("</body>", fit_script + "</body>", 1)
    else:
        html = html + fit_script
    return html


def graph_to_html(
    H: nx.Graph,
    title: str = "Distance Correlation Network",
    height: str = "100%",
    width: str = "100%",
) -> str:
    """Return standalone HTML for the interactive pyvis network."""
    if H.number_of_edges() == 0:
        return (
            "<!DOCTYPE html><html><body style='font-family:Segoe UI,sans-serif;"
            f"padding:2rem;background:{CANVAS_BG};color:{TITLE_COLOR}'>"
            f"<h2>{title}</h2><p>No edges to display for the current settings."
            "</p></body></html>"
        )

    weights = [data["weight"] for _, _, data in H.edges(data=True)]
    min_weight = min(weights)
    max_weight = max(weights)

    def normalize_weight(weight: float) -> float:
        if max_weight == min_weight:
            return 0.5
        return (weight - min_weight) / (max_weight - min_weight)

    n_nodes = H.number_of_nodes()
    spring_length = max(220, min(450, 100 + n_nodes * 3))
    repulsion = -25000 - n_nodes * 150

    net = Network(
        height=height,
        width=width,
        notebook=False,
        cdn_resources="in_line",
        heading=title,
        bgcolor=CANVAS_BG,
        font_color=NODE_FONT,
    )
    net.barnes_hut(
        gravity=repulsion,
        central_gravity=0.03,
        spring_length=spring_length,
        spring_strength=0.04,
        damping=0.25,
        overlap=0.8,
    )

    for node in H.nodes():
        net.add_node(
            node,
            label=str(node),
            size=16,
            color=NODE_COLOR,
            title=str(node),
            font={"size": 14, "face": "arial bold", "color": NODE_FONT},
        )

    for u, v, data in H.edges(data=True):
        weight = data["weight"]
        norm_weight = normalize_weight(weight)
        net.add_edge(
            u,
            v,
            width=0.4 + norm_weight * 0.6,
            color=_weight_to_color(norm_weight),
            title=f"weight: {weight:.3f}",
        )

    net.set_edge_smooth("continuous")
    net.toggle_physics(True)
    return _theme_pyvis_html(net.generate_html(notebook=False))

"""Light, vibrant design tokens shared by the Streamlit CSS and the Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio

PALETTE = {
    "bg": "#F6F7FB",
    "surface": "#FFFFFF",
    "border": "#E6E8F0",
    "text": "#1E2233",
    "muted": "#6B7085",
    "primary": "#6C5CE7",   # violet
    "teal": "#00B8A9",
    "coral": "#FF6B6B",
    "amber": "#FFB020",
    "sky": "#3BA6FF",
    "pink": "#F368E0",
    "lime": "#2ED573",
}

SERIES = [PALETTE[k] for k in ("primary", "teal", "coral", "amber", "sky", "pink", "lime")]
GAIN, LOSS = PALETTE["teal"], PALETTE["coral"]


def register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text"], size=13),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        colorway=SERIES,
        margin=dict(l=40, r=20, t=56, b=40),
        title=dict(font=dict(size=16, color=PALETTE["text"]), x=0.01, xanchor="left"),
        xaxis=dict(gridcolor="#EEF0F6", linecolor=PALETTE["border"], zeroline=False),
        yaxis=dict(gridcolor="#EEF0F6", linecolor=PALETTE["border"], zeroline=False),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PALETTE["surface"], bordercolor=PALETTE["border"], font=dict(color=PALETTE["text"])),
    )
    pio.templates["mktvol"] = template
    pio.templates.default = "plotly_white+mktvol"

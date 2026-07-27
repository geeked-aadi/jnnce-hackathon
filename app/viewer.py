import numpy as np
import plotly.graph_objects as go
from skimage import measure
from scipy import ndimage


LABEL_INFO = {
    1: {"name": "Necrotic/Non-enhancing Core", "color": "#facc15"},
    2: {"name": "Peritumoral Edema",           "color": "#22c55e"},
    4: {"name": "Enhancing Tumor",             "color": "#ef4444"},
}

# Broad anatomical region -> display color for the atlas overlay
REGION_COLOR_MAP = {
    "Frontal Lobe":    "#93c5fd",
    "Parietal Lobe":   "#a78bfa",
    "Temporal Lobe":   "#f9a8d4",
    "Occipital Lobe":  "#fca5a5",
    "Limbic Lobe":     "#fdba74",
    "Sub-lobar":       "#fde68a",
    "Anterior Lobe":   "#5eead4",   # cerebellum
    "Posterior Lobe":  "#2dd4bf",   # cerebellum
    "Brainstem":       "#f472b6",
}
DEFAULT_REGION_COLOR = "#9ca3af"


def _make_mesh(volume, level, color, opacity, name, spacing=(1, 1, 1),
                step_size=1, lighting=None, hovertext=None):
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            volume, level=level, step_size=step_size, spacing=spacing
        )
    except (ValueError, RuntimeError):
        return None

    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=False,
        lighting=lighting or dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.6),
        lightposition=dict(x=100, y=200, z=300),
        showscale=False,
        hoverinfo="text" if hovertext else "name",
        hovertext=hovertext,
        hoverlabel=dict(bgcolor="#171a23", font=dict(color="#e5e7eb", size=12),
                         bordercolor=color),
    )


def _extract_brain_mask(brain_volume):
    nonzero = brain_volume[brain_volume > 0]
    if nonzero.size == 0:
        return None
    thresh = np.percentile(nonzero, 25)
    binary = brain_volume > thresh

    labeled, n = ndimage.label(binary)
    if n > 1:
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        binary = labeled == (np.argmax(sizes) + 1)

    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_closing(binary, iterations=2)
    return binary.astype(np.float32)


def build_mask_viewer_html(mask, brain, title="3D Segmentation",
                            voxel_volume_cm3=1.0, spacing=(1.0, 1.0, 1.0),
                            atlas_native=None, label_names=None,
                            tumor_region_overlaps=None):
    """
    mask, brain            : as before
    atlas_native            : optional 3D int array, same shape as mask, from
                               anatomy.register_and_label() — native-space
                               Talairach lobe labels
    label_names              : dict[int,str] from anatomy.register_and_label()
    tumor_region_overlaps    : optional dict[label_id -> list of region-overlap
                               dicts] from anatomy.tumor_region_overlap(), used
                               to enrich tumor hover text with location info

    Returns (html: str, stats_rows: list[dict])
    """
    mask = np.asarray(mask).astype(np.uint8)
    brain = np.asarray(brain).astype(np.float32)
    if mask.ndim != 3 or brain.ndim != 3:
        raise ValueError("mask and brain must both be 3D arrays")
    if mask.shape != brain.shape:
        raise ValueError(f"mask shape {mask.shape} != brain shape {brain.shape}")

    labels = sorted({int(v) for v in np.unique(mask) if int(v) != 0})
    if not labels:
        return (
            "<div style='padding:16px; border:1px solid #2c3140; border-radius:10px; "
            "background:#0f1117; color:#9ca3af;'>No tumor regions detected.</div>"
        ), []

    fig = go.Figure()

    # --- Brain surface shell (context only, very translucent) ---
    brain_binary = _extract_brain_mask(brain)
    if brain_binary is not None:
        brain_smooth = ndimage.gaussian_filter(brain_binary, sigma=1.5)
        brain_mesh = _make_mesh(
            brain_smooth, level=0.5, color="#d1d5db", opacity=0.15,
            name="Brain", spacing=spacing, step_size=2,
            lighting=dict(ambient=0.6, diffuse=0.85, specular=0.05, roughness=0.8),
            hovertext="Brain (outer surface)",
        )
        if brain_mesh is not None:
            fig.add_trace(brain_mesh)

    # --- Anatomical lobes (cerebrum lobes, cerebellum, brainstem) ---
    if atlas_native is not None and label_names is not None:
        unique_ids = [int(u) for u in np.unique(atlas_native) if int(u) in label_names]
        for uid in unique_ids:
            region_binary = (atlas_native == uid).astype(np.float32)
            if region_binary.sum() < 50:  # skip noise-sized fragments
                continue
            region_smooth = ndimage.gaussian_filter(region_binary, sigma=1.2)
            region_name = label_names[uid]
            color = REGION_COLOR_MAP.get(region_name, DEFAULT_REGION_COLOR)

            region_mesh = _make_mesh(
                region_smooth, level=0.4, color=color, opacity=0.12,
                name=region_name, spacing=spacing, step_size=2,
                hovertext=f"<b>{region_name}</b>",
            )
            if region_mesh is not None:
                fig.add_trace(region_mesh)

    # --- Tumor sub-region surfaces with hover details + anatomical location ---
    stats_rows = []
    total_voxels = mask.size
    for label in labels:
        binary = (mask == label).astype(np.float32)
        binary_smooth = ndimage.gaussian_filter(binary, sigma=1.0)
        info = LABEL_INFO.get(label, {"name": f"Label {label}", "color": "#60a5fa"})

        voxel_count = int(np.sum(mask == label))
        volume_cm3 = round(voxel_count * voxel_volume_cm3, 2)
        percent = round(100 * voxel_count / total_voxels, 3)

        location_line = ""
        if tumor_region_overlaps and label in tumor_region_overlaps:
            top_regions = tumor_region_overlaps[label][:2]  # top 2 overlapping regions
            if top_regions:
                loc_str = ", ".join(f"{r['region']} ({r['percent_of_tumor']}%)" for r in top_regions)
                location_line = f"Location: {loc_str}<br>"

        hover_text = (
            f"<b>{info['name']}</b><br>"
            f"Voxels: {voxel_count:,}<br>"
            f"Volume: {volume_cm3} cm³<br>"
            f"% of scan: {percent}%<br>"
            f"{location_line}"
        )

        tumor_mesh = _make_mesh(
            binary_smooth, level=0.4, color=info["color"], opacity=0.95,
            name=info["name"], spacing=spacing, step_size=1,
            hovertext=hover_text,
        )
        if tumor_mesh is not None:
            fig.add_trace(tumor_mesh)

        stats_rows.append({
            "label": label,
            "name": info["name"],
            "color": info["color"],
            "voxel_count": voxel_count,
            "volume_cm3": volume_cm3,
            "percent_of_volume": percent,
            "location": tumor_region_overlaps.get(label, []) if tumor_region_overlaps else [],
        })

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e5e7eb")),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e5e7eb"),
        scene=dict(
            bgcolor="#0f1117",
            xaxis=dict(showbackground=False, showgrid=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        showlegend=True,
    )

    html = fig.to_html(full_html=True, include_plotlyjs="cdn",
                        default_height="520px", default_width="100%")

    return html, stats_rows
from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


SOURCE = Path(r"D:\downlaods\pipeline (2).drawio")
OUTPUT = Path(r"D:\downlaods\pipeline_C_v5_main_visible.drawio")

W, H = 1800, 900
INK = "#26363D"
MUTED = "#60717A"
PURPLE = "#8B6EAA"
LINE = "#74868B"
PALE = "#F5F1F8"


def add_cell(root: ET.Element, cell_id: str, value: str, style: str, *,
             vertex: bool = False, edge: bool = False, source: str | None = None,
             target: str | None = None, x: float | None = None,
             y: float | None = None, w: float | None = None,
             h: float | None = None, points: list[tuple[float, float]] | None = None):
    attrs = {"id": cell_id, "value": value, "style": style, "parent": "1"}
    if vertex:
        attrs["vertex"] = "1"
    if edge:
        attrs["edge"] = "1"
    if source:
        attrs["source"] = source
    if target:
        attrs["target"] = target
    cell = ET.SubElement(root, "mxCell", attrs)
    if vertex:
        ET.SubElement(cell, "mxGeometry", {
            "x": str(x), "y": str(y), "width": str(w), "height": str(h),
            "as": "geometry",
        })
    elif edge:
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            array = ET.SubElement(geometry, "Array", {"as": "points"})
            for px, py in points:
                ET.SubElement(array, "mxPoint", {"x": str(px), "y": str(py)})
    return cell


def box(root, cell_id, x, y, w, h, *, fill="#FFFFFF", stroke="#C6CDD0", radius=14,
        stroke_width=2, dashed=False):
    style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"strokeWidth={stroke_width};arcSize={radius};shadow=0;spacing=8;")
    if dashed:
        style += "dashed=1;dashPattern=6 4;"
    return add_cell(root, cell_id, "", style, vertex=True, x=x, y=y, w=w, h=h)


def label(root, cell_id, text, x, y, w, h, *, size=22, color=INK, bold=False,
          align="center", family="Arial", italic=False, fill="none"):
    font_style = (1 if bold else 0) + (2 if italic else 0)
    style = (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor={fill};"
             f"fontFamily={family};fontSize={size};fontColor={color};fontStyle={font_style};"
             f"align={align};verticalAlign=middle;spacing=3;")
    return add_cell(root, cell_id, text, style, vertex=True, x=x, y=y, w=w, h=h)


def edge(root, cell_id, source, target, *, color=LINE, width=3, dashed=False,
         points=None, exit_port="1;0.5", entry_port="0;0.5", label_text=""):
    ex, ey = exit_port.split(";")
    ix, iy = entry_port.split(";")
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
             f"html=1;strokeColor={color};strokeWidth={width};endArrow=block;endFill=1;"
             f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;entryX={ix};entryY={iy};"
             "entryDx=0;entryDy=0;")
    if dashed:
        style += "dashed=1;dashPattern=7 5;"
    add_cell(root, cell_id, label_text, style, edge=True, source=source, target=target, points=points)


def make_detail_page() -> ET.Element:
    diagram = ET.Element("diagram", {"id": "c-v5-detail", "name": "C_v5 detail"})
    graph = ET.SubElement(diagram, "mxGraphModel", {
        "dx": str(W), "dy": str(H), "grid": "1", "gridSize": "10", "guides": "1",
        "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
        "pageScale": "1", "pageWidth": str(W), "pageHeight": str(H), "math": "0",
        "shadow": "0", "background": "#FFFFFF",
    })
    root = ET.SubElement(graph, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    # Stage 1: the C_v5 guidance-generation path.
    box(root, "s1_panel", 24, 24, 1752, 650, fill="#F7F3FA", stroke="#C8B6D8", radius=24, stroke_width=2)
    label(root, "s1_title", "Stage 1 · C_v5 feedback-guided 2D guidance generation",
          52, 44, 1250, 50, size=31, color=INK, bold=True, family="Georgia", align="left")
    label(root, "s1_note", "One rendered view · 1024 × 1024",
          1340, 48, 380, 38, size=19, color=MUTED, italic=True)

    card_colors = {
        "rgba": ("#EAF1F6", "#6D8DA8"),
        "qwen": ("#F0E9F6", "#8B6EAA"),
        "sam": ("#E8F2EF", "#5C8C83"),
        "audit": ("#FFF0EF", "#C97E78"),
        "assembly": ("#FFF7E3", "#C7A247"),
        "output": ("#EAF4F0", "#5C8C83"),
    }
    for cell_id, x, y, w, h, key in [
        ("rgba_card", 62, 292, 215, 190, "rgba"),
        ("qwen_card", 345, 292, 225, 190, "qwen"),
        ("sam_card", 645, 292, 225, 190, "sam"),
        ("audit_card", 945, 260, 315, 250, "audit"),
        ("assembly_card", 1320, 248, 405, 282, "assembly"),
        ("map_card", 1380, 565, 285, 82, "output"),
    ]:
        fill, stroke = card_colors[key]
        box(root, cell_id, x, y, w, h, fill=fill, stroke=stroke, radius=16, stroke_width=2.2)

    # Input node.
    label(root, "rgba_title", "Rendered RGBA", 75, 305, 190, 37, size=24, bold=True)
    label(root, "rgba_detail", "renders_rot90/\n1024 × 1024\nα defines object silhouette",
          77, 350, 185, 104, size=20, color=MUTED)

    # Qwen outputs geometry prompts, not colors.
    label(root, "qwen_title", "Qwen3-VL", 360, 306, 195, 39, size=26, bold=True, color=PURPLE)
    label(root, "qwen_detail", "Part names\nBounding boxes\nPositive / negative points",
          360, 352, 195, 91, size=20)
    label(root, "qwen_boundary", "Geometry prompts only · no color prediction",
          353, 446, 209, 30, size=15, color=MUTED, italic=True)

    # SAM2 masks.
    label(root, "sam_title", "Prompted SAM2", 660, 306, 195, 39, size=25, bold=True, color="#4F7F76")
    label(root, "sam_detail", "One mask per part",
          660, 355, 195, 34, size=20)
    # Editable mask-stack glyph.
    box(root, "mask_tile_back", 704, 405, 93, 48, fill="#1D252A", stroke="#56676E", radius=5, stroke_width=1.4)
    box(root, "mask_tile_mid", 714, 399, 93, 48, fill="#1D252A", stroke="#56676E", radius=5, stroke_width=1.4)
    box(root, "mask_tile_front", 724, 393, 93, 48, fill="#1D252A", stroke="#56676E", radius=5, stroke_width=1.4)
    # White mask silhouettes as editable shapes.
    add_cell(root, "mask_blob_1", "", "ellipse;html=1;fillColor=#FFFFFF;strokeColor=none;", vertex=True, x=743, y=405, w=46, h=22)
    add_cell(root, "mask_blob_2", "", "ellipse;html=1;fillColor=#FFFFFF;strokeColor=none;", vertex=True, x=752, y=411, w=40, h=17)
    add_cell(root, "mask_count", "{Mᵢ}", f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;fontFamily=Arial;fontSize=18;fontColor={MUTED};align=center;verticalAlign=middle;",
             vertex=True, x=741, y=452, w=65, h=25)

    # Per-part audit and one-shot correction policy.
    label(root, "audit_title", "Per-part mask audit", 960, 274, 285, 38, size=25, bold=True, color="#A45E5A")
    label(root, "audit_detail", "Review each mask over the original RGB\n"
          "Semi-transparent red mask + 2 px red contour\n"
          "No black-background crop\n\n"
          "If rejected: rerun SAM2 once; replace the mask\n"
          "No iterative loop or quality ranking",
          967, 319, 270, 174, size=18, color=INK)

    # Palette synthesis and foreground completion.
    label(root, "assembly_title", "Palette-driven map assembly", 1335, 260, 375, 40,
          size=25, bold=True, color="#8B7132")
    box(root, "palette_row", 1345, 312, 355, 50, fill="#FFFFFF", stroke="#E2D5B1", radius=8, stroke_width=1.4)
    label(root, "palette_text", "colors_pure.json · 21 colors · use indices 1–20",
          1355, 318, 335, 37, size=18, bold=True)
    box(root, "reuse_row", 1345, 374, 355, 48, fill="#FFFFFF", stroke="#E2D5B1", radius=8, stroke_width=1.4)
    label(root, "reuse_text", "More than 20 parts → reuse palette colors",
          1355, 380, 335, 35, size=18)
    box(root, "alpha_row", 1345, 434, 355, 40, fill="#FFFFFF", stroke="#E2D5B1", radius=8, stroke_width=1.4)
    label(root, "alpha_text", "Clip all painted pixels to rendered α",
          1355, 437, 335, 33, size=18)
    box(root, "fill_row", 1345, 484, 355, 36, fill="#FFFFFF", stroke="#E2D5B1", radius=8, stroke_width=1.4)
    label(root, "fill_text", "Deterministically fill unassigned foreground",
          1355, 485, 335, 32, size=17)
    label(root, "map_title", "C guidance map PNG · 1024 × 1024",
          1390, 579, 265, 52, size=21, bold=True, color="#416D65")

    # Main path.
    edge(root, "e_rgba_qwen", "rgba_card", "qwen_card", color="#6686A9", width=3,
         exit_port="1;0.5", entry_port="0;0.5")
    edge(root, "e_qwen_sam", "qwen_card", "sam_card", color=PURPLE, width=3,
         exit_port="1;0.5", entry_port="0;0.5")
    label(root, "prompt_edge_label", "bbox + points", 565, 356, 105, 24, size=15, color=MUTED, fill=PALE)
    edge(root, "e_sam_audit", "sam_card", "audit_card", color=LINE, width=3,
         exit_port="1;0.5", entry_port="0;0.5")
    label(root, "masks_edge_label", "Per-part masks", 866, 356, 110, 24, size=15, color=MUTED, fill=PALE)
    edge(root, "e_audit_assembly", "audit_card", "assembly_card", color="#C97E78", width=3,
         exit_port="1;0.5", entry_port="0;0.5")
    edge(root, "e_assembly_map", "assembly_card", "map_card", color="#5C8C83", width=3,
         exit_port="0.5;1", entry_port="0.5;0")

    # RGB goes to SAM2; alpha is an independent foreground constraint for assembly.
    edge(root, "e_rgb_to_sam", "rgba_card", "sam_card", color="#6686A9", width=2.2,
         exit_port="0.5;0", entry_port="0.5;0", points=[(170, 220), (757, 220)])
    label(root, "rgb_label", "RGB image to SAM2", 410, 222, 190, 26, size=16, color="#547392", fill=PALE)
    edge(root, "e_alpha_to_assembly", "rgba_card", "assembly_card", color="#6B9B77", width=2.2,
         dashed=True, exit_port="0.5;1", entry_port="0.5;1", points=[(170, 530), (1522, 530)])
    label(root, "alpha_label", "Rendered α foreground constraint", 510, 535, 300, 28,
          size=16, color="#5D8D68", fill=PALE)

    # One-shot feedback edge from audit to SAM2.
    edge(root, "e_retry", "audit_card", "sam_card", color="#B96762", width=2.5, dashed=True,
         exit_port="0;0.22", entry_port="1;0.20", points=[(920, 315), (920, 205), (870, 205)])
    label(root, "retry_label", "Rejected mask → ≤1 SAM2 retry", 950, 209, 300, 25,
          size=16, color="#A45E5A", fill=PALE)

    # Stage 2 context, already shown in full in the overview page.
    box(root, "s2_panel", 24, 704, 1752, 165, fill="#F3F7F8", stroke="#B9C7CA", radius=20, stroke_width=1.8)
    label(root, "s2_title", "Stage 2 · Frozen SegViGen propagation (full geometry path on Overview page)",
          52, 720, 1480, 35, size=23, color=INK, bold=True, family="Georgia", align="left")
    label(root, "s2_flow", "C guidance map → DINOv3 features → tex-SLAT sampling (seed = 42) → texture decoder → slat_to_glb → segmented.glb",
          58, 766, 1682, 36, size=20, color="#536D79")
    label(root, "s2_geometry", "Geometry is reused from shared/preproc VXZ and shared/trellis_glb; it is not regenerated.",
          58, 810, 1682, 30, size=18, color=MUTED, italic=True)

    return diagram


def edit_overview_page(diagram: ET.Element) -> None:
    """Make the C_v5 filter/assembly detail visible on the opening overview page."""
    root = diagram.find("mxGraphModel/root")
    if root is None:
        raise ValueError("Overview page has no mxGraphModel/root")
    cells = {cell.get("id"): cell for cell in root.findall("mxCell")}

    def geometry(cell_id: str, **values: float) -> None:
        cell = cells.get(cell_id)
        geom = None if cell is None else cell.find("mxGeometry")
        if geom is None:
            raise ValueError(f"Missing geometry for {cell_id}")
        geom.attrib.update({key: str(value) for key, value in values.items()})

    def add_background(cell_id: str, x: float, y: float, w: float, h: float,
                       fill: str, stroke: str) -> None:
        cell = box(root, cell_id, x, y, w, h, fill=fill, stroke=stroke,
                   radius=10, stroke_width=1.3)
        root.remove(cell)
        anchor = cells["v91"]
        root.insert(list(root).index(anchor) + 1, cell)

    # Widen the existing module and split its contents into two clearly named stages.
    geometry("v91", x=1038, width=290)
    add_background("c5_overview_audit_card", 1045, 412, 276, 88, "#FFFCFC", "#E7C9C6")
    add_background("c5_overview_assembly_card", 1045, 510, 276, 154, "#FFFEFA", "#E4D8B6")

    funnel = cells["0OxHstXd27_m9RS2V8bF-474"]
    geometry(funnel.get("id"), x=1054, y=432, width=43, height=43)

    audit_title = cells["0OxHstXd27_m9RS2V8bF-481"]
    audit_title.set("value", '<font face="Times New Roman" style="font-size:21px;"><b>Per-part mask audit</b></font>')
    geometry(audit_title.get("id"), x=1100, y=420, width=207, height=27)

    # Turn the decorative plus into a small stage-transition arrow.
    transition = cells["0OxHstXd27_m9RS2V8bF-482"]
    transition.set("style", "shape=downArrow;whiteSpace=wrap;html=1;fillColor=#82b366;strokeColor=#82b366;strokeWidth=1;")
    geometry(transition.get("id"), x=1168, y=492, width=24, height=16)

    assembly_title = cells["0OxHstXd27_m9RS2V8bF-495"]
    assembly_title.set("value", '<font face="Times New Roman" style="font-size:20px;"><b>Guidance-map assembly</b></font>')
    geometry(assembly_title.get("id"), x=1056, y=514, width=255, height=28)

    # Keep the existing editable color-layer glyph, but size it as an inset icon.
    for cell_id, y in zip((
            "0OxHstXd27_m9RS2V8bF-494", "0OxHstXd27_m9RS2V8bF-493",
            "0OxHstXd27_m9RS2V8bF-492", "0OxHstXd27_m9RS2V8bF-491",
            "0OxHstXd27_m9RS2V8bF-490"), (550, 561, 572, 583, 594)):
        geometry(cell_id, x=1058, y=y, width=74, height=18)

    # Re-route the incoming audit connector to the widened module boundary.
    incoming = cells["0OxHstXd27_m9RS2V8bF-190"].find("mxGeometry")
    target_point = incoming.find("mxPoint[@as='targetPoint']") if incoming is not None else None
    if target_point is not None:
        target_point.set("x", "1038")

    # Short, legible labels retain the actual C_v5 operations without implying a learned decoder here.
    for cell_id, text, x, y, w, h, size, color, bold in [
        ("c5_overview_audit_line1", "Red overlay + 2 px contour", 1100, 448, 207, 22, 16, "#745B5A", False),
        ("c5_overview_audit_line2", "Rejected → SAM2 retry once", 1100, 472, 207, 22, 16, "#A45E5A", False),
        ("c5_overview_palette1", "21 colors · IDs 1–20", 1140, 548, 170, 24, 16, "#665A3B", False),
        ("c5_overview_palette2", "Reuse if >20 parts", 1140, 574, 170, 24, 16, "#665A3B", False),
        ("c5_overview_palette3", "Clip to rendered α", 1140, 600, 170, 24, 16, "#665A3B", False),
        ("c5_overview_palette4", "Deterministic gap fill", 1140, 626, 170, 24, 16, "#665A3B", False),
    ]:
        label(root, cell_id, text, x, y, w, h, size=size, color=color, bold=bold,
              family="Times New Roman")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    raw = SOURCE.read_text(encoding="utf-8")
    # Work from the supplied source and write a separate deliverable; never modify the original.
    source_root = ET.fromstring(raw)
    if source_root.tag != "mxfile" or not source_root.findall("diagram"):
        raise ValueError("Input is not a diagrams.net mxfile with a diagram page")
    if any(d.get("id") == "c-v5-detail" for d in source_root.findall("diagram")):
        raise ValueError("C_v5 detail page already exists in the supplied source")

    original_page_count = len(source_root.findall("diagram"))
    edit_overview_page(source_root.findall("diagram")[0])
    source_root.append(make_detail_page())
    modified = ET.tostring(source_root, encoding="unicode", short_empty_elements=True)
    parsed = ET.fromstring(modified)
    diagrams = parsed.findall("diagram")
    if len(diagrams) != original_page_count + 1:
        raise ValueError("Failed to append the detail page")
    detail = next(d for d in diagrams if d.get("id") == "c-v5-detail")
    ids = [c.get("id") for c in detail.findall(".//mxCell")]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate cell ids on detail page")
    valid_ids = set(ids)
    for cell in detail.findall(".//mxCell"):
        for attr in ("parent", "source", "target"):
            ref = cell.get(attr)
            if ref and ref not in valid_ids:
                raise ValueError(f"Dangling {attr}={ref} in {cell.get('id')}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="pipeline_C_v5_detail_", suffix=".tmp", dir=OUTPUT.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(modified)
        # Validate the exact candidate before replacing the target.
        ET.parse(temp_name)
        os.replace(temp_name, OUTPUT)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    print(f"Created: {OUTPUT}")
    print(f"Pages: {len(diagrams)}; detail page cells: {len(ids)}")


if __name__ == "__main__":
    main()

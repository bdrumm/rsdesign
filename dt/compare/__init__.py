"""Compare stage: pixel + structural metrics, the scalar loss, and diff visualisations.

    from dt.compare import evaluate, diff_map, structural_metrics, save_diff_image
    report = evaluate(doc, target_rgb)          # renders doc, compares, returns LossReport
    report.total, report.per_node, report.residual_regions

Submodules: ``pixel`` (ΔE map, metrics, residual regions, alignment), ``structural``
(geometry-fair node matching + IR metrics with line-fair text recall, layout_consistency),
``loss`` (evaluate / LossReport),
``visualize`` (side-by-side diff PNGs).
"""
from dt.compare.loss import LossReport, evaluate, per_node_errors, text_term_from_lines
from dt.compare.pixel import alignment_offset, diff_map, pixel_metrics, residual_regions, to_lab
from dt.compare.structural import (
    geom_type,
    has_layouts,
    layout_consistency,
    levenshtein,
    match_nodes,
    pair_iou,
    structural_metrics,
    text_cer,
    text_recovered,
    types_compatible,
)
from dt.compare.visualize import compose_diff_image, draw_boxes, heatmap, save_diff_image

__all__ = [
    "LossReport", "evaluate", "per_node_errors", "text_term_from_lines",
    "alignment_offset", "diff_map", "pixel_metrics", "residual_regions", "to_lab",
    "geom_type", "has_layouts", "layout_consistency", "levenshtein", "match_nodes", "pair_iou",
    "structural_metrics", "text_cer", "text_recovered", "types_compatible",
    "compose_diff_image", "draw_boxes", "heatmap", "save_diff_image",
]

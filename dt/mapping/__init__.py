"""Design-system ingestion and IR -> component/token mapping (module C).

    from dt.mapping import material3, map_document
    doc = map_document(doc, material3())

Public surface:
  * design_system: Token, Signature, ChildPattern, Variant, Slot, ComponentSpec, DesignSystem
  * material3():   Material 3 baseline catalog (derived from @material/web tokens)
  * matcher:       map_document, match_node, score_signature, derive_signature, snap_* helpers
  * ingest_figma:  from_figma_json / from_figma_file
  * ingest_library: from_dtcg / from_material_theme_builder / load_tokens_file
  * ingest_screenshots: from_reference_screenshots / learn_from_documents
"""
from dt.mapping.design_system import (  # noqa: F401
    ChildPattern, ComponentSpec, DesignSystem, Signature, Slot, Token, Variant, merge_tokens,
)
from dt.mapping.material3 import material3  # noqa: F401
from dt.mapping.matcher import (  # noqa: F401
    collapse_instances, derive_signature, expand_instances, instances, map_document, match_node, score_signature,
)

__all__ = [
    "ChildPattern", "ComponentSpec", "DesignSystem", "Signature", "Slot", "Token", "Variant", "merge_tokens",
    "material3", "map_document", "match_node", "score_signature", "derive_signature", "collapse_instances",
    "expand_instances", "instances",
]

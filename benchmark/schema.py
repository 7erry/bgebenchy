"""Index field lists and Atlas Search / Vector Search definitions.

23 filter fields are indexed for pre-filtering. 3 string fields are
full-text. The vector field is 1024-d cosine, ingested as fp16.
"""

from __future__ import annotations

FILTER_FIELDS: tuple[str, ...] = (
    "data.productNumber",
    "data.vendorName",
    "data.vendorPartNumber",
    "data.UserTypeID",
    "data.ParentID",
    "data.MotionId",
    "data.StepId",
    "data.attributes.UNSPSC",
    "data.attributes.ManufacturerID",
    "data.attributes.mfr_name_NM",
    "data.attributes.product_type_LOV",
    "data.attributes.Active",
    "data.attributes.WebStatus",
    "data.attributes.ItemUOM",
    "data.attributes.PGC_CODE",
    "data.attributes.bearing_type_LOV",
    "data.attributes.cage_type_LOV",
    "data.attributes.parent_id_ID",
    "data.attributes.eCOS_PGC",
    "data.attributes.Prop65",
    "data.attributes.company_name",
    "data.attributes.ItemNumber",
    "meta.source",
)

FULLTEXT_FIELDS: tuple[str, ...] = (
    "data.attributes.WebProductDescription",
    "data.attributes.DerivedProductDescription",
    "meta.descriptionKeywords",
)

VECTOR_PATH = "embedding"
VECTOR_DIMENSIONS = 1024
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODELS: dict[str, dict[str, str]] = {
    "bge-large-en-v1.5": {
        "id": "BAAI/bge-large-en-v1.5",
        "url": "https://huggingface.co/BAAI/bge-large-en-v1.5",
        "provider": "fastembed",
        "slug": "bge-large-en-v1.5",
    },
    "voyage-4-large": {
        "id": "voyage-4-large",
        "url": "https://huggingface.co/voyageai/voyage-4-large",
        "provider": "voyage",
        "slug": "voyage-4-large",
    },
}


def atlas_vector_index(name: str = "motion_vector") -> dict:
    """vectorSearch index: 1024-d cosine vector plus 23 filter fields."""
    fields: list[dict] = [
        {
            "type": "vector",
            "path": VECTOR_PATH,
            "numDimensions": VECTOR_DIMENSIONS,
            "similarity": "cosine",
            "quantization": "none",
        }
    ]
    fields.extend({"type": "filter", "path": path} for path in FILTER_FIELDS)
    return {
        "name": name,
        "type": "vectorSearch",
        "definition": {"fields": fields},
    }


def atlas_fulltext_index(name: str = "motion_fulltext") -> dict:
    """search index: 3 full-text string fields used in hybrid retrieval."""
    mappings = {
        path: {"type": "string", "analyzer": "lucene.standard"}
        for path in FULLTEXT_FIELDS
    }
    return {
        "name": name,
        "type": "search",
        "definition": {
            "mappings": {
                "dynamic": False,
                "fields": mappings,
            }
        },
    }

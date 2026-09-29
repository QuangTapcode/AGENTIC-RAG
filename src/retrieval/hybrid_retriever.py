"""Multilingual hybrid retrieval over the Qdrant medical collections.

The retrieval contract is deliberately explicit:

* ``original_query`` is embedded with multilingual E5 for dense retrieval.
* ``normalized_query`` is used as the stable, case-folded query representation.
* ``translated_query_en`` is sent to BM25 because the current WHO corpus is English.
* dense and sparse results are merged with Reciprocal Rank Fusion (RRF).

No translation API is required for the prototype.  A small medical phrase
dictionary makes common Vietnamese queries useful for BM25, while an optional
callable can be injected later for a production translation service or LLM.
The dictionary is intentionally observable: the query bundle records whether
translation was dictionary-based, identity English, externally supplied, or a
Vietnamese passthrough fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from qdrant_client import QdrantClient, models

try:
    from embedding.ingest_qdrant import (
        DENSE_VECTOR_NAME,
        MODEL_NAME,
        SPARSE_VECTOR_NAME,
        embed_query,
        load_encoder,
        query_sparse_vector,
    )
    from scope_router.scope_router import detect_language, normalise_query
except ImportError:  # Allows direct execution: python src/retrieval/hybrid_retriever.py.
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from embedding.ingest_qdrant import (  # type: ignore[no-redef]
        DENSE_VECTOR_NAME,
        MODEL_NAME,
        SPARSE_VECTOR_NAME,
        embed_query,
        load_encoder,
        query_sparse_vector,
    )
    from scope_router.scope_router import detect_language, normalise_query  # type: ignore[no-redef]


TranslationFunction = Callable[[str], str]


# Longer phrases must be replaced before their shorter components.
VIETNAMESE_MEDICAL_PHRASES: tuple[tuple[str, str], ...] = (
    ("bệnh tiểu đường", "diabetes"),
    ("tiểu đường", "diabetes"),
    ("bệnh hen suyễn", "asthma"),
    ("hen suyễn", "asthma"),
    ("bệnh phổi tắc nghẽn mạn tính", "chronic obstructive pulmonary disease"),
    ("bệnh phổi tắc nghẽn mạn", "chronic obstructive pulmonary disease"),
    ("bệnh tim mạch", "cardiovascular disease"),
    ("bệnh lao", "tuberculosis"),
    ("bệnh dại", "rabies"),
    ("viêm gan b", "hepatitis b"),
    ("viêm gan c", "hepatitis c"),
    ("sốt xuất huyết", "dengue"),
    ("huyết áp cao", "hypertension"),
    ("tăng huyết áp", "hypertension"),
    ("ung thư", "cancer"),
    ("covid 19", "covid-19"),
    ("covid-19", "covid-19"),
    ("đột quỵ", "stroke"),
    ("đau tim", "heart attack"),
    ("béo phì", "obesity"),
    ("viêm phổi", "pneumonia"),
    ("sốt rét", "malaria"),
    ("tiêu chảy", "diarrhoeal disease"),
    ("viêm màng não", "meningitis"),
    ("nhiễm trùng huyết", "sepsis"),
    ("rối loạn trầm cảm", "depressive disorder"),
    ("trầm cảm", "depression"),
    ("được phòng ngừa và điều trị như thế nào", "prevention and treatment"),
    ("có những dấu hiệu nào", "signs"),
    ("có triệu chứng gì", "symptoms"),
    ("điều trị như thế nào", "treatment"),
    ("phòng ngừa như thế nào", "prevention"),
    ("như thế nào", "how"),
    ("là gì", "what is"),
    ("triệu chứng", "symptoms"),
    ("dấu hiệu", "signs"),
    ("thuốc", "medicine"),
    ("dược phẩm", "medication"),
    ("điều trị", "treatment"),
    ("chẩn đoán", "diagnosis"),
    ("phòng ngừa", "prevention"),
    ("phòng bệnh", "prevention"),
    ("vắc xin", "vaccine"),
    ("vaccine", "vaccine"),
    ("tiêm chủng", "vaccination"),
    ("liều lượng", "dosage"),
    ("tác dụng phụ", "side effects"),
    ("chống chỉ định", "contraindications"),
    ("tương tác thuốc", "drug interactions"),
    ("phơi nhiễm", "exposure"),
    ("sau phơi nhiễm", "post exposure"),
    ("nguyên nhân", "causes"),
    ("lây truyền", "transmission"),
    ("có thể chữa", "can be treated"),
    ("bệnh", "disease"),
    ("của", "of"),
    ("và", "and"),
    ("có", "has"),
    ("gì", "what"),
)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)


class MedicalQueryTranslator:
    """Translate common medical Vietnamese phrases without an external API."""

    def __init__(self, external_translator: TranslationFunction | None = None) -> None:
        self.external_translator = external_translator
        self._phrases = tuple(
            (phrase, translation, _phrase_pattern(phrase))
            for phrase, translation in sorted(
                VIETNAMESE_MEDICAL_PHRASES,
                key=lambda item: len(item[0]),
                reverse=True,
            )
        )

    def translate(self, query: str, *, language: str) -> tuple[str, str, list[str]]:
        normalized = normalise_query(query)
        if language == "en":
            return normalized, "identity_en", []

        if self.external_translator is not None:
            translated = self.external_translator(query).strip()
            if translated:
                return normalise_query(translated), "external_translator", []

        translated = normalized
        matched: list[str] = []
        for phrase, replacement, pattern in self._phrases:
            if pattern.search(translated):
                translated = pattern.sub(replacement, translated)
                matched.append(phrase)

        if matched:
            return re.sub(r"\s+", " ", translated).strip(), "dictionary", matched
        return normalized, "passthrough_vi_no_translator", []


@dataclass(frozen=True)
class DiseaseCatalogEntry:
    """One manifest document plus the query phrases that identify it."""

    document_id: str
    title: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class DiseaseMatch:
    document_id: str
    title: str
    term: str


class DiseaseCatalog:
    """Resolve disease names to manifest document IDs before keyword retrieval."""

    def __init__(self, entries: Sequence[DiseaseCatalogEntry]) -> None:
        self.entries = tuple(entries)

    @staticmethod
    def _aliases(document_id: str, title: str) -> tuple[str, ...]:
        normalized_title = normalise_query(title)
        aliases = {normalized_title}
        without_parentheses = normalise_query(re.sub(r"\([^)]*\)", "", title))
        if without_parentheses:
            aliases.add(without_parentheses)
        for parenthetical in re.findall(r"\(([^)]*)\)", title):
            if parenthetical.strip():
                aliases.add(normalise_query(parenthetical))

        id_alias = normalise_query(document_id.removeprefix("who_").replace("_", " "))
        if id_alias:
            aliases.add(id_alias)

        # Match the common singular form as well (for example "disease" vs
        # the manifest title "diseases") without attempting general stemming.
        for alias in tuple(aliases):
            words = alias.split()
            if words and words[-1].endswith("s") and len(words[-1]) > 3:
                aliases.add(" ".join(words[:-1] + [words[-1][:-1]]))

        return tuple(sorted((alias for alias in aliases if alias), key=lambda value: (-len(value), value)))

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "DiseaseCatalog":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = [
            DiseaseCatalogEntry(
                document_id=str(document.get("document_id", "")).strip(),
                title=str(document.get("title", "")).strip(),
                aliases=cls._aliases(
                    str(document.get("document_id", "")),
                    str(document.get("title", "")),
                ),
            )
            for document in manifest.get("documents", [])
            if document.get("document_id") and document.get("title")
        ]
        return cls(entries)

    def match(self, *texts: str) -> list[DiseaseMatch]:
        searchable_text = " ".join(
            normalise_query(text) for text in texts if str(text).strip()
        )
        matches: list[DiseaseMatch] = []
        for entry in self.entries:
            matched_term = next(
                (alias for alias in entry.aliases if _phrase_pattern(alias).search(searchable_text)),
                None,
            )
            if matched_term:
                matches.append(
                    DiseaseMatch(
                        document_id=entry.document_id,
                        title=entry.title,
                        term=matched_term,
                    )
                )
        return matches


@lru_cache(maxsize=1)
def _default_disease_catalog() -> DiseaseCatalog:
    manifest_path = Path(__file__).resolve().parents[2] / "data" / "manifest.json"
    if not manifest_path.exists():
        return DiseaseCatalog(())
    return DiseaseCatalog.from_manifest(manifest_path)


@dataclass(frozen=True)
class QueryBundle:
    original_query: str
    normalized_query: str
    translated_query_en: str
    detected_language: str
    translation_method: str
    translated_terms: list[str] = field(default_factory=list)
    disease_document_ids: list[str] = field(default_factory=list)
    disease_titles: list[str] = field(default_factory=list)
    disease_terms: list[str] = field(default_factory=list)

    @property
    def document_scope_filter(self) -> dict[str, Any] | None:
        if not self.disease_document_ids:
            return None
        return {"document_id": {"$in": list(self.disease_document_ids)}}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"document_scope_filter": self.document_scope_filter}


def build_query_bundle(
    query: str,
    *,
    translator: MedicalQueryTranslator | None = None,
    disease_catalog: DiseaseCatalog | None = None,
) -> QueryBundle:
    original_query = query.strip()
    if not original_query:
        raise ValueError("Query must not be empty")
    normalized_query = normalise_query(original_query)
    detected = detect_language(normalized_query)
    query_translator = translator or MedicalQueryTranslator()
    translated_query, method, terms = query_translator.translate(
        normalized_query,
        language=detected,
    )
    catalog = disease_catalog or _default_disease_catalog()
    disease_matches = catalog.match(normalized_query, translated_query)
    return QueryBundle(
        original_query=original_query,
        normalized_query=normalized_query,
        translated_query_en=translated_query,
        detected_language=detected,
        translation_method=method,
        translated_terms=terms,
        disease_document_ids=[match.document_id for match in disease_matches],
        disease_titles=[match.title for match in disease_matches],
        disease_terms=[match.term for match in disease_matches],
    )


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    rank: int
    score: float
    source: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FusedHit:
    chunk_id: str
    rank: int
    rrf_score: float
    payload: dict[str, Any]
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rrf_fuse(
    dense_results: Sequence[RetrievalHit],
    sparse_results: Sequence[RetrievalHit],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[FusedHit]:
    """Fuse ranked lists using RRF while retaining per-retriever evidence."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    merged: dict[str, FusedHit] = {}
    for source, results in (("dense", dense_results), ("bm25", sparse_results)):
        for hit in results:
            item = merged.get(hit.chunk_id)
            if item is None:
                item = FusedHit(
                    chunk_id=hit.chunk_id,
                    rank=0,
                    rrf_score=0.0,
                    payload=hit.payload,
                )
                merged[hit.chunk_id] = item
            item.rrf_score += 1.0 / (rrf_k + hit.rank)
            if source not in item.sources:
                item.sources.append(source)
            if source == "dense":
                item.dense_rank = hit.rank
                item.dense_score = hit.score
            else:
                item.bm25_rank = hit.rank
                item.bm25_score = hit.score

    ordered = sorted(
        merged.values(),
        key=lambda item: (-item.rrf_score, item.dense_rank or 10**9, item.chunk_id),
    )
    for rank, item in enumerate(ordered[:top_k], start=1):
        item.rank = rank
    return ordered[:top_k]


@dataclass
class RetrievalResult:
    query: QueryBundle
    dense_results: list[RetrievalHit]
    sparse_results: list[RetrievalHit]
    fused_results: list[FusedHit]
    collection_name: str
    top_k_dense: int
    top_k_bm25: int
    top_k_fused: int
    rrf_k: int
    latency_ms: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "collection_name": self.collection_name,
            "configuration": {
                "top_k_dense": self.top_k_dense,
                "top_k_bm25": self.top_k_bm25,
                "top_k_fused": self.top_k_fused,
                "rrf_k": self.rrf_k,
            },
            "latency_ms": round(self.latency_ms, 3),
            "dense_results": [hit.to_dict() for hit in self.dense_results],
            "bm25_results": [hit.to_dict() for hit in self.sparse_results],
            "fused_results": [hit.to_dict() for hit in self.fused_results],
            "notes": self.notes,
        }


class HybridRetriever:
    """Query one chunk-size collection using dense, BM25 and RRF retrieval."""

    def __init__(
        self,
        *,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str,
        bm25_artifact_path: Path,
        model_name: str = MODEL_NAME,
        client: QdrantClient | None = None,
        encoder: Any | None = None,
        translator: MedicalQueryTranslator | None = None,
        disease_catalog: DiseaseCatalog | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.bm25_artifact_path = bm25_artifact_path
        self.model_name = model_name
        self.client = client or QdrantClient(url=qdrant_url)
        self.encoder = encoder
        self.translator = translator or MedicalQueryTranslator()
        self.disease_catalog = disease_catalog or _default_disease_catalog()
        self.bm25_artifact = json.loads(bm25_artifact_path.read_text(encoding="utf-8"))

    def _get_encoder(self) -> Any:
        if self.encoder is None:
            self.encoder = load_encoder(self.model_name)
        return self.encoder

    @staticmethod
    def _point_to_hit(point: Any, *, source: str, rank: int) -> RetrievalHit:
        payload = dict(point.payload or {})
        chunk_id = str(payload.get("chunk_id") or point.id)
        return RetrievalHit(
            chunk_id=chunk_id,
            rank=rank,
            score=float(point.score),
            source=source,
            payload=payload,
        )

    def _dense_search(
        self,
        query: str,
        top_k: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievalHit]:
        vector = embed_query(self._get_encoder(), query)
        search_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "query": vector.tolist(),
            "using": DENSE_VECTOR_NAME,
            "limit": top_k,
            "with_payload": True,
        }
        if query_filter is not None:
            search_kwargs["query_filter"] = query_filter
        response = self.client.query_points(**search_kwargs)
        return [
            self._point_to_hit(point, source="dense", rank=rank)
            for rank, point in enumerate(response.points, start=1)
        ]

    def _sparse_search(
        self,
        query_en: str,
        top_k: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> tuple[list[RetrievalHit], str | None]:
        vector: models.SparseVector = query_sparse_vector(query_en, self.bm25_artifact)
        if not vector.indices:
            return [], "translated_query_en_has_no_corpus_terms"
        search_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "query": vector,
            "using": SPARSE_VECTOR_NAME,
            "limit": top_k,
            "with_payload": True,
        }
        if query_filter is not None:
            search_kwargs["query_filter"] = query_filter
        response = self.client.query_points(**search_kwargs)
        return (
            [
                self._point_to_hit(point, source="bm25", rank=rank)
                for rank, point in enumerate(response.points, start=1)
            ],
            None,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k_dense: int = 10,
        top_k_bm25: int = 10,
        top_k_fused: int = 5,
        rrf_k: int = 60,
    ) -> RetrievalResult:
        for name, value in (
            ("top_k_dense", top_k_dense),
            ("top_k_bm25", top_k_bm25),
            ("top_k_fused", top_k_fused),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        bundle = build_query_bundle(
            query,
            translator=self.translator,
            disease_catalog=self.disease_catalog,
        )
        document_filter = self._document_filter(bundle)
        started = time.perf_counter()
        dense_results = self._dense_search(
            bundle.original_query,
            top_k_dense,
            query_filter=document_filter,
        )
        sparse_results, sparse_note = self._sparse_search(
            bundle.translated_query_en,
            top_k_bm25,
            query_filter=document_filter,
        )
        fused_results = rrf_fuse(
            dense_results,
            sparse_results,
            top_k=top_k_fused,
            rrf_k=rrf_k,
        )
        notes = [sparse_note] if sparse_note else []
        if bundle.disease_document_ids:
            notes.append("disease_scope=" + ",".join(bundle.disease_document_ids))
        return RetrievalResult(
            query=bundle,
            dense_results=dense_results,
            sparse_results=sparse_results,
            fused_results=fused_results,
            collection_name=self.collection_name,
            top_k_dense=top_k_dense,
            top_k_bm25=top_k_bm25,
            top_k_fused=top_k_fused,
            rrf_k=rrf_k,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            notes=notes,
        )

    @staticmethod
    def _document_filter(bundle: QueryBundle) -> models.Filter | None:
        if not bundle.disease_document_ids:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=bundle.disease_document_ids),
                )
            ]
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="User query in Vietnamese or English")
    parser.add_argument("--chunk-size", type=int, choices=(300, 800), default=300)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", MODEL_NAME))
    parser.add_argument("--top-k-dense", type=int, default=10)
    parser.add_argument("--top-k-bm25", type=int, default=10)
    parser.add_argument("--top-k-fused", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise print the retrieval trace",
    )
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root: Path = args.project_root
    retriever = HybridRetriever(
        qdrant_url=args.qdrant_url,
        collection_name=f"medical_chunks_{args.chunk_size}",
        bm25_artifact_path=(
            project_root / "data" / "vector_store" / f"chunks_{args.chunk_size}" / "bm25_index.json"
        ),
        model_name=args.model,
    )
    result = retriever.retrieve(
        args.query,
        top_k_dense=args.top_k_dense,
        top_k_bm25=args.top_k_bm25,
        top_k_fused=args.top_k_fused,
        rrf_k=args.rrf_k,
    )
    payload = result.to_dict()
    if args.output:
        output_path = args.output if args.output.is_absolute() else project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

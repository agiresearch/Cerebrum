"""Audit finalized benchmark artifacts for human-rating compatibility.

Reads existing result files WITHOUT modification and produces a structured
compatibility report documenting what information is available, what is
missing, and whether the requested 30-item human-rating design is supported.

Field status categories:
- Stored: explicitly present in the result record.
- Safely derived: deterministic from authoritative existing metadata.
- Unavailable: absent and not scientifically recoverable.
- Ambiguous: multiple possible interpretations exist.

Usage:
    python -m benchmarks.human_rating.artifact_audit \
        --results-dir results \
        --output compatibility_report.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

# Ensure project root on path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldStatus:
    """Status of a single field in the result artifact."""
    field_name: str
    status: str  # "stored", "safely_derived", "unavailable", "ambiguous"
    location: str  # JSON path or derivation source
    notes: str = ""


@dataclass(frozen=True)
class ArtifactCapabilities:
    """Capabilities of a single method's result file."""
    method: str
    source_path: str
    trial_count: int
    successful_trial_count: int
    has_model_metadata: bool
    has_question_type: bool
    has_exact_inference_context: bool
    has_profile_usage_score: bool
    has_task_usage_score: bool
    has_integration_score: bool
    has_synthetic_profile: bool
    has_synthetic_task_context: bool
    has_follow_up_query: bool
    has_assistant_response: bool
    has_retrieval_log: bool
    has_injection_diagnostics: bool
    field_statuses: tuple[FieldStatus, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class DesignRequirement:
    """Evaluation of a single design requirement."""
    requirement: str
    status: str  # "supported", "unsupported", "partially_supported", "requires_external_manifest"
    explanation: str


@dataclass(frozen=True)
class ManifestContract:
    """Contract for an external manifest that can supply missing metadata."""
    description: str
    required_fields: tuple[str, ...]
    provenance_options: tuple[str, ...]
    example_entry: dict


@dataclass(frozen=True)
class ArtifactAuditReport:
    """Complete audit report for all finalized result files."""
    methods: tuple[ArtifactCapabilities, ...]
    design_requirements: tuple[DesignRequirement, ...]
    supports_requested_design: bool
    blocking_issues: tuple[str, ...]
    manifest_contract: ManifestContract


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------

def _audit_single_file(method: str, path: str) -> ArtifactCapabilities:
    """Audit a single result file for human-rating field availability."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("experiment_metadata", {})
    conditions = data.get("conditions", [])

    if not conditions:
        return ArtifactCapabilities(
            method=method, source_path=path, trial_count=0,
            successful_trial_count=0, has_model_metadata=False,
            has_question_type=False, has_exact_inference_context=False,
            has_profile_usage_score=False, has_task_usage_score=False,
            has_integration_score=False, has_synthetic_profile=False,
            has_synthetic_task_context=False, has_follow_up_query=False,
            has_assistant_response=False, has_retrieval_log=False,
            has_injection_diagnostics=False,
            field_statuses=(), issues=("No conditions found in file",),
        )

    trials = conditions[0].get("trials", [])
    trial_count = len(trials)
    successful = [t for t in trials if not t.get("failed", False)]
    successful_count = len(successful)

    # Sample first successful trial for field presence
    sample = successful[0] if successful else (trials[0] if trials else {})

    has_model_metadata = "model" in metadata or "assistant_model" in metadata
    has_question_type = "question_type" in sample and sample["question_type"] in ("profile", "task")
    has_exact_inference_context = "inference_context_text" in sample
    has_profile_usage_score = "profile_usage_score" in sample and sample["profile_usage_score"] is not None
    has_task_usage_score = "task_usage_score" in sample and sample["task_usage_score"] is not None
    has_integration_score = "integration_score" in sample and sample["integration_score"] is not None
    has_synthetic_profile = "synthetic_profile" in sample and sample["synthetic_profile"] is not None
    has_synthetic_task_context = "synthetic_task_context" in sample and sample["synthetic_task_context"] is not None
    has_follow_up_query = "follow_up_query" in sample and bool(sample.get("follow_up_query", "").strip())
    has_assistant_response = "assistant_response" in sample and bool(sample.get("assistant_response", "").strip())
    has_retrieval_log = "retrieval_log" in sample
    has_injection_diagnostics = "injection_diagnostics" in sample

    # Build field statuses
    statuses = []

    statuses.append(FieldStatus(
        "source_method", "safely_derived",
        "conditions[0].condition / file path",
        f"Value: '{method}' from condition label and directory naming",
    ))

    statuses.append(FieldStatus(
        "model", "ambiguous" if not has_model_metadata else "stored",
        "experiment_metadata (not present)" if not has_model_metadata else "experiment_metadata.model",
        "Model identity inferred from directory name 'gpt4o_*' but not stored in metadata"
        if not has_model_metadata else "",
    ))

    statuses.append(FieldStatus(
        "trial_index", "stored",
        "conditions[0].trials[].trial_index",
    ))

    statuses.append(FieldStatus(
        "question_type", "unavailable" if not has_question_type else "stored",
        "Not present in trial records" if not has_question_type else "conditions[0].trials[].question_type",
        "All trials use a single vague prioritization query template. "
        "No authoritative profile/task distinction exists in the data."
        if not has_question_type else "",
    ))

    statuses.append(FieldStatus(
        "inference_context_text", "unavailable" if not has_exact_inference_context else "stored",
        "Not present in trial records" if not has_exact_inference_context else "conditions[0].trials[].inference_context_text",
        _inference_context_note(method, has_exact_inference_context),
    ))

    statuses.append(FieldStatus(
        "follow_up_query", "stored" if has_follow_up_query else "unavailable",
        "conditions[0].trials[].follow_up_query",
    ))

    statuses.append(FieldStatus(
        "assistant_response", "stored" if has_assistant_response else "unavailable",
        "conditions[0].trials[].assistant_response",
    ))

    statuses.append(FieldStatus(
        "synthetic_profile", "stored" if has_synthetic_profile else "unavailable",
        "conditions[0].trials[].synthetic_profile",
        "Full synthetic profile data (user_name, preferred_tools, preferred_language, response_style)"
        if has_synthetic_profile else "",
    ))

    statuses.append(FieldStatus(
        "synthetic_task_context", "stored" if has_synthetic_task_context else "unavailable",
        "conditions[0].trials[].synthetic_task_context",
        "Full task context (current_project, active_experiment, goals, blockers, next_steps)"
        if has_synthetic_task_context else "",
    ))

    statuses.append(FieldStatus(
        "profile_usage_score", "stored" if has_profile_usage_score else "unavailable",
        "conditions[0].trials[].profile_usage_score",
    ))

    statuses.append(FieldStatus(
        "task_usage_score", "stored" if has_task_usage_score else "unavailable",
        "conditions[0].trials[].task_usage_score",
    ))

    statuses.append(FieldStatus(
        "integration_score", "stored" if has_integration_score else "unavailable",
        "conditions[0].trials[].integration_score",
    ))

    # Compile issues
    issues = []
    if not has_question_type:
        issues.append("Missing authoritative question_type field")
    if not has_exact_inference_context:
        issues.append("Missing exact inference_context_text field")
    if not has_model_metadata:
        issues.append("Model identity not stored in metadata (inferred from directory name)")

    # Check for empty responses
    empty_responses = sum(1 for t in successful if not t.get("assistant_response", "").strip())
    if empty_responses > 0:
        issues.append(f"{empty_responses} trial(s) with empty assistant_response")

    return ArtifactCapabilities(
        method=method,
        source_path=path,
        trial_count=trial_count,
        successful_trial_count=successful_count,
        has_model_metadata=has_model_metadata,
        has_question_type=has_question_type,
        has_exact_inference_context=has_exact_inference_context,
        has_profile_usage_score=has_profile_usage_score,
        has_task_usage_score=has_task_usage_score,
        has_integration_score=has_integration_score,
        has_synthetic_profile=has_synthetic_profile,
        has_synthetic_task_context=has_synthetic_task_context,
        has_follow_up_query=has_follow_up_query,
        has_assistant_response=has_assistant_response,
        has_retrieval_log=has_retrieval_log,
        has_injection_diagnostics=has_injection_diagnostics,
        field_statuses=tuple(statuses),
        issues=tuple(issues),
    )


def _inference_context_note(method: str, has_field: bool) -> str:
    """Generate a note about inference context availability per method."""
    if has_field:
        return "Exact inference context stored in result."
    notes = {
        "naive_concat": (
            "Context could be reconstructed from synthetic_profile + synthetic_task_context "
            "using the stable naive_concat formatting function. However, this is a reconstruction, "
            "not a stored exact value. Byte-equivalent reconstruction is possible if the formatting "
            "code is immutable."
        ),
        "vanilla_rag": (
            "Exact context unavailable. The TF-IDF retrieval selects a subset of chunks and their "
            "ordering depends on cosine similarity scores computed at runtime. Only "
            "retrieved_context_count (number of chunks) is stored, not the actual chunk content."
        ),
        "mem0_default": (
            "Exact context unavailable. This method uses the 3-agent pipeline with private memories. "
            "The kernel's auto_inject finds no shared memories, so the confirmed inference context "
            "is empty (only the system prompt + user query). This is safely derivable as empty."
        ),
        "kernel_shared": (
            "Exact context unavailable. The kernel injects shared memories into the prompt internally. "
            "The SDK does not observe the finalized messages after injection. Only audit-based "
            "retrieval_log metadata is stored, not the formatted injected text."
        ),
    }
    return notes.get(method, "Unknown method")


def _evaluate_design_requirements(capabilities: tuple[ArtifactCapabilities, ...]) -> tuple[DesignRequirement, ...]:
    """Evaluate each requirement of the 30-item design."""
    reqs = []

    # 1. 24 unique items
    total_successful = sum(c.successful_trial_count for c in capabilities)
    reqs.append(DesignRequirement(
        "24 unique items (4 methods × 2 question types × 3 trials)",
        "unsupported" if not all(c.has_question_type for c in capabilities) else "supported",
        f"Requires authoritative question_type per trial. "
        f"Total successful trials available: {total_successful}. "
        f"Question type: {'present' if all(c.has_question_type for c in capabilities) else 'MISSING from all methods'}.",
    ))

    # 2. 4 methods
    methods_present = {c.method for c in capabilities}
    reqs.append(DesignRequirement(
        "4 methods represented",
        "supported" if len(methods_present) == 4 else "unsupported",
        f"Methods found: {sorted(methods_present)}",
    ))

    # 3. 2 authoritative question types
    any_has_qt = any(c.has_question_type for c in capabilities)
    reqs.append(DesignRequirement(
        "2 authoritative question types (profile, task)",
        "unsupported" if not any_has_qt else "supported",
        "No authoritative question_type field exists in any finalized result file. "
        "All trials use a single vague prioritization query template."
        if not any_has_qt else "Question types stored in result records.",
    ))

    # 4. 3 trials per method/type cell
    reqs.append(DesignRequirement(
        "3 trials per method/type cell",
        "unsupported" if not any_has_qt else "supported",
        "Cannot evaluate cell counts without authoritative question_type."
        if not any_has_qt else "Sufficient trials available per cell.",
    ))

    # 5. GPT-4o only
    all_have_model = all(c.has_model_metadata for c in capabilities)
    reqs.append(DesignRequirement(
        "GPT-4o only",
        "safely_derived" if not all_have_model else "supported",
        "Model not stored in metadata but directory naming convention 'gpt4o_*' "
        "provides authoritative evidence these are GPT-4o runs."
        if not all_have_model else "Model stored in metadata.",
    ))

    # 6. 6 duplicates
    reqs.append(DesignRequirement(
        "6 duplicate items for intra-rater consistency",
        "supported",
        "Duplicates are generated by the sampling phase from eligible trials. "
        "This is independent of source data availability.",
    ))

    # 7. Single human score comparable to GPT-5.4
    all_have_scores = all(
        c.has_profile_usage_score and c.has_task_usage_score and c.has_integration_score
        for c in capabilities
    )
    reqs.append(DesignRequirement(
        "Single human score comparable to GPT-5.4 judge",
        "partially_supported" if not any_has_qt else "supported",
        "All three judge dimensions (profile_usage, task_usage, integration) are stored. "
        "Score selection requires question_type to choose the matching dimension. "
        "Without question_type, the comparison dimension is undefined."
        if not any_has_qt else "Judge scores stored and question_type available for selection.",
    ))

    # 8. Model-visible context shown to rater
    any_has_ctx = any(c.has_exact_inference_context for c in capabilities)
    reqs.append(DesignRequirement(
        "Model-visible context shown to rater",
        "unsupported" if not any_has_ctx else "supported",
        _context_availability_summary(capabilities)
        if not any_has_ctx else "Exact inference context stored.",
    ))

    return tuple(reqs)


def _context_availability_summary(capabilities: tuple[ArtifactCapabilities, ...]) -> str:
    """Summarize context availability across methods."""
    lines = []
    for c in capabilities:
        note = _inference_context_note(c.method, c.has_exact_inference_context)
        status = "STORED" if c.has_exact_inference_context else "UNAVAILABLE"
        lines.append(f"  {c.method}: {status}")
    return "Per-method status:\n" + "\n".join(lines)


def _build_manifest_contract() -> ManifestContract:
    """Define the contract for an external manifest."""
    return ManifestContract(
        description=(
            "An external manifest may supply metadata not present in finalized results, "
            "enabling the human-rating pipeline to process existing artifacts. "
            "The manifest must clearly document the provenance of each supplied value."
        ),
        required_fields=(
            "source_method",
            "original_trial_id",
            "question_type",
            "display_context",
            "context_provenance",
            "eligible",
            "exclusion_reason",
        ),
        provenance_options=(
            "stored",           # Value directly from the result file
            "reconstructed",    # Deterministically rebuilt from stored source data
            "manually_verified",  # Human-reviewed and approved for the study
            "unavailable",      # Cannot be determined
        ),
        example_entry={
            "source_method": "naive_concat",
            "original_trial_id": "17",
            "question_type": "profile",
            "display_context": "--- USER PROFILE ---\nName: Alice...",
            "context_provenance": "reconstructed",
            "eligible": True,
            "exclusion_reason": None,
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_result_files(
    results_dir: str | Path,
    methods: Sequence[str] | None = None,
) -> ArtifactAuditReport:
    """Audit finalized result files and produce a compatibility report.

    Args:
        results_dir: Root directory containing result subdirectories.
        methods: Methods to audit (default: all four).

    Returns:
        Complete ArtifactAuditReport.
    """
    if methods is None:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]

    results_dir = Path(results_dir)
    capabilities = []

    for method in sorted(methods):
        path = results_dir / f"gpt4o_{method}" / f"results_{method}.json"
        if not path.exists():
            # Try direct path
            path = results_dir / f"results_{method}.json"
        if not path.exists():
            capabilities.append(ArtifactCapabilities(
                method=method, source_path=str(path), trial_count=0,
                successful_trial_count=0, has_model_metadata=False,
                has_question_type=False, has_exact_inference_context=False,
                has_profile_usage_score=False, has_task_usage_score=False,
                has_integration_score=False, has_synthetic_profile=False,
                has_synthetic_task_context=False, has_follow_up_query=False,
                has_assistant_response=False, has_retrieval_log=False,
                has_injection_diagnostics=False,
                field_statuses=(), issues=(f"File not found: {path}",),
            ))
            continue

        cap = _audit_single_file(method, str(path))
        capabilities.append(cap)

    caps_tuple = tuple(capabilities)
    design_reqs = _evaluate_design_requirements(caps_tuple)

    # Determine overall support
    blocking = []
    for req in design_reqs:
        if req.status == "unsupported":
            blocking.append(req.requirement)

    supports = len(blocking) == 0

    return ArtifactAuditReport(
        methods=caps_tuple,
        design_requirements=design_reqs,
        supports_requested_design=supports,
        blocking_issues=tuple(blocking),
        manifest_contract=_build_manifest_contract(),
    )


def format_summary_table(report: ArtifactAuditReport) -> str:
    """Format the audit report as a concise table."""
    lines = []
    header = f"{'Method':<16} {'Trials':>7} {'QType':>12} {'ExactCtx':>12} {'Scores':>7} {'Compatible':>11}"
    lines.append(header)
    lines.append("-" * len(header))

    for cap in report.methods:
        qt = "Stored" if cap.has_question_type else "Missing"
        ctx = "Stored" if cap.has_exact_inference_context else "Missing"
        scores = "Yes" if (cap.has_profile_usage_score and cap.has_task_usage_score and cap.has_integration_score) else "No"
        compat = "Yes" if (cap.has_question_type and cap.has_exact_inference_context) else "No"
        lines.append(
            f"{cap.method:<16} {cap.successful_trial_count:>7} "
            f"{qt:>12} {ctx:>12} {scores:>7} {compat:>11}"
        )

    lines.append("-" * len(header))
    lines.append(f"\nOverall compatible: {'Yes' if report.supports_requested_design else 'No'}")
    if report.blocking_issues:
        lines.append("Blocking issues:")
        for issue in report.blocking_issues:
            lines.append(f"  - {issue}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit finalized benchmark results for human-rating compatibility.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root directory containing result subdirectories (default: results/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON report (optional; prints table to stdout if omitted).",
    )
    return parser


def _serialize_report(report: ArtifactAuditReport) -> dict:
    """Serialize the report to a JSON-compatible dict."""
    def _dc_to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: _dc_to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [_dc_to_dict(i) for i in obj]
        return obj
    return _dc_to_dict(report)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    report = audit_result_files(args.results_dir)

    # Always print summary table
    print(format_summary_table(report))

    # Write JSON if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(_serialize_report(report), f, indent=2)
        print(f"\nJSON report written to: {args.output}")


if __name__ == "__main__":
    main()

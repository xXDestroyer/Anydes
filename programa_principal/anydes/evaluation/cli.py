from __future__ import annotations

import argparse
import json
from pathlib import Path

from anydes.evaluation.catalog import conversation_evaluation_catalog
from anydes.evaluation.contracts import HumanConversationReview
from anydes.evaluation.runner import (
    ConfiguredConversationEvaluationRunner,
    ConfiguredUnifiedInferenceComparisonRunner,
    LocalConversationEvaluationRunner,
    write_safe_report,
    write_safe_unified_comparison_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Anydes synthetic conversation quality evaluation."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation-results") / "conversation-quality.json",
        help="Safe JSON report path; responses and user messages are never included.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero status when any scenario fails.",
    )
    parser.add_argument(
        "--route",
        choices=("fast-local", "configured"),
        default="fast-local",
        help=(
            "Use deterministic local checks or the currently configured local/Ollama "
            "conversation route in an isolated data directory."
        ),
    )
    parser.add_argument(
        "--human-review",
        type=Path,
        help="Optional transcript-free JSON review for the human quality half.",
    )
    parser.add_argument(
        "--require-stabilized",
        action="store_true",
        help="Return a non-zero status unless technical and human stabilization gates pass.",
    )
    parser.add_argument(
        "--compare-unified-shadow",
        action="store_true",
        help=(
            "Run active and one-call unified reasoning over the same configured "
            "synthetic conversations. Requires --route configured."
        ),
    )
    args = parser.parse_args(argv)

    reviews = _load_human_reviews(args.human_review) if args.human_review else ()
    if args.compare_unified_shadow:
        if args.route != "configured":
            parser.error("--compare-unified-shadow requires --route configured")
        if reviews:
            parser.error("Human reviews cannot be attributed automatically to shadow output.")
        comparison = ConfiguredUnifiedInferenceComparisonRunner().run_comparison(
            conversation_evaluation_catalog()
        )
        write_safe_unified_comparison_report(comparison, args.output)
        print("Anydes Unified Inference Shadow")
        print(
            "Scenarios active/shadow .. "
            f"{comparison.active.passed_scenarios}/{comparison.active.total_scenarios} "
            f"vs {comparison.unified_shadow.passed_scenarios}/"
            f"{comparison.unified_shadow.total_scenarios}"
        )
        print(
            "Technical score .......... "
            f"{comparison.active.technical_score:.1%} vs "
            f"{comparison.unified_shadow.technical_score:.1%}"
        )
        print(
            "Average latency .......... "
            f"{comparison.active_average_turn_latency_ms:.1f} ms vs "
            f"{comparison.shadow_average_turn_latency_ms:.1f} ms"
        )
        print(
            "P95 latency .............. "
            f"{comparison.active_p95_turn_latency_ms:.1f} ms vs "
            f"{comparison.shadow_p95_turn_latency_ms:.1f} ms"
        )
        print(
            "Model calls per turn ..... "
            f"{comparison.active_calls_per_turn:.2f} vs "
            f"{comparison.shadow_calls_per_turn:.2f}"
        )
        print(f"Safe report .............. {args.output.resolve()}")
        if args.strict and (
            comparison.unified_shadow.passed_scenarios
            != comparison.unified_shadow.total_scenarios
        ):
            return 1
        return 0

    runner = (
        ConfiguredConversationEvaluationRunner()
        if args.route == "configured"
        else LocalConversationEvaluationRunner()
    )
    report = runner.run(conversation_evaluation_catalog(), human_reviews=reviews)
    write_safe_report(report, args.output)

    print("Anydes Conversation Quality")
    print(f"Route ............ {args.route}")
    print(f"Scenarios ........ {report.passed_scenarios}/{report.total_scenarios}")
    print(f"Technical score .. {report.technical_score:.1%}")
    if report.human_score is None:
        print("Human score ...... pending")
        print("Balanced score ... pending")
    else:
        print(f"Human score ...... {report.human_score:.1%}")
        print(f"Balanced score ... {report.balanced_score:.1%}")
    print(f"Stabilized ....... {'yes' if report.stabilization_ready else 'no'}")
    print(f"Average latency .. {report.average_latency_ms:.1f} ms")
    print(f"P95 latency ...... {report.p95_latency_ms:.1f} ms")
    print(f"Recovery rate .... {report.recovery_rate:.1%}")
    for dimension, score in sorted(report.dimension_scores.items()):
        print(f"{dimension:<19} {score:.1%}")
    print(f"Safe report ...... {args.output.resolve()}")

    if args.strict and report.passed_scenarios != report.total_scenarios:
        return 1
    if args.require_stabilized and not report.stabilization_ready:
        return 1
    return 0


def _load_human_reviews(path: Path) -> tuple[HumanConversationReview, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("reviews", ()) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Human review JSON must be a list or contain a reviews list.")
    return tuple(HumanConversationReview.from_dict(dict(item)) for item in rows)


if __name__ == "__main__":
    raise SystemExit(main())

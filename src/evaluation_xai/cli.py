import argparse

from .demo import run_demo


def main():
    parser = argparse.ArgumentParser(description="Evaluation & XAI offline tooling")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Run synthetic baselines (not PPO performance)"
    )
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--with-shap", action="store_true")
    imported = commands.add_parser(
        "import-rl",
        help="Analyze RL team's published test CSV (provisional, no model load)",
    )
    imported.add_argument("--models-dir", required=True)
    imported.add_argument("--output", required=True)
    check = commands.add_parser(
        "check-data", help="Validate data-pipeline processed inputs"
    )
    check.add_argument("--processed-dir", required=True)
    check.add_argument("--window", type=int, default=1)
    check.add_argument(
        "--replay-only",
        action="store_true",
        help="Permit a snapshot without prices; no price-return validation",
    )
    stats = commands.add_parser("anova", help="Run three mission tests independently")
    stats.add_argument("--samples", required=True)
    stats.add_argument("--output", required=True)
    xai = commands.add_parser(
        "explain-ppo", help="Replay trusted PPO with original processed data"
    )
    xai.add_argument("--model", required=True)
    xai.add_argument("--processed-dir", required=True)
    xai.add_argument("--rl-source-dir", required=True)
    xai.add_argument("--target-asset", required=True)
    xai.add_argument("--decision-date")
    xai.add_argument(
        "--reference-daily", help="Match this producer CSV before publishing SHAP"
    )
    xai.add_argument("--output", required=True)
    comparison = commands.add_parser(
        "compare-rl",
        help="Replay PPO runs and monthly return-based baselines (resumable)",
    )
    comparison.add_argument("--models-dir", required=True)
    comparison.add_argument("--processed-dir", required=True)
    comparison.add_argument("--rl-source-dir", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument(
        "--all-variants",
        action="store_true",
        help="Also replay lambda 0.5/5.0 (60 PPO); ANOVA keeps the predeclared set",
    )
    comparison.add_argument(
        "--resume",
        action="store_true",
        help="Continue a failed run in the same folder; refuses changed inputs",
    )
    comparison.add_argument(
        "--link-explanation",
        action="append",
        default=[],
        metavar="DIR",
        help="Existing explain-ppo bundle; attached only if the observation matches",
    )
    args = parser.parse_args()
    if args.command in {"check-data", "anova", "explain-ppo", "compare-rl"}:
        try:
            if args.command == "compare-rl":
                from .rl_comparison import ANOVA_VARIANTS, run_comparison
                from .team_import import VARIANTS

                result = run_comparison(
                    args.models_dir,
                    args.processed_dir,
                    args.rl_source_dir,
                    args.output,
                    variants=VARIANTS if args.all_variants else ANOVA_VARIANTS,
                    resume=args.resume,
                    explanation_dirs=args.link_explanation,
                )
                reconciliation = result["reference_reconciliation"]
                print(f"Provisional comparison: {args.output}/evaluation.json")
                print(
                    {
                        k: reconciliation[k]
                        for k in ("status", "mismatched_runs", "matched_run_count")
                    }
                )
                print(
                    {
                        link["source"]: link["status"]
                        for link in result["explanation_links"]
                    }
                )
                print(
                    {
                        k: v["status"]
                        for k, v in result["mission_anova"].items()
                        if isinstance(v, dict) and "status" in v
                    }
                )
            elif args.command == "check-data":
                from .data_loader import load_processed_window

                data = load_processed_window(
                    args.processed_dir, args.window, require_prices=not args.replay_only
                )
                print(
                    f"Input format validated: {len(data.asset_order)} assets, "
                    f"{len(data.features)} rows; FX/timing not certified."
                )
                print(data.verification)
            elif args.command == "anova":
                from .mission import run_mission_file

                result = run_mission_file(args.samples, args.output)
                print({name: item["status"] for name, item in result["tests"].items()})
            else:
                from .real_xai import explain_checkpoint

                explain_checkpoint(
                    args.model,
                    args.processed_dir,
                    args.rl_source_dir,
                    args.output,
                    target_asset=args.target_asset,
                    decision_date=args.decision_date,
                    reference_daily=args.reference_daily,
                )
                print(f"PPO explanation prepared: {args.output}/evaluation.json")
        except (ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Evaluation failed: {exc}\n")
        return
    if args.command == "import-rl":
        from .team_import import import_rl_results

        try:
            result = import_rl_results(args.models_dir, args.output)
        except (ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"RL import failed: {exc}\n")
        print(
            f"Provisional RL analysis: {args.output}/evaluation.json "
            f"({len(result['runs'])} runs)"
        )
        print(
            "WARNING: imported, not rerun; "
            "RF/MVO/SHAP and final acceptance remain incomplete."
        )
        return
    if args.command == "demo":
        try:
            result = run_demo(args.output, with_shap=args.with_shap)
        except ValueError as exc:
            parser.exit(2, f"Evaluation failed: {exc}\n")
        print(
            f"Synthetic demo ready: {args.output}/evaluation.json "
            f"({len(result['runs'])} runs)"
        )
        print(
            "WARNING: no real market data or trained PPO; "
            "not evidence of investment performance."
        )


if __name__ == "__main__":
    main()

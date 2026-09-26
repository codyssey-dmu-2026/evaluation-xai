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
    args = parser.parse_args()
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

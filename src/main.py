from __future__ import annotations

import argparse

from agent.orchestrator import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim graph research agent")
    parser.add_argument("--prompt", required=True, help="Research prompt")
    parser.add_argument("--k_per_query", type=int, default=8)
    parser.add_argument("--max_urls", type=int, default=30)
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--config", default="config/source_weights.json")
    args = parser.parse_args()

    run(
        prompt=args.prompt,
        k_per_query=args.k_per_query,
        max_urls=args.max_urls,
        out_dir=args.out_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()


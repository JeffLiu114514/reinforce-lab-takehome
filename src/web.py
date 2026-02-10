from __future__ import annotations

import argparse

import uvicorn

from agent.webapp import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Research agent web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.reload:
        uvicorn.run("agent.webapp:app", host=args.host, port=args.port, reload=True)
        return

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

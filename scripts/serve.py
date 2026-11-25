"""Run the CRAC API and dashboard locally.

    python scripts/serve.py              # http://127.0.0.1:8000
    python scripts/serve.py --port 8080 --reload

Scripts in this directory are thin argparse wrappers. Logic lives in crac/.
"""

import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description="Serve the CRAC API and dashboard.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="restart on code changes")
    args = p.parse_args(argv)

    import uvicorn

    print(f"CRAC dashboard -> http://{args.host}:{args.port}")
    print(f"API docs       -> http://{args.host}:{args.port}/docs")
    uvicorn.run("crac.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

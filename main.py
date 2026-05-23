"""
main.py — Entry point for the Customer Churn Prediction project.

Usage:
    python main.py            # Run full pipeline
    python main.py --api      # Start prediction API (requires fastapi + uvicorn)
"""

import argparse
import logging
import sys

from pipeline.churn_pipeline import run
from utils.logger import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Churn Prediction")
    parser.add_argument("--api", action="store_true",
                        help="Start the FastAPI prediction server")
    args = parser.parse_args()

    setup_logging()

    if args.api:
        try:
            import uvicorn
            from api.predict import app
            logging.getLogger(__name__).info("Starting API server …")
            uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
        except ImportError:
            print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        run()


if __name__ == "__main__":
    main()

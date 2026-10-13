"""Check saved comparison hashes and independently recompute return metrics."""

from __future__ import annotations

import argparse
import json

from alphaforge.comparison import verify_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(verify_comparison(args.directory), indent=2))

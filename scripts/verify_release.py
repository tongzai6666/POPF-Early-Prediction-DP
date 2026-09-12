import argparse
from utils.locking import verify_manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bundle_dir")
    args = p.parse_args()
    ok = verify_manifest(args.bundle_dir)
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

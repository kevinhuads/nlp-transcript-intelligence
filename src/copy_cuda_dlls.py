import os
import shutil
import site
import sys
from glob import glob


def iter_candidate_roots() -> list[str]:
    roots: list[str] = []
    for sp in site.getsitepackages():
        if os.path.isdir(sp):
            roots.append(sp)

    usp = site.getusersitepackages()
    if usp and os.path.isdir(usp):
        roots.append(usp)

    return roots


def find_dlls(roots: list[str]) -> list[str]:
    patterns = [
        "**\\cublas64_12.dll",
        "**\\cublasLt64_12.dll",
        "**\\cudart64_12.dll",
        "**\\cudnn*.dll",
        "**\\cusolver64_12.dll",
        "**\\cusparse64_12.dll",
        "**\\curand64_12.dll",
        "**\\cufft64_12.dll",
        "**\\nvrtc64_12*.dll",
        "**\\nvJitLink_12*.dll",
    ]

    found: set[str] = set()
    for root in roots:
        for pat in patterns:
            for p in glob(os.path.join(root, pat), recursive=True):
                if os.path.isfile(p):
                    found.add(p)

    return sorted(found)


def copy_missing(found: list[str], target_dir: str) -> tuple[int, int]:
    copied = 0
    skipped = 0

    for src in found:
        dst = os.path.join(target_dir, os.path.basename(src))
        if os.path.exists(dst):
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    return copied, skipped


def main() -> int:
    if os.name != "nt":
        print("Windows only.")
        return 2

    target_dir = os.path.dirname(sys.executable)
    roots = iter_candidate_roots()
    found = find_dlls(roots)

    if not found:
        print("No CUDA-related DLLs found under site-packages.")
        return 1

    copied, skipped = copy_missing(found, target_dir)

    print(f"target_dir={target_dir}")
    print(f"roots={len(roots)}")
    print(f"found={len(found)}")
    print(f"copied={copied}")
    print(f"skipped_existing={skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

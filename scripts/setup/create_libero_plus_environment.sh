#!/usr/bin/env bash
set -euo pipefail

TMD_PLUS_ENV_NAME="${TMD_PLUS_ENV_NAME:-tmdpolicy-libero-plus}"
TMD_BASE_ENV_NAME="${TMD_BASE_ENV_NAME:-tmdpolicy}"
TMD_LIBERO_PLUS_COMMIT="${TMD_LIBERO_PLUS_COMMIT:-4976dc30028e805ff8094b55501d532c48fec182}"
TMD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMD_LIBERO_PLUS_ROOT="${TMD_PROJECT_ROOT}/.deps/LIBERO-plus"

if ! conda env list --json | conda run --no-capture-output -n base python -c \
  'import json,sys; name=sys.argv[1]; data=json.load(sys.stdin); raise SystemExit(not any(p.rsplit("/",1)[-1] == name for p in data["envs"]))' \
  "${TMD_PLUS_ENV_NAME}"; then
  conda create -n "${TMD_PLUS_ENV_NAME}" --clone "${TMD_BASE_ENV_NAME}" -y
fi

mkdir -p "${TMD_PROJECT_ROOT}/.deps"
if [[ ! -d "${TMD_LIBERO_PLUS_ROOT}/.git" ]]; then
  git clone https://github.com/sylvestf/LIBERO-plus.git "${TMD_LIBERO_PLUS_ROOT}"
fi
if [[ "$(git -C "${TMD_LIBERO_PLUS_ROOT}" rev-parse HEAD)" != "${TMD_LIBERO_PLUS_COMMIT}" ]]; then
  git -C "${TMD_LIBERO_PLUS_ROOT}" fetch origin "${TMD_LIBERO_PLUS_COMMIT}"
  git -C "${TMD_LIBERO_PLUS_ROOT}" checkout --detach "${TMD_LIBERO_PLUS_COMMIT}"
fi

conda install -n "${TMD_PLUS_ENV_NAME}" -c conda-forge \
  expat fontconfig imagemagick unzip -y

TMD_PLUS_PYTHON="$(
  conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" \
    python -c 'import sys; print(sys.executable)'
)"
TMD_PIP_VERSION="$(
  conda run --no-capture-output -n "${TMD_BASE_ENV_NAME}" \
    python -c 'import pip; print(pip.__version__)'
)"
conda run --no-capture-output -n "${TMD_BASE_ENV_NAME}" \
  python -m pip \
  --python "${TMD_PLUS_PYTHON}" \
  install --no-cache-dir --force-reinstall "pip==${TMD_PIP_VERSION}"

conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" python -m pip install \
  -r "${TMD_LIBERO_PLUS_ROOT}/extra_requirements.txt"
conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" python -m pip install \
  "gym==0.26.2"
conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" python -m pip install \
  -e "${TMD_PROJECT_ROOT}[test]"

# LeRobot's `libero` extra installs the hf-libero distribution, which exports the
# same `libero` import package as LIBERO-Plus. Install project dependencies first,
# remove both possible providers, and install LIBERO-Plus last so Python cannot
# resolve the standard package ahead of the editable fork.
conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" python -m pip uninstall -y \
  hf-libero libero
conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" python -m pip install \
  --no-deps -e "${TMD_LIBERO_PLUS_ROOT}"

# LIBERO-Plus's setup.py searches for packages one directory too high and its
# editable wheel therefore contains no import mapping. Keep the repository root
# on sys.path so the outer `libero` namespace and inner `libero.libero` package
# resolve exactly as expected by the fork and LeRobot's LIBERO-Plus adapter.
conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" python -c \
  'import site,sys; from pathlib import Path; Path(site.getsitepackages()[0], "libero_plus_repo.pth").write_text(str(Path(sys.argv[1]).resolve()) + "\n", encoding="utf-8")' \
  "${TMD_LIBERO_PLUS_ROOT}"

# A global Conda LD_LIBRARY_PATH makes Mesa libEGL shadow NVIDIA EGL. Wand is
# preloaded narrowly by the verifier/evaluator instead, so remove any value
# left by an older version of this setup script.
if conda env config vars list -n "${TMD_PLUS_ENV_NAME}" | grep -q '^LD_LIBRARY_PATH'; then
  conda env config vars unset -n "${TMD_PLUS_ENV_NAME}" LD_LIBRARY_PATH
fi

TMD_ASSET_ROOT="${TMD_LIBERO_PLUS_ROOT}/libero/libero/assets"
TMD_EMBEDDED_ASSET_ROOT="${TMD_LIBERO_PLUS_ROOT}/libero/libero/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
if [[ ! -d "${TMD_ASSET_ROOT}/textures" ]]; then
  # Older runs extracted the archive verbatim. Its entries carry a long build-
  # machine prefix, so recover that already-downloaded tree before downloading
  # or extracting anything again.
  if [[ -d "${TMD_EMBEDDED_ASSET_ROOT}/textures" ]]; then
    if [[ -e "${TMD_ASSET_ROOT}" ]]; then
      echo "Refusing to replace incomplete LIBERO-Plus assets at ${TMD_ASSET_ROOT}" >&2
      exit 1
    fi
    mv -- "${TMD_EMBEDDED_ASSET_ROOT}" "${TMD_ASSET_ROOT}"
  else
    TMD_ASSET_ZIP="$(conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" python -c \
      'from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id="Sylvest/LIBERO-plus", filename="assets.zip", repo_type="dataset"))' \
      | tail -n 1)"
    TMD_ASSET_STAGING="$(mktemp -d "${TMD_PROJECT_ROOT}/.deps/libero-plus-assets.XXXXXX")"
    trap 'rm -rf -- "${TMD_ASSET_STAGING}"' EXIT
    unzip -q -o "${TMD_ASSET_ZIP}" -d "${TMD_ASSET_STAGING}"
    TMD_STAGED_ASSET_ROOT="$(find "${TMD_ASSET_STAGING}" -type d \
      -path '*/LIBERO-plus-0/assets' -print -quit)"
    if [[ -z "${TMD_STAGED_ASSET_ROOT}" || ! -d "${TMD_STAGED_ASSET_ROOT}/textures" ]]; then
      echo "Downloaded LIBERO-Plus archive does not contain the expected assets tree" >&2
      exit 1
    fi
    if [[ -e "${TMD_ASSET_ROOT}" ]]; then
      echo "Refusing to replace incomplete LIBERO-Plus assets at ${TMD_ASSET_ROOT}" >&2
      exit 1
    fi
    mv -- "${TMD_STAGED_ASSET_ROOT}" "${TMD_ASSET_ROOT}"
    rm -rf -- "${TMD_ASSET_STAGING}"
    trap - EXIT
  fi
fi

MUJOCO_GL=egl conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" \
  env -u LD_LIBRARY_PATH python "${TMD_PROJECT_ROOT}/scripts/setup/verify_libero_plus.py"

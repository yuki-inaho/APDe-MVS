set shell := ["bash", "-lc"]

default:
    @just --list

setup:
    uv venv
    uv sync

install:
    just setup
    just install-cu118
    just install-sam

install-cu118:
    uv pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118

install-sam:
    uv pip install "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git"

configure build_dir="build":
    cmake -S . -B {{build_dir}}

build build_dir="build":
    cmake --build {{build_dir}} --target APD

clean build_dir="build":
    rm -rf {{build_dir}}

run-scan data_dir +scans:
    @if [ -z "{{scans}}" ]; then \
        uv run python run.py --APD_path ./build/APD --data_dir {{data_dir}}; \
    else \
        uv run python run.py --APD_path ./build/APD --data_dir {{data_dir}} --scans {{scans}}; \
    fi

prep-scene colmap_dir:
    uv run python prepare_scene.py --data-dir {{colmap_dir}} --ensure-symlink

convert-colmap colmap_dir save_dir model_ext=".bin" script_args="":
    @set -euo pipefail; \
    if [ -d "{{colmap_dir}}/undist" ]; then \
        dense_folder="{{colmap_dir}}/undist"; \
    else \
        dense_folder="{{colmap_dir}}"; \
    fi; \
    uv run python tools/colmap2mvsnet.py --dense_folder "$dense_folder" --save_folder "{{save_dir}}" --model_ext {{model_ext}} {{script_args}}

reconstruct colmap_dir run_args="" convert_args="":
    @set -euo pipefail; \
    scene="$$(basename "{{colmap_dir}}")"; \
    dataset_root="results/$$scene"; \
    output_scan="$$dataset_root/$$scene"; \
    mkdir -p "$$dataset_root"; \
    uv run python prepare_scene.py --data-dir "{{colmap_dir}}" --ensure-symlink; \
    if [ -d "{{colmap_dir}}/undist" ]; then \
        dense_folder="{{colmap_dir}}/undist"; \
    else \
        dense_folder="{{colmap_dir}}"; \
    fi; \
    uv run python tools/colmap2mvsnet.py --dense_folder "$dense_folder" --save_folder "$output_scan" --model_ext .bin {{convert_args}}; \
    uv run python run.py --APD_path ./build/APD --data_dir "$$dataset_root" --scans "$$scene" {{run_args}}

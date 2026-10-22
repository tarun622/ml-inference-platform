.PHONY: install models quantize bench test cpu-repo up gateway loadtest

install:
	pip install -r requirements-dev.txt

models:            ## train the three MLPs and export FP32 ONNX (version 1)
	python models/train_models.py

quantize:          ## INT8 dynamic quantization -> version 2
	python quantization/quantize.py --mode dynamic

bench:             ## FP32 vs INT8: size, F1, latency, throughput
	python quantization/benchmark.py --out quantization/results.json

test: models quantize
	pytest -q

cpu-repo:          ## Triton repo variant for machines without a GPU
	python scripts/make_cpu_repo.py model_repository build/model_repository_cpu

up: models quantize cpu-repo
	docker compose up --build

gateway:           ## gateway alone, models run in-process
	uvicorn gateway.app.main:app --port 8080 --workers 2

loadtest:
	python loadtest/bench.py --url http://localhost:8080 --concurrency 1 16 64 256 --requests 20000

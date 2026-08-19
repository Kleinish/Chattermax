.PHONY: build web web-down corpus generate generate-all status validate prepare train export gpu
build:
	docker compose build
web:
	docker compose up -d --build webui
web-down:
	docker compose stop webui
corpus:
	docker compose run --rm corpus
generate:
	docker compose run --rm generator python scripts/generate.py
generate-all:
	docker compose run --rm generator python scripts/run_generation.py
status:
	docker compose run --rm generator python scripts/status.py
validate:
	docker compose run --rm generator python scripts/validate.py
prepare:
	docker compose run --rm generator python scripts/prepare_dataset.py
train:
	docker compose run --rm trainer bash scripts/train.sh
export:
	docker compose run --rm trainer bash scripts/export.sh
gpu:
	docker compose run --rm generator nvidia-smi

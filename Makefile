# Define constants

# General
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
current_abs_path := $(subst Makefile,,$(mkfile_path))

# Project
project_name := perpetual
project_dir := "$(current_abs_path)"
notebooks_dir := $(project_dir)notebooks
data_dir := $(project_dir)data

# Build Docker images and run containers in different modes
run-pipeline:
	cd $(current_abs_path) && \
	. ./set_architecture.sh && \
	docker compose up --build

run-notebooks:
	docker build -t $(project_name)-notebooks $(notebooks_dir)
	docker run -it --rm -p 8888:8888 \
		-v $(notebooks_dir):/$(project_name)/notebooks \
		-v $(data_dir):/$(project_name)/data \
		$(project_name)-notebooks jupyter lab \
		--port=8888 --ip='*' --NotebookApp.token='' \
		--NotebookApp.password='' --no-browser --allow-root

run-full-stack:
	cd $(current_abs_path) && \
	. ./set_architecture.sh && \
	docker compose --profile full-stack up --build
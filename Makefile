# Define constants

# General
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
current_abs_path := $(subst Makefile,,$(mkfile_path))

# Project
project_name := "perpetual"
project_dir := "$(current_abs_path)"

# Read environment variables from file
include .env

# Build Docker image and run containers in different modes
run-backend:
	cd $(current_abs_path) && \
	. ./set_architecture.sh && \
	docker compose --profile backend up --build
		
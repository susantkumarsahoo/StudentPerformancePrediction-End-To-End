
import os
from pathlib import Path

# Define project name (main source folder)
project_name = "src"

# Define additional folders
cicd_folder       = ".github/workflows"
configs_folder    = "configs"
data_folder       = "data"
notebooks_folder  = "notebooks"
static_css_folder = "static/css"
templates_folder  = "templates"
tests_folder      = "tests"
scripts_folder    = "scripts"
docs_folder       = "docs"

# List of files & folders to create

list_of_files = [
    # Source code (directly inside project_name)
    
    f"{project_name}/__init__.py",

    f"{project_name}/config/__init__.py",
    f"{project_name}/config/config_reader.py",

    f"{project_name}/connections_configuration/__init__.py",
    f"{project_name}/connections_configuration/aws_configuration_manager.py",
    f"{project_name}/connections_configuration/db_configuration_manager.py",

    f"{project_name}/constants/__init__.py",
    f"{project_name}/constants/aws_db_api_model_key.py",

    f"{project_name}/data_access/__init__.py",
    f"{project_name}/data_access/project_data_access.py",

    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/model_entity.py",

    f"{project_name}/features_components/__init__.py",
    f"{project_name}/features_components/data_ingestion.py",
    f"{project_name}/features_components/data_validation.py",
    f"{project_name}/features_components/data_preprocessing.py",
    f"{project_name}/features_components/feature_transformer.py",

    f"{project_name}/models/__init__.py",
    f"{project_name}/models/model_train.py",
    f"{project_name}/models/model_evaluate.py",
    f"{project_name}/models/model_predict.py",
    f"{project_name}/models/model_save_load.py",
    f"{project_name}/models/model_pusher.py",

    f"{project_name}/pipelines/__init__.py",
    f"{project_name}/pipelines/training_pipeline.py",
    f"{project_name}/pipelines/prediction_pipeline.py",
    f"{project_name}/pipelines/deployment_pipeline.py",

    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/logger.py",

    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception.py",

    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_handler.py",
    f"{project_name}/utils/perm_metrics.py",
    f"{project_name}/utils/helpers.py",

    f"{project_name}/visualization/__init__.py",
    f"{project_name}/visualization/plots.py",
    f"{project_name}/visualization/dashboard.py",

    # Cloud storage
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/cloud_storage/aws.py",

    # API
    f"{project_name}/api/__init__.py",
    f"{project_name}/api/routes.py",

    # Monitoring
    f"{project_name}/monitoring/__init__.py",
    f"{project_name}/monitoring/data_drift.py",
    f"{project_name}/monitoring/model_drift.py",

    # Outside project_name
    f"{cicd_folder}/ci_cd.yml",
    f"{configs_folder}/project_configs.yaml",
    f"{data_folder}/.gitkeep",
    f"{notebooks_folder}/01_note01.ipynb",
    f"{notebooks_folder}/02_note02.ipynb",

    f"{templates_folder}/project.html",
    f"{static_css_folder}/style.css",

    # Scripts
    f"{scripts_folder}/data_automation.py",
    f"{scripts_folder}/run_pipeline.sh",
    f"{scripts_folder}/db_migration.py",

    # Tests
    f"{tests_folder}/__init__.py",
    f"{tests_folder}/test_models.py",
    f"{tests_folder}/test_api.py",
    f"{tests_folder}/test_data_pipeline.py",

    # Docs
    f"{docs_folder}/data_dictionary.md",
    f"{docs_folder}/architecture.md",
    f"{docs_folder}/pipeline_flow.md",

    # Root-level files
    "requirements.txt",
#   "README.md",
    ".env",
    "setup.py",
#   ".gitignore",
    ".dockerignore",
    "Dockerfile",
    "app.py",
    "demo.py",
]


# Create files and directories
for filepath in list_of_files:
    file_path = Path(filepath)
    dir_path = file_path.parent
    os.makedirs(dir_path, exist_ok=True)  # Create directories
    if not file_path.exists():
        file_path.touch()  # Create empty file
        print(f"Created: {file_path}")
    else:
        print(f"Already exists: {file_path}")




# Folders (features_components, models, pipelines, logs, configs, data, docs, notebooks, cicd)

# __init__.py files inside each Python module folder

# Feature files (data_ingestion.py, data_validation.py, data_preprocessing.py, feature_transformer.py)

# Model files (model_train.py, model_evaluate.py, model_predict.py, model_save_load.py, model_pusher.py)

# Pipeline files (training_pipeline.py, prediction_pipeline.py, deployment_pipeline.py)

# file (ml_config.yaml)

# Log files (train.log, inference.log, error.log)

# Data placeholders (.gitkeep)

# Docs placeholders (.gitkeep)

# Notebook (01_EDA.ipynb)

# CI/CD (github_actions.yml)

# Helper files (utils.py, main.py, run_pipeline.sh)

# Root files (requirements.txt, README.md, .env, setup.py, .gitignore, Dockerfile, app.py, demo.py)





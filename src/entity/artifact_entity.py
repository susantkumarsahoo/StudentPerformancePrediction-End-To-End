from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_path: str
    processed_data_path: str
    train_data_path: str
    test_data_path: str
    metadata_path: str
    schema_path: str



@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: str
    data_drift_report_path: str
    missing_columns_report_path: str
    data_type_report_path: str
    validation_status_path: str



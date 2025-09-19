from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_path: str
    processed_data_path: str
    train_data_path: str
    test_data_path: str
    metadata_path: str
    schema_path: str
    
    
    

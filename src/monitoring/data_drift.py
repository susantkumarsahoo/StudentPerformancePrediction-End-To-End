import os
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


class DataDriftChecker:
    def __init__(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, report_dir: str):
        self.reference_data = reference_data
        self.current_data = current_data
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_report(self) -> dict:
        """Generate Evidently drift report."""
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_data, current_data=self.current_data)
        report_dict = report.as_dict()

        # Save JSON
        report_path = os.path.join(self.report_dir, "drift_report.json")
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)

        return report_dict

    def check_drift(self):
        """Check drift and raise error if detected."""
        report_dict = self.generate_report()
        drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]

        if drift_detected:
            raise ValueError("❌ Data drift detected between train and test!")
        else:
            print("✅ No data drift detected between train and test.")

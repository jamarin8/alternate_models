import joblib
import os
import lambda_local


class Model:
    name = "model_v1"
    cur_path = os.getcwd()
    model_pipeline = joblib.load(os.path.join(cur_path, "model_pipeline.joblib"))
    data = None

    def get_model(self):
        cur_path = os.getcwd()
        self.model_pipeline = joblib.load(os.path.join(cur_path, "model_pipeline.joblib"))

    def call(self, data):
        json_data = lambda_local.run_all(data)
        prob_fraud = self.model_pipeline.predict_proba(json_data)[:, 1]
        output = {
            "keyAttributes": {
                "scores": {
                    self.name: {
                        "probability_fraud": float(prob_fraud)
                    }
                },
                "applicationId": data["ca_id"]
            }
        }
        return output


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

        first = self.model_pipeline[0]
        second = self.model_pipeline[0:1]
        third = self.model_pipeline[0:2]
        fourth = self.model_pipeline[0:3]
        final_test = self.model_pipeline[3]

        first_out = first.transform(json_data)
        second_out = second.transform(json_data)
        third_out = third.transform(json_data)
        fourth_out = fourth.transform(json_data)
        test_out = final_test.predict_proba(json_data)

        output = {
            "keyAttributes": {
                "scores": {
                    self.name: {
                        "probability_fraud": float(prob_fraud)
                    }
                },
                "intermediate_values": {
                    "first_step": first_out,
                    "second_step": second_out,
                    "third_step": third_out,
                    "fourth_step": fourth_out,
                    "final_step": test_out
                },
                "applicationId": data["ca_id"]
            }
        }
        return output


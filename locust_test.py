
from locust import HttpUser, task, between
from main import TextToImageRequest
import json


class PerformanceTests(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def test_tf_predict(self):
        sample = TextToImageRequest(prompt="Cat playing with puppy")
        headers = {'Accept': 'application/json',
                   'Content-Type': 'application/json'}
        self.client.post("/text-to-image",
                         data=json.dumps(sample.dict()),
                         headers=headers)
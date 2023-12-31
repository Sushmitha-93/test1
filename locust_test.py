
from locust import HttpUser, task, between
from main import TextToImageRequest
import json


class PerformanceTests(HttpUser):
    host = "http://127.0.0.1:8000"
    wait_time = between(1, 3)

    @task(1)
    def text_to_image_task(self):
        sample = TextToImageRequest(prompt="Cat playing with puppy", steps=25, negative_prompt="disfigured, ugly, deformed")
        headers = {'Accept': 'application/json',
                   'Content-Type': 'application/json'}
        self.client.post("/text-to-image",
                         data=json.dumps(sample.model_dump()),
                         headers=headers)

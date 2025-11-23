build:
	docker build -f docker/Dockerfile -t titanic-inference .

train:
	docker run -it --rm \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/mlflow.db:/app/mlflow.db \
		-v $(PWD)/data:/app/data \
		titanic-inference python src/train.py

mlflow-ui:
	docker run -p 5001:5001 \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/mlflow.db:/app/mlflow.db \
		titanic-inference mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 --host 0.0.0.0

run-api:
	docker run -p 8000:8000 \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/mlflow.db:/app/mlflow.db \
		titanic-inference

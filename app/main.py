# Set environment variables before any imports
import os
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['FACENET_PYTORCH_HOME'] = '/tmp/facenet_pytorch'
os.environ["ULTRALYTICS_SETTINGS"] = "/tmp/ultralytics/settings.json"


from mangum import Mangum
from fastapi import FastAPI

# Import router after environment variables are set
from app.main import router

# Create FastAPI app
app = FastAPI(title="ProctorAI Lambda", 
              description="Lambda API for face proctoring",
              version="1.0.0")

# Add router
app.include_router(router)

handler = Mangum(app=app)
def lambda_handler(event, context):
    import logging
    import traceback
    try:
        logging.info(f"Lambda event: {event}")
        return handler(event, context)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "statusCode": 500,
            "body": "Internal Server Error"
        }
import asyncio
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes import code_execution, library_management
from mangum import Mangum

# ==========================================
#               FastAPI Setup
# ==========================================

app = FastAPI()

# ==========================================
#            Include Routers
# ==========================================
app.include_router(code_execution.router)
app.include_router(library_management.router)

# ==========================================
#       Include Mangum [AWS-LAMBDA]
# ==========================================

handler = Mangum(app=app)

# ==========================================
#               Main Function
# ==========================================
if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting FastAPI server on http://0.0.0.0:8082")
    uvicorn.run(app, host="0.0.0.0", port=8082)

from fastapi import FastAPI, Request
from judgeval import Tracer

tracer = Tracer.init(
    project_name="basic-distributed-tracing",
    resource_attributes={"service.name": "server"},
)

app = FastAPI()


@app.middleware("http")
async def activate_tracer(request: Request, call_next):
    tracer.set_active()
    return await call_next(request)


@Tracer.observe(span_type="agent")
def handle(message: str) -> str:
    return f"server received: {message}"


@app.post("/run")
async def run(request: Request):
    with Tracer.continue_trace(request.headers):
        payload = await request.json()
        return {"result": handle(payload["message"])}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

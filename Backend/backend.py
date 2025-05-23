from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class URLRequest(BaseModel):
    url: str

@app.post("/scan")
async def scan_url(request: URLRequest):
    url = request.url

    if "malicious" in url.lower():
        threat = True
        message = "Threat detected in the URL!"
    else:
        threat = False
        message = "URL is safe."

    return {"threat": threat, "message": message}

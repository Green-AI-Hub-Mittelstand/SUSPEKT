import os
import secrets

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
router = APIRouter()

USERNAME = os.getenv("ADMIN_USERNAME")
PASSWORD = os.getenv("ADMIN_PASSWORD")
if not USERNAME or not PASSWORD:
    raise RuntimeError(
        "ADMIN_USERNAME / ADMIN_PASSWORD environment variables are not set. "
        "Set them in your .env (see webapp/.env.example) before starting the app.")

# Single source of truth for the admin session marker. Templates check the
# same value inline via request.session.get('user') == 'admin'.
ADMIN_SESSION_VALUE = "admin"

def is_admin(request: Request) -> bool:
    return request.session.get("user") == ADMIN_SESSION_VALUE

@router.get("/admin-only", response_class=HTMLResponse)
@router.get("/login")
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if secrets.compare_digest(username, USERNAME) and secrets.compare_digest(password, PASSWORD):
        request.session["user"] = ADMIN_SESSION_VALUE
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Ungültiger Benutzername oder Passwort"
    })

@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)


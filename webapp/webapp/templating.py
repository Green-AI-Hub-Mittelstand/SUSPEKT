"""Gemeinsame Jinja2-Umgebung für alle Routen.

Zuvor legte jedes Modul sein eigenes ``Jinja2Templates`` an. Globale Werte
mussten dadurch achtmal gesetzt werden - wurde eines vergessen, fehlte der Wert
genau auf dessen Seiten. Eine gemeinsame Instanz verhindert das.
"""
import os

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")


def _static_version() -> str:
    """Änderungszeit von style.css als Versionskennung.

    Ohne sie liefert der Browser die zwischengespeicherte Datei weiter aus -
    Layoutkorrekturen kommen dann schlicht nicht an.
    """
    try:
        return str(int(os.path.getmtime("static/style.css")))
    except OSError:
        return "0"


templates.env.globals["static_version"] = _static_version()

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

#from .auth import get_current_user
from .neo4j_database import db
from .resource_efficiency import fetch_reusable_components, analyze_reusable_components
from .transport_emission import get_transport_emissions,get_total_distance

router = APIRouter(prefix="/inventory")
templates = Jinja2Templates(directory="templates")

# Simulierte Nutzer-Datenbank
users_db = {
    "admin": {"username": "admin", "role": "admin"},
    "user": {"username": "user", "role": "user"}
}


@router.get("", response_class=HTMLResponse)
async def get_inventory(request: Request):
    """Holt die Inventardaten aus der Neo4J-Datenbank und rendert sie im Template"""
    try:
       #query = """
        #MATCH (f:Furniture)<-[:PART_OF]-(c:Component) RETURN f.name AS Möbelstück, c.name AS Komponente, c.material AS Material , c.state AS Zustand;

        #"""
        query = """MATCH (o:Order)<-[:PART_OF]-(c:Component) RETURN o.order_id AS Auftrag, c.class AS Komponente, c.farbe AS Material, c.zustand AS Zustand; """
        inventory = db.run_query(query)

        return templates.TemplateResponse("inventory.html", {
            "request": request,
            "inventory": inventory,
            "username": "admin",
            "role": "admin"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
@router.put("/components/update/{component_id}")
async def update_component(
    component_id: str, data: dict, user: dict = Depends(get_current_user)
):
    #Nur Admins können Komponenten bearbeiten
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Keine Berechtigung")

    # Hier sollte die Neo4j-Query stehen
    return JSONResponse(content={"message": "Komponente aktualisiert"})


@router.delete("/components/delete/{component_id}")
async def delete_component(component_id: str, user: dict = Depends(get_current_user)):
    #Nur Admins können Komponenten löschen
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Keine Berechtigung")

    # Hier sollte die Neo4j-Query stehen
    # await broadcast_update('{"action": "delete", "id": "' + component_id + '"}')
    return JSONResponse(content={"message": "Komponente gelöscht"})
"""
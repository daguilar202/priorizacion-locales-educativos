# Despliegue Streamlit

Esta carpeta contiene solo los archivos necesarios para publicar la versión Streamlit.

## Subir a GitHub desde el navegador

Suba a la raíz del repositorio:
- app_priorizacion_regional.py
- criterios_priorizacion.py
- config_priorizacion.json
- requirements.txt
- header_digeie_diplan.png
- plantilla_actualizacion_s1.csv
- plantilla_seguridad.csv
- carpeta data/ con locales_priorizacion.csv.gz

`locales_priorizacion.csv.gz` se lee directamente con pandas; no debe descomprimirse.

## Streamlit Community Cloud

Main file path: `app_priorizacion_regional.py`

No se requiere `index.html` ni los archivos `.bat` para el despliegue web.

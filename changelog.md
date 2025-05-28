# Changelog del Pipeline de MLOps para Predicción Clínica

Este changelog compara la versión inicial del pipeline planteada en la Unidad 1 con la versión reestructurada final que incorpora prácticas modernas de MLOps aplicadas al caso clínico.

| Componente           | Versión Inicial          | Versión Final                               |
|----------------------|---------------------------|---------------------------------------------|
| Entrada de datos     | CSV simple                | CSV + API FHIR, validación HL7              |
| Preprocesamiento     | Limpieza básica           | Pipeline validado, trazabilidad             |
| Entrenamiento        | Script manual             | Iteración controlada con métricas           |
| Despliegue           | Local                     | Docker + FastAPI + ONNX                     |
| Interfaz             | No incluía                | Streamlit funcional                         |
| Registro             | No había                  | PostgreSQL + logs JSON                      |
| Monitoreo            | Ausente                   | Prometheus + Grafana + MLflow               |
| Escalabilidad        | Limitada                  | Nube y modularización completa              |
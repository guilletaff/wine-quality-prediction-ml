# Predicción de la Calidad del Vino Tinto (Wine Quality Prediction)

Este proyecto aplica técnicas de Machine Learning y Análisis Exploratorio de Datos (EDA) para predecir la calidad de los vinos tintos basándose en sus propiedades físico-químicas. Se trata de un problema de regresión donde se busca estimar una puntuación numérica de calidad.

## Requisitos
Para poder ejecutar los notebooks y reproducir los resultados de este proyecto, necesitas tener instalado Python 3.x junto con las siguientes librerías:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

Si lo deseas, se ha adjuntado un archivo "requirements.txt" el cual permite instalar las librerías directamente mediante el comando en terminal:

```bash
pip install -r requirements.txt
```

## 1. El Problema y Objetivo
La calidad de un vino suele ser determinada por catadores expertos en un proceso subjetivo. El objetivo de este proyecto es determinar si las medidas de laboratorio (pH, alcohol, acidez, etc.) pueden predecir esa puntuación de manera objetiva.

**Preguntas clave:**
* ¿Qué componentes químicos influyen más en la calidad?
* ¿Es posible automatizar la clasificación de calidad con un margen de error aceptable?

## 2. Descripción de los Datos
El dataset utilizado es el WineQT, descargado a traves de "Kaggle" (https://www.kaggle.com/datasets/yasserh/wine-quality-dataset), que consta de 1143 registros de vinos tintos.
* **Variable Objetivo:** `quality` (puntuación de 3 a 8).
* **Variables Predictoras (11):** 
    * Acidez (fija, volátil, cítrica).
    * Azúcar residual y Cloruros.
    * Dióxido de azufre (libre y total).
    * Densidad, pH, Sulfatos y Alcohol.

**Problema del dataset:** Se detectó un fuerte desbalanceo. La mayoría de los vinos tienen calidades 5 y 6, mientras que hay muy pocas muestras de calidades extremas (3 o 8), lo que representa un reto para el aprendizaje del modelo.

## 3. Estructura del Proyecto
El flujo de trabajo está dividido en cuatro fases secuenciales, documentadas en Jupyter Notebooks:

* **`00_EDA_Proyecto.ipynb` (Análisis Exploratorio de Datos):** Carga de datos, análisis de distribuciones, detección de outliers y visualización de correlaciones, además de la eliminación de duplicados y valores nulos. Identificamos las variables con mayor impacto en la calidad.
* **`01_feature_engineering.ipynb` (Ingeniería de Características):** Limpieza de datos, división en conjuntos de entrenamiento y prueba (Train/Test Split) y escalado de variables utilizando `StandardScaler` para preparar los datos para los modelos.
* **`02_modelado.ipynb` (Entrenamiento de Modelos):** Implementación de tres modelos de regresión:
  * Regresión Lineal Simple
  * Regresión Ridge (Regularización L2)
  * Regresión Lasso (Regularización L1)
* **`03_evaluacion.ipynb` (Evaluación y Conclusiones):** Comparativa de métricas de error (MSE, MAE, R²), análisis de los coeficientes de los modelos y conclusiones finales sobre el rendimiento.

## 4. Resultados del Modelo
Tras las pruebas, los modelos lineales mostraron resultados muy similares, indicando estabilidad en las predicciones:

| Métrica | Valor (Test) | Interpretación |
| :--- | :--- | :--- |
| **MAE** | ~0.56 | El modelo se equivoca en promedio ~0.5 puntos de calidad. |
| **RMSE** | ~0.71 | Raíz del error cuadrático medio en la escala original. |
| **R²** | **0.388** | El modelo explica el 38.8% de la variabilidad de la calidad. |

### Factores Influyentes
El análisis de los coeficientes determinó que:
* **Alcohol:** Es el factor con mayor impacto positivo (a más alcohol, mayor calidad percibida).
* **Acidez Volátil:** Es el factor con mayor impacto negativo (altos niveles degradan la calidad).

## 5. Conclusiones

1. **Generalización:** El modelo no presenta overfitting, ya que el error en el conjunto de prueba es consistente con el de entrenamiento.
2. **Limitaciones:** Tiende a predecir valores cercanos a la media (5-6) debido a la escasez de ejemplos de vinos "excelentes" o "muy malos". 
3. **Selección de características:** El modelo Lasso redujo a cero los coeficientes de la `densidad` y el `dióxido de azufre libre`, confirmando que aportaban información redundante o poco relevante para nuestra predicción.
4. **Métricas de rendimiento:** 
    * **R² (Coeficiente de Determinación):** ~0.39. El modelo no sufre de sobreajuste (el R² en test es ligeramente superior al de train). Dado que la calidad es subjetiva y depende de factores no medibles químicamente, es un resultado coherente.
   * **MAE (Error Absoluto Medio):** 0.56. En promedio, el modelo se equivoca en apenas medio punto dentro de la escala de calidad (3 a 8).

---

## Autores
Proyecto desarrollado por:
* **Oriol García Miró** - oriol.garcia@cunef.edu
* **Rodrigo González** - r.gonzalezjimenez@cunef.edu
* **Guillermo Taffouraud** - g.taffouraud@cunef.edu

---
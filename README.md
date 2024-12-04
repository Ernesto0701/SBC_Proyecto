# K-Means Clustering en Pokémon

## Descripción del Proyecto
Este proyecto implementa el algoritmo de aprendizaje no supervisado **K-Means Clustering** aplicado a los datos de Pokémon, con el objetivo de identificar patrones ocultos y agrupar a los Pokémon en clústeres basados en sus atributos principales. La finalidad es proporcionar herramientas que permitan mejorar estrategias competitivas y facilitar el análisis de los datos para jugadores, diseñadores de videojuegos e investigadores.

## Motivación
El análisis de datos permite extraer conocimiento útil a partir de grandes volúmenes de información. En el contexto de Pokémon, cada criatura posee atributos únicos como estadísticas de combate, tipos y habilidades. Este proyecto busca emplear técnicas de aprendizaje no supervisado para explorar estos datos y ayudar a los jugadores a diseñar equipos competitivos equilibrados.

## Objetivos

### Objetivo General
Agrupar Pokémon en clústeres según sus estadísticas de combate utilizando **K-Means Clustering**, para identificar patrones y categorías que faciliten su interpretación y aplicación.

### Objetivos Específicos
- Preprocesar el dataset eliminando atributos irrelevantes y normalizando las estadísticas principales (HP, Ataque, Defensa, etc.).
- Determinar el número óptimo de clústeres utilizando el **método del codo**.
- Aplicar el algoritmo **K-Means Clustering** al dataset procesado.
- Visualizar y analizar los resultados mediante gráficos.
- Evaluar las implicaciones de los clústeres en términos de estrategia competitiva y diseño de videojuegos.
- Documentar los hallazgos y proporcionar una base reproducible para investigaciones futuras.

## Implementación
El proyecto combina algoritmos de aprendizaje no supervisado con una interfaz gráfica interactiva que permite realizar análisis flexibles del dataset de Pokémon. 

### Herramientas Utilizadas
- **Python** para la implementación del algoritmo.
- **Bibliotecas**: `Scikit-learn`, `Tkinter`, `Plotly`, `KneeLocator`, `Matplotlib`.
- **Estadísticas principales**: HP, Ataque, Defensa, Velocidad y más.

### Flujo del Proyecto
1. **Preprocesamiento**: Eliminación de atributos irrelevantes y normalización de los datos.
2. **Método del Codo**: Identificación del número óptimo de clústeres basados en la métrica WCSS.
3. **Aplicación del Algoritmo**: Agrupamiento de Pokémon en clústeres.
4. **Visualización**: Uso de gráficos para representar la distribución y análisis de los clústeres.

## Resultados
- **Agrupamiento Eficiente**: Los Pokémon fueron agrupados en categorías significativas según sus estadísticas principales.
- **Estrategia Competitiva**: Los clústeres permiten identificar combinaciones equilibradas para los equipos de batalla.
- **Validación**: La calidad de los clústeres fue evaluada mediante el **Silhouette Score**, que confirmó su coherencia y separación adecuada.

## Aplicaciones Futuras
- Comparación con otros algoritmos de agrupamiento, como DBSCAN o Agglomerative Clustering.
- Ampliación del dataset con atributos adicionales como habilidades o movimientos.
- Desarrollo de una interfaz más interactiva para la exploración de datos en tiempo real.

## Requisitos del Sistema
- **Python** 3.8 o superior
- Bibliotecas requeridas:
  ```
  pip install -r requirements.txt
  ```
  - `Scikit-learn`
  - `Matplotlib`
  - `Tkinter`
  - `Plotly`
  - `KneeLocator`

## Cómo Usar
1. Clona el repositorio:
   ```
   git clone https://github.com/tu_usuario/tu_repositorio.git
   ```
2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Ejecuta el programa principal:
   ```
   python main.py
   ```
4. Usa la interfaz gráfica para elegir estadísticas, filtrar Pokémon y realizar clustering.
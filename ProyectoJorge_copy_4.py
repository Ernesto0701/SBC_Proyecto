import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score  # Importación agregada
from kneed import KneeLocator
import matplotlib.pyplot as plt
import plotly.express as px


data = pd.read_csv("filtered_pokemons.csv")
team = [None] * 6  
visited = set() 
filtered_data = data  
stats = ['atk', 'speed']  
include_types = False 

def calculate_elbow_method(features_scaled, max_clusters=10):
    """
        Automatically calculates the optimal number of clusters using the Elbow Method (no graphing).
    """
    inertia = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=None)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
    
    # Automatically find the "elbow" using KneeLocator
    kneedle = KneeLocator(range(1, max_clusters + 1), inertia, curve="convex", direction="decreasing")
    return kneedle.knee or 3   # Default to 3 clusters or any other reasonable fallback

def perform_clustering(stats, filtered_data, include_types=False):
    features = filtered_data[stats].copy()
    if include_types:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        filtered_data['type2'] = filtered_data['type2'].fillna('None')
        types_encoded = encoder.fit_transform(filtered_data[['type1', 'type2']])
        types_df = pd.DataFrame(types_encoded, columns=encoder.get_feature_names_out(['type1', 'type2']))
        features = pd.concat([features.reset_index(drop=True), types_df.reset_index(drop=True)], axis=1)
    
    features = features.fillna(features.mean())
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    optimal_k = calculate_elbow_method(features_scaled)
    kmeans = KMeans(n_clusters=optimal_k, random_state=None)
    filtered_data['Cluster'] = kmeans.fit_predict(features_scaled)
    return filtered_data

def update_clustering():
    global filtered_data, stats, include_types
    filtered_data = perform_clustering(stats, filtered_data, include_types)
    messagebox.showinfo("Clustering Actualizado", "El clustering se ha realizado con éxito.")

# Filter Pokémon based on type1 or type2
def filter_by_type_window():
    def apply_filter():
        global filtered_data, include_types
        chosen_type = type_combobox.get()
        if chosen_type == "Todos":
            filtered_data = data
        else:
            filtered_data = data[(data['type1'] == chosen_type) | (data['type2'] == chosen_type)].copy()
        include_types = chosen_type != "Todos"
        update_clustering()
        filter_window.destroy()
        messagebox.showinfo("Filtrado", f"Se han filtrado los Pokémon por tipo: {chosen_type}")

    filter_window = tk.Toplevel(root)
    filter_window.title("Filtrar Pokémon por Tipo")
    tk.Label(filter_window, text="Seleccione un tipo de Pokémon:").pack(pady=5)
    type_combobox = ttk.Combobox(filter_window, values=['Todos'] + data['type1'].unique().tolist())
    type_combobox.set("Todos")
    type_combobox.pack(pady=5)
    tk.Button(filter_window, text="Aplicar Filtro", command=apply_filter).pack(pady=10)

def choose_stats_window():
    def apply_stats():
        global stats
        stats[0] = stat1_combobox.get()
        stats[1] = stat2_combobox.get()
        update_clustering()
        stats_window.destroy()
        messagebox.showinfo("Estadísticas Elegidas", f"Estadísticas seleccionadas: {stats[0]} y {stats[1]}")

    stats_window = tk.Toplevel(root)
    stats_window.title("Elegir Estadísticas")
    tk.Label(stats_window, text="Seleccione dos estadísticas para el clustering:").pack(pady=5)
    stat1_combobox = ttk.Combobox(stats_window, values=['hp', 'atk', 'def', 'spatk', 'spdef', 'speed'])
    stat1_combobox.set("atk")
    stat1_combobox.pack(pady=5)

    stat2_combobox = ttk.Combobox(stats_window, values=['hp', 'atk', 'def', 'spatk', 'spdef', 'speed'])
    stat2_combobox.set("speed")
    stat2_combobox.pack(pady=5)

    tk.Button(stats_window, text="Aplicar Estadísticas", command=apply_stats).pack(pady=10)

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def compare_clustering_algorithms(stats, filtered_data):
    """
    Compara el rendimiento de KMeans y DBSCAN utilizando Silhouette Score y muestra una gráfica.
    """
    features = filtered_data[stats].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.fillna(features.mean()))

    # KMeans
    optimal_k = calculate_elbow_method(features_scaled)
    kmeans = KMeans(n_clusters=optimal_k, random_state=None)
    kmeans_labels = kmeans.fit_predict(features_scaled)
    kmeans_silhouette = silhouette_score(features_scaled, kmeans_labels)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(features_scaled)

    # Calcular Silhouette Score para DBSCAN (solo si hay más de un cluster válido)
    if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
        dbscan_silhouette = silhouette_score(features_scaled, dbscan_labels)
    else:
        dbscan_silhouette = None  # Si no es válido, asignamos None

    # Preparar datos para la gráfica
    algorithms = ['KMeans', 'DBSCAN']
    silhouette_scores = [kmeans_silhouette, dbscan_silhouette]
    colors = ['skyblue', 'grey']  # Colores: uno para KMeans, otro para N/A (DBSCAN)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        algorithms, 
        [score if score is not None else 0 for score in silhouette_scores],  # Si es None, usar 0 para la barra
        color=colors, 
        alpha=0.7
    )

    # Añadir etiquetas a las barras
    for i, score in enumerate(silhouette_scores):
        if score is not None:
            plt.text(i, score + 0.02, f"{score:.3f}", ha='center', fontsize=10)
        else:
            plt.text(i, 0.02, "N/A", ha='center', fontsize=10, color='red')  # Mostrar N/A en rojo

    # Configurar la gráfica
    plt.title("Comparación de Rendimiento de Algoritmos de Clustering", fontsize=16)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.ylim(0, 1)  # Limitar el eje Y para mantener consistencia
    plt.xticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Mostrar métricas en consola
    print("\n--- Métricas Comparativas ---")
    print(f"KMeans Silhouette Score: {kmeans_silhouette:.3f}")
    print(f"DBSCAN Silhouette Score: {'N/A' if dbscan_silhouette is None else f'{dbscan_silhouette:.3f}'}")


def select_team_member_window():
    """
    Permite seleccionar un Pokémon para un espacio del equipo basado en los clusters y usando la función `select_pokemon_knn_extreme`.
    """
    if 'Cluster' not in filtered_data:
        messagebox.showerror("Error", "Primero debe realizar el clustering.")
        return

    def add_to_team():
        selected_pokemon = select_pokemon_knn_extreme(filtered_data, stats, visited)
        
        # Verificar si el resultado es None
        if selected_pokemon is None:
            messagebox.showinfo("Equipo Completo", "Todos los Pokémon ya han sido visitados.")
            team_window.destroy()
            return
        
        # Asignar Pokémon al equipo
        selected_space = space_combobox.current()
        team[selected_space] = selected_pokemon
        team_window.destroy()
        messagebox.showinfo("Equipo Actualizado", 
                            f"{selected_pokemon['name']} añadido al espacio {selected_space + 1}.")

    
    team_window = tk.Toplevel(root)
    team_window.title("Seleccionar Pokémon para el Equipo")

    tk.Label(team_window, text="Seleccione un espacio para el Pokémon:").pack(pady=5)
    space_combobox = ttk.Combobox(team_window, values=list(range(1, 7)), state="readonly")
    space_combobox.set(1)
    space_combobox.pack(pady=10)

    tk.Button(team_window, text="Añadir al Equipo", command=add_to_team).pack(pady=10)

# Función `select_pokemon_knn_extreme` (ya está en tu código, pero la integramos)
def select_pokemon_knn_extreme(filtered_data, stats, visited):
    """
    Selecciona el mejor Pokémon del dataset filtrado y clusterizado basándose en las estadísticas.
    """
    # Eliminar Pokémon ya visitados
    unvisited_data = filtered_data[~filtered_data['name'].isin(visited)].copy()
    
    if unvisited_data.empty:
        print("Todos los Pokémon ya fueron visitados. No hay más Pokémon disponibles.")
        return None
    
    # Calcular el puntaje basado en las estadísticas seleccionadas
    unvisited_data['score'] = 0.5 * unvisited_data[stats[0]] + 0.5 * unvisited_data[stats[1]]
    best_pokemon = unvisited_data.sort_values(by='score', ascending=False).iloc[0]
    print(f"Pokémon seleccionado: {best_pokemon['name']} (Cluster {best_pokemon['Cluster']})")
    visited.add(best_pokemon['name'])
    return best_pokemon

import matplotlib.pyplot as plt

def display_metrics_with_graph(stats, filtered_data):
    """
    Muestra métricas de rendimiento del clustering con una gráfica de distribución.
    """
    print("\n--- Métricas de Rendimiento del Clustering ---")
    
    # Calcular métricas
    features = filtered_data[stats].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.fillna(features.mean()))
    kmeans = KMeans(n_clusters=len(filtered_data['Cluster'].unique()), random_state=None)
    kmeans.fit(features_scaled)
    inertia = kmeans.inertia_

    if len(filtered_data['Cluster'].unique()) > 1:
        silhouette_avg = silhouette_score(features_scaled, filtered_data['Cluster'])
        silhouette_text = f"{silhouette_avg:.3f}"
    else:
        silhouette_avg = None
        silhouette_text = "No disponible"

    num_clusters = len(filtered_data['Cluster'].unique())
    cluster_counts = filtered_data['Cluster'].value_counts()

    # Crear gráfico
    plt.figure(figsize=(10, 6))
    cluster_counts.sort_index().plot(kind='bar', color='skyblue', alpha=0.7)
    plt.title("Distribución de Clusters", fontsize=16)
    plt.xlabel("Clusters", fontsize=12)
    plt.ylabel("Número de Elementos", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Añadir métricas como texto en la gráfica
    plt.figtext(0.15, 0.8, f"Inercia: {inertia:.2f}", fontsize=12, color='black')
    plt.figtext(0.15, 0.75, f"Puntaje de Silhouette: {silhouette_text}", fontsize=12, color='black')
    plt.figtext(0.15, 0.7, f"Número de Clusters: {num_clusters}", fontsize=12, color='black')

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()

def show_team():
    team_info = [
        f"Espacio {i + 1}: {pokemon['name']} (Cluster {pokemon['Cluster']})" 
        if pokemon is not None and not pokemon.empty 
        else f"Espacio {i + 1}: Vacío"
        for i, pokemon in enumerate(team)
    ]
    
    team_window = tk.Toplevel(root)
    team_window.title("Equipo Actual")
    team_listbox = tk.Listbox(team_window, height=6, width=40)
    for info in team_info:
        team_listbox.insert(tk.END, info)
    team_listbox.pack()

def handle_menu_choice():
    choice = menu_entry.get()
    if choice == '1':
        choose_stats_window()
    elif choice == '2':
        filter_by_type_window()
    elif choice == '3':
        show_team()
    elif choice == "4":
        # Mostrar métricas de clustering
        if len(stats) < 2:  # Verificar si se han seleccionado dos estadísticas
            messagebox.showerror("Error", "Primero debe seleccionar las estadísticas.")
        else:
            display_metrics_with_graph(stats, filtered_data)
    elif choice == "5":
        # Comparar algoritmos
        if len(stats) < 2:
            messagebox.showerror("Error", "Primero debe seleccionar las estadísticas.")
        else:
            compare_clustering_algorithms(stats, filtered_data)
    elif choice == '6':
        select_team_member_window()
    elif choice == '7':
        root.destroy()
    else:
        messagebox.showerror("Error", "Opción inválida. Intente de nuevo.")


# Interfaz gráfica principal
root = tk.Tk()
root.title("Gestión de Pokémon con Clustering")

# Menú principal
menu_label = tk.Label(root, text="--- Menú Principal ---", font=("Arial", 14))
menu_label.pack()
menu_text = tk.Label(root, text="1. Elegir estadísticas para clustering\n"
                                 "2. Filtrar Pokémon por tipo\n"
                                 "3. Mostrar equipo actual\n"
                                 "4. Mostrar métricas de clustering\n"
                                 "5. Comparar algoritmos de clustering\n"
                                 "6. Seleccionar un Pokémon para un espacio del equipo\n"
                                 "7. Salir")

menu_text.pack()

menu_entry = tk.Entry(root, width=10)
menu_entry.pack()
menu_button = tk.Button(root, text="Seleccionar Opción", command=handle_menu_choice)
menu_button.pack()

root.mainloop()

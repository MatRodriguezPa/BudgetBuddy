import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np 

platillos_df = pd.read_csv('C:/Users/ESTUDIANTE/Downloads/Algoritmo-20230928T140743Z-001/Algoritmo/AlgoritmoSugerencias/platillos_con_ingredientes.csv') 
stop_words_espanol = ["de", "la", "el", "los", "las", "en", "un", "una", "y", "o", "para", "por", "con", "al", "del", "su", "sus", "como", "más", "menos", "si", "no", "pero", "a", "ante", "bajo", "cabe", "contra", "de", "desde", "durante", "en", "entre", "hacia", "hasta", "mediante", "para", "por", "segun", "sin", "sobre", "tras", "y", "aunque", "conque", "con tal de que", "como", "para que", "pues", "ya que"]

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_espanol)
tfidf_matrix = tfidf_vectorizer.fit_transform(platillos_df['nombre_ingrediente'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

usuario_seleccion = ['Champiñones con puerros', 'Ensalada de champiñón y calabacín', 'Pollo marinado con salsa de limón y miel']  
usuario_seleccion1 = ['Lentejas', 'Macarrones con calabacin', 'Pollo marinado con salsa de limón y miel', 'Verduras a la plancha']  

# 1. Filtra los platillos seleccionados por el usuario
platillos_seleccionados = platillos_df[platillos_df['nombre_platillo'].isin(usuario_seleccion)]

# 2. Obtiene los índices de los platillos seleccionados
indices_seleccionados = platillos_seleccionados.index

# 3. Selecciona las filas correspondientes en la matriz TF-IDF
perfil_usuario_tfidf = tfidf_matrix[indices_seleccionados]

# 4. Calcula el promedio a lo largo del eje 0 (columnas) para obtener el perfil del usuario
perfil_usuario = perfil_usuario_tfidf.mean(axis=0)

# 5. Ajusta el perfil del usuario al mismo formato de la matriz TF-IDF
perfil_usuario = perfil_usuario.reshape(1, -1)

# 6. Convierte perfil_usuario y tfidf_matrix a numpy arrays
perfil_usuario = np.asarray(perfil_usuario)
tfidf_matrix = tfidf_matrix.toarray()

# 7. Calcula la similitud entre el perfil del usuario y todos los platillos
cosine_sim_perfil = linear_kernel(perfil_usuario, tfidf_matrix)

# 8. Enumera las similitudes y ordena las recomendaciones
sim_scores = list(enumerate(cosine_sim_perfil[0]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
platillos_similares = sim_scores[1:11]  # Recomienda los 10 platillos más similares (el primero es el propio platillo del usuario)

# Imprime las recomendaciones
for i, (indice, similitud) in enumerate(platillos_similares, start=1):
    print(f"Recomendación {i}: {platillos_df['nombre_platillo'].iloc[indice]} (Similitud: {similitud:.2f})")

import os
import pandas as pd

def load_csv_files(directory:str) -> dict:
  dfs:dict = {}  # Dictionnaire pour stocker les DataFrames

  for filename in os.listdir(directory):
    # Vérifier si le fichier est un fichier CSV
    if filename.endswith(".csv"):
      # Construire le chemin complet du fichier
      file_path = os.path.join(directory, filename)
      df = pd.read_csv(file_path)
      key_name:str = filename.split('.')[0]
      dfs[key_name] = df
      print(f"Chargement du fichier {filename}")

  # Retourner le dictionnaire contenant les DataFrames
  return dfs

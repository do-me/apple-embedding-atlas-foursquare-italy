# Apple Embedding-Atlas Foursquare Italy
Embedding atlas for Foursquare data in Italy with more than 3 million points!

Using Foursquare data with Minish static embeddings from this repo: https://huggingface.co/datasets/do-me/foursquare_places_100M

![](foursquare_italy_embs_1.png)
![](foursquare_italy_embs_2.png)

## 1) Data Preparation: Adding UMAP x and y cols  

This script gets the job done and took 17 minutes on my M3 Max, but could probably be improved a lot. 

```python
# uv pip install umap-learn duckdb pandas numpy

import duckdb
import numpy as np
import pandas as pd
import umap

# --- Configuration ---
input_parquet_file = 'foursquare_places_italy_embeddings.parquet'
output_parquet_file = 'foursquare_places_all_italy_embeddings_with_umap.parquet'
embeddings_column = 'embeddings'
tsne_dimension_1_column = 'x'
tsne_dimension_2_column = 'y'

# 1. Load just the embeddings using DuckDB (faster & memory efficient)
print(f"Reading embeddings from '{input_parquet_file}' with DuckDB...")
con = duckdb.connect()
query = f"SELECT {embeddings_column} FROM read_parquet('{input_parquet_file}')"
embedding_rows = con.execute(query).fetchall()
print("Embeddings read.")

# 2. Convert to NumPy
print("Converting to NumPy array...")
embeddings = np.array([list(e[0]) for e in embedding_rows])
print(f"Shape: {embeddings.shape}")

# 3. Run UMAP (2D projection)
print("Applying UMAP for dimensionality reduction...")
umap_model = umap.UMAP(n_components=2, n_neighbors=15, metric='euclidean')
umap_results = umap_model.fit_transform(embeddings)
print("UMAP complete.")

# 4. Read full data (efficient if done after projection)
print("Loading full data from Parquet...")
df = pd.read_parquet(input_parquet_file)

# 5. Attach UMAP results and drop embeddings
print("Attaching UMAP results...")
df[tsne_dimension_1_column] = umap_results[:, 0]
df[tsne_dimension_2_column] = umap_results[:, 1]
df = df.drop(columns=[embeddings_column, "geometry"], errors='ignore')  # Drop geometry if exists

# 6. Write back to Parquet
print(f"Writing updated data to '{output_parquet_file}'...")
df.to_parquet("without_geometry" + output_parquet_file, index=False)
print("Done.")
```

## 2) Use Embedding-Atlas

Install embedding-atlas

```shell
uv pip install embedding-atlas
```

Then run this command, avoiding embedding creation and passing precalculated umap x and y coords.
```shell
embedding-atlas foursquare_places_all_italy_names_with_umap_coords.parquet --duckdb server --no-embedding --text name --x x --y y
```

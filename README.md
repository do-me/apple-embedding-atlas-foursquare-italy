# Apple Embedding-Atlas Foursquare Italy
[Embedding atlas](https://github.com/apple/embedding-atlas) for Foursquare POI data in Italy with more than 3 million points!

**APP: https://do-me.github.io/apple-embedding-atlas-foursquare-italy/**

---
## Info

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

## Misc 

1. Note that this repo is almost maxing out what you can host for free on GitHub pages. The file limit on GitHub is 100Mb; the file size of the dataset used in this app is around 93Mb.
2. Thanks to Apple for open sourcing this amazing piece of software!
3. You can export all labels in JS with in your browser console with

```javascript
[...new Set(Array.from(document.querySelectorAll('g')).filter(g => g.querySelectorAll('g').length === 0 && g.querySelectorAll('text').length > 0).map(g => Array.from(g.querySelectorAll('text')).map(t => t.textContent).join('')).filter(label => label.trim() !== ''))];
```

<details>
 <summary>and you'll get 113 labels (click to expand)</summary>

```javascript
[
    "musica-music-musicale-logistica",
    "cooperativa-societ-sociale-coop",
    "ristorante-pizzeria-trattoria-osteria",
    "snc-sas-studio-bar",
    "immobiliare-agenzia-immobilier-immobiliari",
    "carrozzeria-autofficina-auto-officina",
    "mercato-architettura-universit-architetto",
    "gelateria-psicologa-poliambulatorio-psicoterapeuta",
    "fiori-fiore-fioreria-piante",
    "polisportiva-polizia-commissariato-municipale",
    "tabaccheria-tabacchi-ricevitoria-aula",
    "locanda-birrificio-sentiero-confartigianato",
    "scuola-elementare-nido-materna",
    "breakfast-bed-secondaria-grado",
    "tecnico-laboratorio-analisi-architetto",
    "viaggi-travel-konoba-tour",
    "food-self-drink-fast",
    "caff-caf-caffetteria-cafe",
    "cimitero-supermercato-supermercati-super",
    "parrucchieri-parrucchiere-parrucchiera-erboristeria",
    "pasticceria-pastificio-gelateria-caffetteria",
    "farmacia-dott-studio-dentistico",
    "treno-regionale-aula-flight",
    "obrt-ottica-rijeka-riello",
    "agricola-azienda-societ-agrituristica",
    "srl-italia-group-srls",
    "foto-video-fotografo-fotografico",
    "casa-hotel-apartments-villa",
    "informatica-confcommercio-rinaire-epoque",
    "ferramenta-tenuta-tende-sindacato",
    "dentistico-medico-studio-dott",
    "comune-palazzo-municipio-castello",
    "enoteca-vino-vinoteca-l'enoteca",
    "avv-avvocato-legale-madonna",
    "shop-store-fitness-lab",
    "trasporti-2000-transport-trasporto",
    "food-drink-fast-street",
    "musica-music-musicale-musicali",
    "maria-san-giuseppe-antonio",
    "carrozzeria-autocarrozzeria-carrosserie-officina",
    "tattoo-danza-dance-fitness",
    "associazione-group-culturale-formazione",
    "club-tennis-calcio-golf",
    "fisioterapia-fisioterapico-riabilitazione-fisioterapista",
    "circolo-arci-borgo-loco",
    "nido-asilo-d'infanzia-infanzia",
    "sagra-ponte-via-parco",
    "frizerski-salon-madonnina-bancomat",
    "libreria-pub-beach-cinema",
    "locanda-locarno-loca-loc",
    "tecnico-geom-geometra-odontotecnico",
    "costruzioni-meccaniche-edili-costruzione",
    "pizzeria-osteria-trattoria-pizza",
    "service-servizi-servizio-assistenza",
    "panificio-pastificio-pasticceria-pane",
    "polisportiva-polizia-municipale-polisportivo",
    "italy-italia-aurelia-mago",
    "estetico-estetica-eventi-centro",
    "self-lavanderia-service-wash",
    "calzature-alimentari-acconciature-caseificio",
    "art-galleria-toelettatura-d'arte",
    "sushi-dogane-ramen-kong",
    "biblioteca-piscina-bagno-beach",
    "consorzio-associazione-servizi-istituto",
    "piccolo-polifunzionale-piccola-dogana",
    "arredamenti-chiosco-arreda-arredo",
    "kebab-kebap-polispecialistico-istanbul",
    "impianti-elettrici-ricambi-autofficina",
    "cinese-giapponese-sushi-japanese",
    "erboristeria-pulizie-ippico-calzaturificio",
    "agricola-azienda-agrituristica-societ",
    "fermata-benessere-spa-centro",
    "sanpaolo-intesa-",
    "service-servizi-srl-medical",
    "analisi-laboratorio-odontotecnico-cliniche",
    "sportivo-sport-club-sportiva",
    "chen-cinese-zhou-wang",
    "agriturismo-podere-taverna-lido",
    "snc-sas-via-giuseppe",
    "gelateria-gelato-artigianale-yogurteria",
    "pescheria-pesce-peschiera-pesci",
    "supermercato-supermercati-super-supermarket",
    "obrt-bar-museo-birreria",
    "hair-beauty-bellezza-nails",
    "villa-giardino-giardini-hotel",
    "dott-ssa-ambulatorio-veterinario",
    "ufficio-office-postale-anagrafe",
    "cuore-delizie-sacro-magistrale",
    "funebri-onoranze-funebre-legno",
    "gioielleria-gioielli-orologeria-amedeo",
    "volontariato-cooperativo-credito-protezione",
    "officina-meccanica-cascina-infissi",
    "stazione-sud-onlus-fermata",
    "assicurazioni-tappezzeria-impianti-condominio",
    "avis-sede-comunale-fattoria",
    "gradnja-projekt-commerce-frimm",
    "ottica-obrt-rijeka-riello",
    "architettura-architetto-engineering-arch",
    "carabinieri-",
    "ottica-graziella-magazzini-ottico",
    "consulente-lavoro-scuderia-consulenza",
    "macelleria-traslochi-salumeria-calzaturificio",
    "viaggi-travel-turismo-tour",
    "notte-bistr-coworking-noce",
    "tipografia-nuova-new-nuovo",
    "parrocchia-parrocchiale-maddalena-santa",
    "farmacia-ssa-dott-parafarmacia",
    "autoriparazioni-autoricambi-emporio-autorimessa",
    "abbigliamento-edile-impresa-salumificio",
    "legale-studio-associato-avv",
    "breakfast-bed-room-rome",
    "lavoro-work-sangue-camera",
    "design-spazio-concept-creazioni"
]
```
</details>


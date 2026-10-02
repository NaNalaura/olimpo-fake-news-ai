#Extração de metadados dos textos
import requests, zipfile, io
import spacy
import pandas as pd
import glob, os
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

#------| Funçoes |-----------------------------------------------------------
def carregar_textos(pasta, label): 
    registros = []
    for caminho in sorted(glob.glob(os.path.join(pasta, "*.txt"))):
        repWith = ""
        if (label == 0):
            repWith = "t"
        id_noticia = os.path.basename(caminho).replace(".txt", repWith)
        with open(caminho, encoding="utf-8") as f:
            texto = f.read()
        registros.append({"id": id_noticia, "texto": texto, "label": label})
    return pd.DataFrame(registros)

def metaExtractionFromText(text, dependency = True):
    types_counter  = Counter()
    types_counter['quant_tokens'] = len(text)
    
    for sentenca in text.sents:
        for token in sentenca:
            morph_str = str(token.morph)
            if morph_str:
                types_counter[morph_str] += 1

            if token.pos_:
                types_counter[token.pos_] += 1
            if dependency:
                if token.dep_:
                    types_counter[token.dep_] +=1
    return types_counter


def metaExtractionFromDataset(dataframe):
    docs = list(npl.pipe(dataframe, batch_size=100, disable=["ner"]))
    acumulador = Counter()
    for doc in docs:
        acumulador.update(metaExtractionFromText(doc))
    return acumulador


#------| Carregando dados |-----------------------------------------------------------
url = "https://github.com/roneysco/Fake.br-Corpus/archive/refs/heads/master.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(".")  # cria a pasta Fake.br-Corpus-master/
os.listdir(".")
df_fake = carregar_textos("Fake.br-Corpus-master/full_texts/fake", label=1)
df_true = carregar_textos("Fake.br-Corpus-master/full_texts/true", label=0)
df_textos = pd.concat([df_fake, df_true], ignore_index=True)

npl = spacy.load("pt_core_news_sm")

metadado_fake =  metaExtractionFromDataset(df_fake['texto'])
metadado_true = metaExtractionFromDataset(df_true['texto'])
df_metadados_fake = pd.DataFrame(list(metadado_fake.items()), columns=["Metadado", "Quantidade"])
df_metadados_true = pd.DataFrame(list(metadado_true.items()), columns=["Metadado", "Quantidade"])
df_metadados_total = pd.merge(df_metadados_fake, df_metadados_true, on="Metadado", how="outer").fillna(0)

df_metadados_total.to_csv("metadados_comparativos.csv", index=False, encoding="utf-8")
print("Dados salvos em 'metadados_comparativos.csv'.")

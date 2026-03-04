import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from dataset import DataSet
from agent0 import agente


dados = DataSet(n=50) 

resultados = []

print(f"Iniciando avaliação de {len(dados)} notícias...")

for item in tqdm(dados):
    noticia = item["Noticia"]
    label_real = item["Label"]  # 0 = Verdadeira, 1 = Falsa
    
    estado_inicial = {
        "noticia": noticia,
        "veracidade": None,
        "tipo": None,
        "justificativa": None,
        "topicos_verdadeiros": [],
        "topicos_falsos": []
    }
    
    try:
        resposta = agente.invoke(estado_inicial)
        
        predicao_texto = resposta.get("veracidade", "Inconclusivo")

        if predicao_texto and isinstance(predicao_texto, str):
            pred_limpa = predicao_texto.strip().lower()
            if "verdadeira" in pred_limpa or "verdadeiro" in pred_limpa:
                label_pred = 0
            elif "falsa" in pred_limpa or "falso" in pred_limpa:
                label_pred = 1
            else:
                label_pred = -1
        else:
            label_pred = -1

        resultados.append({
            "texto_original": noticia,
            "label_real": label_real,
            "label_pred": label_pred,
            "predicao_texto": predicao_texto,
            "tipo_pred": resposta.get("tipo"),
            "justificativa": resposta.get("justificativa"),
            "topicos_verdadeiros": resposta.get("topicos_verdadeiros"),
            "topicos_falsos": resposta.get("topicos_falsos")
        })
        
    except Exception as e:
        print(f"Erro no processamento: {e}")
        resultados.append({
            "label_real": label_real,
            "label_pred": -1
        })

df_res = pd.DataFrame(resultados)

erros = df_res[df_res['label_pred'] == -1]
if not erros.empty:
    print(f"\n[AVISO] {len(erros)} notícias resultaram em erro ou resposta inconclusiva.")
    print("Exemplos de respostas inválidas:")
    print(erros[['predicao_texto']].head())

df_validos = df_res[df_res['label_pred'] != -1]

if df_validos.empty:
    print("\n[ERRO CRÍTICO] Nenhuma predição válida foi gerada. Verifique se a chave da API está correta ou se o modelo está respondendo.")
else:
    print("\n" + "="*40)
    print("RELATÓRIO DE DESEMPENHO")
    print("="*40)

    acc = accuracy_score(df_validos['label_real'], df_validos['label_pred'])
    print(f"Acurácia Global: {acc:.2%}\n")

    print("Relatório Detalhado:")
    print(classification_report(
        df_validos['label_real'], 
        df_validos['label_pred'], 
        target_names=['Verdadeiro (0)', 'Falso (1)'],
        labels=[0, 1] 
    ))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cm = confusion_matrix(df_validos['label_real'], df_validos['label_pred'], labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Pred: Verdadeiro', 'Pred: Falso'],
                yticklabels=['Real: Verdadeiro', 'Real: Falso'])
    axes[0].set_title("Matriz de Confusão")

    df_fakes_preditas = df_validos[df_validos['label_pred'] == 1]
    
    if not df_fakes_preditas.empty:
        if df_fakes_preditas['tipo_pred'].isnull().all():
             axes[1].text(0.5, 0.5, "Tipos não identificados (None)", ha='center')
        else:
            sns.countplot(y='tipo_pred', data=df_fakes_preditas, 
                          order=df_fakes_preditas['tipo_pred'].value_counts().index, 
                          palette='viridis', ax=axes[1])
            axes[1].set_title("Classificação dos Tipos de Desinformação")
            axes[1].set_xlabel("Quantidade")
            axes[1].set_ylabel("")
    else:
        axes[1].text(0.5, 0.5, "Nenhuma Fake News Detectada", ha='center')

    plt.tight_layout()
    plt.show()

    print("\n" + "="*40)
    print("ANÁLISE QUALITATIVA")
    print("="*40)

    ex_true = df_validos[(df_validos['label_real'] == 0) & (df_validos['label_pred'] == 0)].head(1)
    if not ex_true.empty:
        r = ex_true.iloc[0]
        print(f"\n[Exemplo Correto - Verdadeiro]")
        print(f"Notícia: {r['texto_original'][:200]}...")
        print(f"Resultado Agente: {r['predicao_texto']}")

    ex_fake = df_validos[(df_validos['label_real'] == 1) & (df_validos['label_pred'] == 1)].head(1)
    if not ex_fake.empty:
        r = ex_fake.iloc[0]
        print(f"\n[Exemplo Correto - Fake News ({r['tipo_pred']})]")
        print(f"Notícia: {r['texto_original'][:200]}...")
        print(f"Justificativa: {r['justificativa']}")

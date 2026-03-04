from dataset import DataSet
from typing import TypedDict, Optional, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import json
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)





# Estado

class Estado(TypedDict):
    noticia: str
    veracidade: Optional[str]
    tipo: Optional[str]
    justificativa: Optional[str]
    topicos_verdadeiros: Optional[List[str]]
    topicos_falsos: Optional[List[str]]


# Nós

def LerVerificar(state: Estado) -> Command[Literal["Classificador", "saida"]]:
    """
    Verifica a veracidade da notícia e,
    se falsa, separa tópicos verdadeiros e falsos.
    Decide o fluxo internamente.
    """

    noticia = state["noticia"]

    prompt = f"""
Você é um verificador de notícias.

Sua tarefa é ANALISAR APENAS o texto da notícia fornecida.

Tarefas:
1. Determinar se a notícia é "Verdadeira" ou "Falsa".
2. SE A NOTÍCIA FOR FALSA:
   - Extrair TRECHOS LITERAIS da própria notícia que sejam:
     a) claramente verdadeiros
     b) claramente falsos

REGRAS OBRIGATÓRIAS:
- Use SOMENTE o texto da notícia.
- NÃO explique, interprete ou reescreva.
- NÃO crie frases novas.
- Cada item nas listas DEVE ser uma cópia literal de um trecho da notícia.
- Os trechos devem aparecer exatamente como estão escritos no texto original.
- Se um trecho falso for a notícia inteira, repita a frase completa como tópico falso.
- Se não houver trecho claramente verdadeiro, retorne lista vazia.
- Se não for possível separar, retorne listas vazias.

Formato OBRIGATÓRIO (JSON válido):
{{
  "veracidade": "Verdadeira" | "Falsa",
  "topicos_verdadeiros": [
    "trecho literal copiado da notícia"
  ],
  "topicos_falsos": [
    "trecho literal copiado da notícia"
  ]
}}

Notícia:
{noticia}
"""

    resposta = llm.invoke(prompt)

    texto = resposta.content.strip()
    inicio = texto.find("{")
    fim = texto.rfind("}") + 1
    texto = texto[inicio:fim]

    try:
        dados_resp = json.loads(texto)
        veracidade = dados_resp["veracidade"]
        topicos_verdadeiros = dados_resp.get("topicos_verdadeiros", [])
        topicos_falsos = dados_resp.get("topicos_falsos", [])
    except Exception:
        veracidade = "Falsa"
        topicos_verdadeiros = []
        topicos_falsos = []

    if veracidade == "Falsa":
        goto = "Classificador"
    else:
        goto = "saida"

    return Command(
        goto=goto,
        update={
            "veracidade": veracidade,
            "topicos_verdadeiros": topicos_verdadeiros,
            "topicos_falsos": topicos_falsos
        }
    )


def Classificador(state: Estado) -> Command[Literal["saida"]]:
    """
    Classifica o tipo de desinformação e gera justificativa.
    """

    noticia = state["noticia"]

    prompt = f"""
Você é um classificador de tipos de desinformação.

A notícia abaixo JÁ FOI classificada como FALSA.
Você NÃO deve reavaliar a veracidade.

Sua tarefa tem DUAS PARTES, com REGRAS DIFERENTES.

────────────────────────────────
PARTE 1 — ESCOLHA DO TIPO (REGRAS ESTRITAS)
────────────────────────────────

1. Leia APENAS as definições abaixo.
2. Compare a notícia com essas definições.
3. Escolha O ÚNICO tipo cuja definição MELHOR se aplica.

REGRAS OBRIGATÓRIAS PARA O TIPO:
- Use EXCLUSIVAMENTE as definições fornecidas abaixo.
- NÃO utilize conhecimento externo para escolher o tipo.
- NÃO considere se a frase é verdadeira ou não.
- NÃO crie novos tipos.
- NÃO responda "não verificável".

Definições dos tipos:

- Sátira ou paródia: não tem intenção de prejudicar, mas tem potencial para enganar.
  Conteúdo humorístico ou fictício que pode ser confundido com notícia real.

- Conteúdo enganador: uso enganoso de informações reais para enquadrar uma questão ou indivíduo.
  A informação existe, mas é apresentada de forma distorcida.

- Conteúdo impostor: quando fontes genuínas são imitadas.
  Sites, perfis ou veículos falsos que se passam por fontes legítimas e conhecidas.

- Conteúdo fabricado: conteúdo novo, 100% falso, criado para ludibriar e prejudicar.
  Não tem nenhuma base factual — é inteiramente inventado.

- Conexão falsa: quando manchetes, ilustrações ou legendas não confirmam o conteúdo.
  O título ou imagem diz uma coisa, o texto diz outra.

- Contexto falso: quando conteúdo genuíno é compartilhado com informação de contexto falsa.
  O fato existiu, mas a data, local ou situação estão errados.

- Contexto manipulado: quando informação ou imagem genuína é manipulada para enganar.
  O conteúdo original foi alterado para mudar seu significado.

────────────────────────────────
PARTE 2 — JUSTIFICATIVA (REGRAS DIFERENTES)
────────────────────────────────

Agora, JUSTIFIQUE a escolha do tipo.

Para a JUSTIFICATIVA, você:
- PODE usar conhecimento factual geral.
- PODE explicar por que a afirmação é falsa no mundo real.
- PODE mencionar autoria inexistente, citações falsas, fatos históricos, etc.

MAS:
- A justificativa NÃO pode mudar o tipo escolhido.
- A justificativa DEVE ser compatível com a definição do tipo escolhida.

────────────────────────────────
Notícia falsa:
{noticia}

Formato OBRIGATÓRIO (JSON válido):
{{
  "tipo": "<um dos tipos da lista>",
  "justificativa": "Explique a escolha."
}}
"""

    resposta = llm.invoke(prompt)

    texto = resposta.content.strip()
    inicio = texto.find("{")
    fim = texto.rfind("}") + 1
    texto = texto[inicio:fim]

    try:
        dados_resp = json.loads(texto)
        tipo = dados_resp["tipo"]
        justificativa = dados_resp["justificativa"]
    except Exception:
        tipo = "não verificável"
        justificativa = "A resposta do modelo não pôde ser interpretada corretamente."

    return Command(
        goto="saida",
        update={
            "tipo": tipo,
            "justificativa": justificativa
        }
    )


def saida(state: Estado) -> dict:
    if state["veracidade"] == "Verdadeira":
        return {
            "resultado": "Verdadeira"
        }

    return {
        "resultado": "Falsa",
        "tipo": state.get("tipo"),
        "justificativa": state.get("justificativa"),
        "topicos_verdadeiros": state.get("topicos_verdadeiros", []),
        "topicos_falsos": state.get("topicos_falsos", [])
    }


# Montagem do grafo

graph = StateGraph(Estado)

graph.add_node("LerVerificar", LerVerificar)
graph.add_node("Classificador", Classificador)
graph.add_node("saida", saida)

graph.add_edge(START, "LerVerificar")
graph.add_edge("Classificador", "saida")
graph.add_edge("saida", END)

agente = graph.compile()
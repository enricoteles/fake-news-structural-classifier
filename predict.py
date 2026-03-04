from dataset import DataSet
from agent import agente

dados = DataSet()

print("Digite o número da notícia:")
i = int(input("> "))

noticia = dados[i]["Noticia"]

print(noticia)
print("\n")


estado_inicial = {
    "noticia": noticia,
    "veracidade": None,
    "tipo": None,
    "justificativa": None,
    "topicos_verdadeiros": None,
    "topicos_falsos": None
}

resultado = agente.invoke(estado_inicial)

print(resultado)

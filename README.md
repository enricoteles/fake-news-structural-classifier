# Fake News Structural Classifier

[English](#english) | [Português](#português)

---

<a name="english"></a>
## English

Fake news detection agent based on structural and semantic text analysis.

This repository contains a multi-stage LLM agent designed to identify and classify fake news in Portuguese. The system focuses on the analysis of text structure and semantics to identify misinformation patterns without relying on real-time internet searches for fact-checking.

### How it Works
The classification process is managed by a stateful graph (**LangGraph**) composed of specialized nodes. The agent receives processed texts directly from the chosen database through the `dataset.py` module.

1. **Read and Verify**: The agent receives the text and performs an initial analysis to determine veracity (True or False). If the news is identified as false, the agent already isolates literal snippets that prove the misinformation.
2. **Misinformation Classifier**: If the text is flagged as false, it enters a second stage where the agent compares the content against predefined structural categories (such as Fabricated Content, Imposter, or Manipulated Context).
3. **Justification and Type Identification**: Using internalized definitions (which in v1.0 were fetched via RAG from the site `focanasmidias.com.br`), the agent assigns the final category and generates a logical justification based on its factual knowledge and identified patterns.
4. **Output**:
    * **Predict (Individual)**: The single verification results in a structured JSON containing the analyzed news, veracity verdict, misinformation type, technical justification, and the separation of true and false snippets.
    * **Batch (Evaluation)**: Mass execution processes the complete dataset and presents the agent's performance relative to ground truth data, generating accuracy, recall metrics, and confusion matrices for statistical analysis.

### Project Evolution
* **v1.1 (Main Branch)**: Optimized Prompt-only architecture. Misinformation definitions are integrated directly into the agent's logic, eliminating external scraping and reducing latency and costs.
* **v1.0 (Branch Initial-version)**: Used RAG to fetch definitions from `focanasmidias.com.br` via `WebBaseLoader`. Data was vectorized with `OpenAIEmbeddings` and stored in `FAISS` to serve as the classifier's base.

### Results Analysis

#### Batch Evaluation (evaluate.py)
The model was validated with 100 news items from the **FakeRecogna2** dataset.

* **Global Accuracy**: 79.00%.
* **Confusion Matrix**: The system correctly identified 48 out of 50 fake news items, presenting only 2 false positives.
* **Recall (Fake)**: 0.96. Demonstrates high sensitivity for detecting misinformation, with an error rate of only 4% for fake news.




 
<img width="1735" height="822" alt="Captura de tela 2026-03-03 202349" src="https://github.com/user-attachments/assets/223f73a7-d324-41a4-ad06-67b2b8b0d649" />



<img width="975" height="262" alt="Captura de tela 2026-03-03 202440" src="https://github.com/user-attachments/assets/d30037e1-b8e4-44c3-8204-e673199cb746" />



#### Single News Verification (predict.py)
The agent generates a detailed JSON output for each analysis, classifying veracity, identifying the type, justifying the choice, and extracting relevant snippets for analysis.


<img width="1355" height="226" alt="Captura de tela 2026-03-05 111302" src="https://github.com/user-attachments/assets/3dd4076e-c341-483e-bdd0-e6c356221b2b" />



### Tools Used
* Python
* LangGraph & LangChain
* OpenAI GPT-4o-mini
* FAISS & Scikit-learn

### How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure your key in the `.env` file.
4. To test a single news item: `python predict.py`.
5. For batch metrics: `python evaluate.py`.

---

<a name="português"></a>
## Português

Agente de detecção de fake news baseado em análise estrutural e semântica de texto.

Este repositório contém um agente LLM de múltiplos estágios projetado para identificar e classificar fake news em português. O sistema foca na análise da estrutura e semântica do texto para identificar padrões de desinformação sem depender de buscas em tempo real na internet para checagem de fatos.

### Como Funciona
O processo de classificação é gerenciado por um grafo de estados (**LangGraph**) composto por nós especializados. O agente recebe os textos processados diretamente do banco de dados escolhido através do módulo `dataset.py`.

1. **Leitura e Verificação**: O agente recebe o texto e realiza uma análise inicial para determinar a veracidade (Verdadeiro ou Falso). Caso a notícia seja identificada como falsa, o agente já isola os trechos literais que comprovam a desinformação.
2. **Classificador de Desinformação**: Se o texto for sinalizado como falso, ele entra em um segundo estágio onde o agente compara o conteúdo com categorias estruturais predefinidas (como Conteúdo Fabricado, Impostor ou Contexto Manipulado).
3. **Justificativa e Identificação de Tipo**: Utilizando as definições internalizadas (que na v1.0 eram buscadas via RAG no site `focanasmidias.com.br`), o agente atribui a categoria definitiva e gera uma justificativa lógica baseada em seu conhecimento factual e nos padrões identificados.
4. **Saída**:
    * **Predict (Individual)**: A verificação unitária resulta em um JSON estruturado contendo a notícia analisada, o veredito de veracidade, o tipo de desinformação, a justificativa técnica e a separação de trechos verdadeiros e falsos.
    * **Lote (Batch/Evaluation)**: A execução em massa processa o dataset completo e apresenta o desempenho do agente em relação aos dados reais, gerando métricas de acurácia, recall e matrizes de confusão para análise estatística.

### Evolução do Projeto
* **v1.1 (Branch Main)**: Arquitetura Prompt-only otimizada. As definições de desinformação estão integradas diretamente na lógica do agente, eliminando o scraping externo e reduzindo latência e custos.
* **v1.0 (Branch Initial-version)**: Utilizava RAG para buscar definições no site `focanasmidias.com.br` via `WebBaseLoader`. Os dados eram vetorizados com `OpenAIEmbeddings` e armazenados no `FAISS` para servir de base para o classificador.

### Análise dos Resultados

#### Avaliação em Lote (evaluate.py)
O modelo foi validado com 100 notícias do dataset **FakeRecogna2**.

* **Acurácia Global**: 79.00%.
* **Matriz de Confusão**: O sistema identificou corretamente 48 das 50 notícias falsas, apresentando apenas 2 falsos positivos.
* **Recall (Falso)**: 0.96. Demonstra alta sensibilidade para detectar desinformação, com uma taxa de erro de apenas 4% para notícias falsas.

<img width="1735" height="822" alt="Captura de tela 2026-03-03 202349" src="https://github.com/user-attachments/assets/223f73a7-d324-41a4-ad06-67b2b8b0d649" />



<img width="975" height="262" alt="Captura de tela 2026-03-03 202440" src="https://github.com/user-attachments/assets/d30037e1-b8e4-44c3-8204-e673199cb746" />

#### Verificação de Notícia Única (predict.py)
O agente gera um output detalhado em JSON para cada análise, classificando a veracidade da notícia, identificando o tipo, justificando a escolha e extraindo trechos relevantes para a análise.

<img width="1355" height="226" alt="Captura de tela 2026-03-05 111302" src="https://github.com/user-attachments/assets/3dd4076e-c341-483e-bdd0-e6c356221b2b" />

### Ferramentas Utilizadas
* Python
* LangGraph & LangChain
* OpenAI GPT-4o-mini
* FAISS & Scikit-learn

### Como Executar
1. Clone o repositório.
2. Instale as dependências: `pip install -r requirements.txt`.
3. Configure sua chave no arquivo `.env`.
4. Para testar uma notícia: `python predict.py`.
5. Para métricas em lote: `python evaluate.py`.

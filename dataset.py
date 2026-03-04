from datasets import load_dataset, concatenate_datasets

def DataSet(n=50):
    dataset = load_dataset("recogna-nlp/fakerecogna2-extrativa")
    train = dataset["train"]

    verdadeiras = train.filter(lambda x: x["Label"] == 0).select(range(n))
    falsas = train.filter(lambda x: x["Label"] == 1).select(range(n))

    return concatenate_datasets([verdadeiras, falsas])

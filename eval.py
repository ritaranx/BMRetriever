from torch import Tensor

from typing import List, Dict, Union, Tuple
import numpy as np
import logging

from tqdm import trange

import torch

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers.file_utils import PaddingStrategy

import logging
import pathlib, os

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    return embedding
    

def get_detailed_instruct_query(task_description: str, query: str) -> str:
    return f'{task_description}\nQuery: {query}'

def get_detailed_instruct_passage(passage: str) -> str:
    return f'Represent this passage\npassage: {passage}'


logger = logging.getLogger(__name__)


class SentenceBERT:
    def __init__(self, model_path: Union[str, Tuple] = "BMRetriever/BMRetriever-7B", sep: str = " ", **kwargs):
        self.sep = sep
        self.task = 'Given a scientific claim, retrieve documents that support or refute the claim'
        
        self.model = AutoModel.from_pretrained(model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = 512
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side="left"

    def encode(self, input_texts, batch_size=8, **kwargs):
        # Tokenize the input texts
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for i in trange(0, len(input_texts), batch_size):
                input_text = input_texts[i: (i+batch_size)]
                batch_dict = self.tokenizer(
                        input_text, 
                        max_length=self.max_length-1, 
                        return_attention_mask=False, 
                        return_token_type_ids=False,
                        padding=PaddingStrategy.DO_NOT_PAD, 
                        truncation=True
                    )
                batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
                batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to("cuda")
                outputs = self.model(**batch_dict)
                embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim = 0)
        print(embeddings.shape)
        return embeddings


    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        queries = [get_detailed_instruct_query(self.task, query) for query in queries]
        return self.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        sentences = [get_detailed_instruct_passage(passage) for passage in sentences]
        return self.encode(sentences, batch_size=batch_size, **kwargs)



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


#### Load the SBERT model and retrieve using cosine-similarity
model = DRES(SentenceBERT(model_path="BMRetriever/BMRetriever-1B"), batch_size=32)
retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity #cos_sim
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

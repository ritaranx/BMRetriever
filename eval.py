import os
import pathlib
import logging
from datetime import timedelta
from typing import List, Dict, Union, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist
from tqdm import trange
from transformers import AutoTokenizer, AutoModel
from transformers.file_utils import PaddingStrategy

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

# Initialize distributed process group and set the current CUDA device
dist.init_process_group(timeout=timedelta(minutes=60))
torch.cuda.set_device(dist.get_rank())

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
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

class SentenceBERT:
    def __init__(self, model_path: Union[str, Tuple] = "BMRetriever/BMRetriever-7B", sep: str = " ", dataset="", **kwargs):
        self.sep = sep
        self.task = 'Given a scientific claim, retrieve documents that support or refute the claim'
        self.dataset = dataset
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32, cache_dir="/localscratch/yueyu/cache")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()
        self.max_length = 512
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    @torch.no_grad()
    def encode(self, input_texts: List[str], batch_size: int = 8, **kwargs) -> Tensor:
        embeddings = []
        self.model.eval()
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
            with torch.cuda.amp.autocast():
                batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
                batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to("cuda")
                outputs = self.model(**batch_dict)
                embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Tensor:
        queries = [get_detailed_instruct_query(self.task, query) for query in queries]
        embeddings = self.encode(queries, batch_size=batch_size, **kwargs)
        return embeddings

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> Tensor:
        if isinstance(corpus, dict):
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        sentences = [get_detailed_instruct_passage(passage) for passage in sentences]
        embeddings = self.encode(sentences, batch_size=batch_size, **kwargs)
        return embeddings

# Download and load dataset
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# Load corpus, queries, and qrels
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

model = DRES(SentenceBERT(model_path="BMRetriever/BMRetriever-1B"), batch_size=64)
retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus, queries)

# Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

# Dense Passage Retrieval 稠密段落检索

Dense Passage Retrieval (`DPR`) - is a set of tools and models for state-of-the-art open-domain Q&A research.
它是基于以下论文实现的代码。
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih. [Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/abs/2004.04906) Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, 2020.

If you find this work useful, please cite the following paper:

```
@inproceedings{karpukhin-etal-2020-dense,
    title = "Dense Passage Retrieval for Open-Domain Question Answering",
    author = "Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.550",
    doi = "10.18653/v1/2020.emnlp-main.550",
    pages = "6769--6781",
}
```
如果你对基于我们的模型checkpoint重现论文中的实验结果感兴趣（也就是说，不想从头开始训练编码器），你可以考虑使用[Pyserini toolkit]（https://github.com/castorini/pyserini/blob/master/docs/experiments-dpr.md），它通过`pip`将实验很好地打包。
他们的工具包也报告了更高的BM25和混合分数。


## 特点
1. 密集检索器模型是基于双编码器结构的。
2. Extractive Q&A reader&ranker joint model inspired by [this](https://arxiv.org/abs/1911.03868) paper.
3. 相关的数据预处理和后处理工具。
4. 推理时间逻辑的密集检索器组件是基于FAISS索引的。

## New (March 2021) release
DPR代码库进行了升级，增加了一些增强功能和新模型。
主要变化。
1. [Hydra](https://hydra.cc/)基于所有命令行工具的配置，数据加载器除外（即将转换）。
2. 可插入的数据处理层，支持自定义数据集
3. 新的检索模型checkpoint具有更好的性能。

## New (March 2021) retrieval model
现在提供了一个只在NQ数据集上训练的新的双编码器模型：一个新的checkpoint、训练数据、检索结果和维基百科嵌入。
它是在原始的DPR NQ训练集和它的版本上训练的，其中困难负样本词是使用DPR索引本身使用以前的NQcheckpoint挖掘出来的。
使用这个新的训练数据与我们原来的NQ训练数据相结合，从头开始训练一个双编码器模型。这个训练方案给了一个很好的检索性能提升。

在NQ测试集（3610道题）上，新旧top-k文档的检索精度对比。

| Top-k passages        | Original DPR NQ model           | New DPR model  |
| ------------- |:-------------:| -----:|
| 1      | 45.87 | 52.47 |
| 5      | 68.14      |   72.24 |
| 20  | 79.97      |    81.33 |
| 100  | 85.87      |    87.29 |

新模型的可下载资源名称（见下文如何使用download_data脚本）。

Checkpoint: checkpoint.retriever.single-adv-hn.nq.bert-base-encoder

New training data: data.retriever.nq-adv-hn-train

Retriever resutls for NQ test set: data.retriever_results.nq.single-adv-hn.test

Wikipedia embeddings: data.retriever_results.nq.single-adv-hn.wikipedia_passages


## 安装

从源头安装。推荐使用Python的虚拟环境或Conda环境。

```bash
git clone git@github.com:facebookresearch/DPR.git
cd DPR
pip install .
```

DPR在Python 3.6+和PyTorch 1.2.0+上测试。
DPR依靠第三方库来实现编码器代码。
它目前支持Huggingface（版本<=3.1.0）BERT、Pytext BERT和Fairseq RoBERTa编码器模型。
由于tokenization过程的通用性，DPR目前使用Huggingfacetokenization器。所以Huggingface是唯一需要依赖的，Pytext和Fairseq是可选的。
如果你想使用这些编码器，请单独安装它们。



## Resources & Data formats
首先，你需要为检索器或阅读器训练准备数据。
每个DPR组件都有自己的输入/输出数据格式。
你可以看到下面的格式描述。
DPR提供NQ和Trivia预处理的数据集（和模型checkpoint），使用我们的dpr/data/download_data.py工具从云端下载。人们需要指定要下载的资源名称。运行 "python data/download_data.py "来查看所有选项。


```bash
python data/download_data.py \
	--resource {key from download_data.py's RESOURCES_MAP}  \
	[optional --output_dir {your location}]
```
资源名称的匹配是基于前缀的。因此，如果你需要下载所有的数据资源，只需使用 --resource data。

## Retriever input data format
Retriever训练数据的默认数据格式是JSON。
它包含了每个问题的2种类型的负样本段落，以及正样本段落和一些附加信息。

```
[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"text": "...."
	}],
	"negative_ctxs": ["..."],
	"hard_negative_ctxs": ["..."]
  },
  ...
]
```
negative_ctxs和hard_negative_ctxs的元素结构与positive_ctxs完全相同。
可供下载的预处理数据还包含一些额外的属性，这些属性对模型的修改可能是有用的（如每个段落的bm25得分）。不过，它们目前还没有被DPR使用。

你可以通过使用'data.retriever.nq'键的前缀下载论文中使用的准备好的NQ数据集。只有开发和训练子集是以这种格式提供的。
我们还为所有训练/dev/测试子集提供了只有问题和答案的CSV数据文件。这些文件用于模型评估，因为我们的NQ预处理步骤丢失了一部分原始样本集。
使用'data.retriever.qas.*'资源键来获得相应的评估集。


```bash
python data/download_data.py
	--resource data.retriever
	[optional --output_dir {your location}]
```

## DPR data formats and custom processing 
人们可以通过继承DPR在dpr/data/{biencoder|retriever|reader}_data.py文件中的数据集类并实现load_data()和__getitem__()方法来使用自己的数据格式和自定义数据解析和加载逻辑。参见[DPR hydra configuration](https://github.com/facebookresearch/DPR/blob/master/conf/README.md)说明。


## Retriever training
检索器的训练质量取决于它的有效批次大小。论文中报道的那个使用了8 x 32GB的GPU。
为了在一台机器上开始训练。

```bash
python train_dense_encoder.py \
train_datasets=[list of train datasets, comma separated without spaces] \
dev_datasets=[list of dev datasets, comma separated without spaces] \
train=biencoder_local \
output_dir={path to checkpoints dir}
```

Example for NQ dataset

```bash
python train_dense_encoder.py \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_local \
output_dir={path to checkpoints dir}
```
DPR默认使用HuggingFace BERT-base作为编码器。其他就绪的选项包括Fairseq的ROBERTA和Pytext BERT模型。
人们可以通过改变编码器配置文件（conf/encoder/hf_bert.yaml）或在conf/encoder dir中提供一个新的配置文件并通过encoder={新文件名}命令行参数启用来选择它们。

注意。
- 如果你想使用pytext BERT或fairseq roberta，你需要下载预训练的权重并指定encoder.pretrained_file参数。为RoBERTa模型的'pretrained.fairseq.roberta-base'资源前缀指定下载文件的dir位置，或者为pytext BERT指定文件路径（资源名称'pretrained.pytext.bert-base.model'）。
- 验证和checkpoint的保存根据train.eval_per_epoch参数值进行。
- 除了指定的训练次数（train.num_train_epochs 配置参数），没有停止条件。
- 每次评估都会保存一个模型checkpoint。
- 最佳checkpoint被记录在训练过程的输出中。
- 用于双编码器训练的常规NLL分类损失验证可以用平均排名评估代替。它从输入数据段落池中聚合段落和问题向量，对这些表示进行大的相似性矩阵计算，然后对每个问题的glod段落的排名进行平均。我们发现这个指标与最终的检索性能和nll分类损失更加相关。但是请注意，这种平均排名验证在DistributedDataParallel和DataParallel PyTorch模式下的工作方式不同。参见train.val_av_rank_*参数集以启用该模式并修改其设置。

请看下面的 "最佳超参数设置 "一节，作为e2e的例子，介绍我们的最佳设置。

## Retriever inference
为静态文档数据集生成表示向量是一个高度可并行的过程，如果在单个GPU上计算，可能需要几天时间。你可能想使用多个可用的GPU服务器，在每个服务器上独立运行脚本并指定它们自己的分片。

```bash
python generate_dense_embeddings.py \
	model_file={path to biencoder checkpoint} \
	ctx_src={name of the passages resource, set to dpr_wiki to use our original wikipedia split} \
	shard_id={shard_num, 0-based} num_shards={total number of shards} \
	out_file={result files location + name PREFX}	
```
ctx_src参数的资源名称 
或只是conf/ctx_sources/default_sources.yaml文件中的资源名称。

注意：与训练模式相比，你可以在这里使用大得多的批次大小。例如，为2个GPU(16GB)的服务器设置batch_size 128应该可以正常工作。
你可以使用资源键'data.retriever_results.nq.single.wikipedia_passages'从我们的原始模型（在NQ数据集上训练）下载已经生成的维基百科嵌入文件。
新的更好的模型的嵌入资源名称为'data.retriever_results.nq.single-adv-hn.wikipedia_passages'

我们一般在50个2-gpu节点上使用以下参数： batch_size=128 shard_id=0 num_shards=50



## 检索器对整个文件集进行验证。

```bash

python dense_retriever.py \
	model_file={path to a checkpoint downloaded from our download_data.py as 'checkpoint.retriever.single.nq.bert-base-encoder'} \
	qa_dataset={the name os the test source} \
	ctx_datatsets=[{list of passage sources's names, comma separated without spaces}] \
	encoded_ctx_files=[{list of encoded document files glob expression, comma separated without spaces}] \
	out_file={path to output json file with results} 
	
```
例如，如果你生成的嵌入为两个段落设置为~/myproject/embeddings_passages1/wiki_passages_*和~/myproject/embeddings_passages2/wiki_passages_*文件，并想在NQ数据集上评估。

```bash
python dense_retriever.py \
	model_file={path to a checkpoint file} \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki] \
	encoded_ctx_files=[\"~/myproject/embeddings_passages1/wiki_passages_*\",\"~/myproject/embeddings_passages2/wiki_passages_*\"] \
	out_file={path to output json file with results} 
```

该工具将检索到的结果写入指定的out_file，用于后续的阅读器模型训练。
它是一个json文件，格式如下。

```
[
    {
        "question": "...",
        "answers": ["...", "...", ... ],
        "ctxs": [
            {
                "id": "...", # passage id from database tsv file
                "title": "",
                "text": "....",
                "score": "...",  # retriever score
                "has_answer": true|false
     },
]
```
结果按其相似度得分排序，从最相关到最不相关。

默认情况下，dense_retriever使用穷举式搜索过程，但你可以选择使用有损索引类型。
我们提供HNSW和HNSW_SQ索引选项。
通过indexer=hnsw或indexer=hnsw_sq命令行参数启用它们。
请注意，从研究的角度来看，使用这种索引可能是无用的，因为它们的快速检索过程是以更长的索引时间和更高的RAM使用率为代价的。
所提供的相似性分数是默认情况下的穷举搜索（indexer=flat）的点积，以及在HNSW索引的情况下修改过的表示空间中的L2距离。


## Reader model training
```bash
python train_extractive_reader.py \
	encoder.sequence_length=350 \
	train_files={path to the retriever train set results file} \
	dev_files={path to the retriever dev set results file}  \
	output_dir={path to output dir}
```

默认的超参数是为8个gpus的单节点设置的。
根据需要在conf/train/extractive_reader_default.yaml和conf/extractive_reader_train_cfg.yaml cpnfiguration文件中进行修改，或者从命令行覆盖特定参数。
第一次运行将对train_files和dev_files进行预处理，并将其转换为同一位置的序列化的.pkl文件集，并在随后的所有运行中使用它们。

注意。
- 如果你想使用pytext BERT或fairseq roberta，你需要下载预训练的权重并指定encoder.pretrained_file参数。为RoBERTa模型的'pretrained.fairseq.roberta-base'资源前缀指定下载文件的dir位置，或者为pytext BERT指定文件路径（资源名称'pretrained.pytext.bert-base.model'）。
- 阅读器训练pipeline在每个train.eval_step批次进行模型验证
- 像双编码器一样，它在每次验证时都会保存模型checkpoint
- 和双编码器一样，除了指定的训练次数，没有停止条件。
- 和双编码器一样，没有最佳checkpoint的选择逻辑，所以需要根据训练过程输出中记录的 dev set 验证性能来选择。
- 我们目前的代码只计算精确匹配指标。


## Reader model inference
为了进行推理，运行`train_reader.py`而不指定`train_files`。确保指定`model_file`与checkpoint的路径，`passages_per_question_predict`与每个问题的段落数（在保存预测文件时使用），以及`eval_top_docs`与顶级段落阈值列表，从中选择问题的答案跨度（将被打印为日志）。命令行的例子如下。

```bash
python train_extractive_reader.py \
  prediction_results_file={path to a file to write the results to} \
  eval_top_docs=[10,20,40,50,80,100] \
  dev_files={path to the retriever results file to evaluate} \
  model_file= {path to the reader checkpoint} \
  train.dev_batch_size=80 \
  passages_per_question_predict=100 \
  encoder.sequence_length=350
```

## Distributed training
使用Pytorch的分布式训练启动器工具。

```bash
python -m torch.distributed.launch \
	--nproc_per_node={WORLD_SIZE}  {non distributed scipt name & parameters}
```
Note:
- 在分布式模式(DistributedDataParallel)中，所有与批次大小有关的参数都是为每个gpu指定的，而在DataParallel(单节点-多gpu)模式中，则是为所有可用的gpu指定。

## Best hyperparameter settings

e2e的例子，对NQ数据集的最佳设置。

### 1. 下载所有检索器训练和验证数据。

```bash
python data/download_data.py --resource data.wikipedia_split.psgs_w100
python data/download_data.py --resource data.retriever.nq
python data/download_data.py --resource data.retriever.qas.nq
```

### 2. Biencoder(Retriever) training in the single set mode.

我们在一台8个GPU x 32 GB的服务器上使用分布式训练模式

```bash
python -m torch.distributed.launch --nproc_per_node=8
train_dense_encoder.py \
train=biencoder_nq \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_nq \
output_dir={your output dir}
```

新的模型训练结合了两个NQ数据集。

```bash
python -m torch.distributed.launch --nproc_per_node=8
train_dense_encoder.py \
train=biencoder_nq \
train_datasets=[nq_train,nq_train_hn1] \
dev_datasets=[nq_dev] \
train=biencoder_nq \
output_dir={your output dir}
```
这需要大约一天的时间来完成40个epochs的训练。它在第30个epoch时切换到Average Rank验证，最后应该是25左右或更少。
双编码器的最佳checkpoint通常是最后一个，但如果你在epoch ~ 25之后采取任何checkpoint，应该不会有太大差别。

### 3. Generate embeddings for Wikipedia.
只需使用 "Generating representations for large documents set "的说明。在50个2个GPU服务器上产生21百万个段落表示向量大约需要40分钟。

### 4. 评估检索精度，并为每个训练/dev/测试数据集生成顶级段落结果。

```bash

python dense_retriever.py \
	model_file={path to the best checkpoint or use our proivded checkpoints (Resource names like checkpoint.retriever.*)  } \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki] \
	encoded_ctx_files=["{glob expression for generated embedding files}"] \
	out_file={path to the output file}
```
根据可用的GPU数量调整batch_size，64-128应该适用于2个GPU服务器。

### 5. Reader training
我们使用单台8GPU x 32 GB的服务器为大型数据集训练了阅读器模型。所有的默认参数都已经设置为我们的最佳NQ设置。
请同时下载data.gold_passages_info.nq_train和data.gold_passages_info.nq_dev资源用于NQ数据集--它们在为NQ阅读器训练预处理数据时用于特殊的NQ启发式方法。如果你已经在没有指定gold_passages_src和gold_passages_src_dev的情况下在NQ数据上运行了读码器训练，请删除相应的.pkl文件，以便重新生成它们。

```bash
python train_extractive_reader.py \
	encoder.sequence_length=350 \
	train_files={path to the retriever train set results file} \
	dev_files={path to the retriever dev set results file}  \
	gold_passages_src={path to data.gold_passages_info.nq_train file} \
	gold_passages_src_dev={path to data.gold_passages_info.nq_dev file} \
	output_dir={path to output dir}
```
我们发现，使用上述学习率在静态时间表下效果最好，所以需要根据评估性能动态手动停止训练。
我们的最佳结果是在16-18个训练epoch或大约6万个模型更新后实现的。

我们为e2e pipeline提供了NQ数据集的所有输入和中间结果，以及Trivia的大部分类似资源。

## Misc.
- TREC验证需要基于regexp的匹配。我们只支持regexp模式下的retriever验证。参见--match参数选项。
- WebQ验证需要实体归一化，目前还不包括这个。

## License
DPR is CC-BY-NC 4.0 licensed as of now.

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import multiprocessing
from transformers import BertConfig, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets, load_from_disk
import os
from collections import Counter
from itertools import chain
import torch

class Cifar10():
    def __init__(self, config=None) -> None:
        transform_argumented = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.batch_size = 64
        self.num_classes = 10

        # train_dataset = torchvision.datasets.CIFAR10('datasets/cifar10', train=True, download=True, transform=transform_argumented)
        train_dataset = torchvision.datasets.CIFAR10('datasets/cifar10', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # test_dataset = torchvision.datasets.CIFAR10('datasets/cifar10', train=False, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10('datasets/cifar10', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

class Mnist():
    def __init__(self, config=None) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.batch_size = 256
        self.num_classes = 10

        train_dataset = torchvision.datasets.MNIST('datasets/mnist', train=True, download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = torchvision.datasets.MNIST('datasets/mnist', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

class Wikitext():
    def group_texts(self, examples):
        block_size = self.block_size

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2

        raw_datasets = load_dataset('wikitext', config.dataset_name)
        tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)
        lm_dataset.save_to_disk(path)
        return lm_dataset

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        path = os.path.join(config.dataset_cache[config.dataset_name], str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class IMDB():
    def group_texts(self, examples):
        block_size = self.block_size
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys() if k != 'label'}
        total_length = len(concatenated_examples[list(examples.keys())[1]])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def __init__(self, config) -> None:
        self.block_size = config.seq_len
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.batch_size = config.batch_size

        raw_datasets = load_dataset('imdb')
        tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text'], padding='max_length', truncation=True), batched=True, num_proc=16, remove_columns=["text"])

        path = os.path.join(config.dataset_cache[config.dataset_name], str(self.block_size))
        if not config.preprocessed:
            lm_datasets = tokenized_datasets.map(self.group_texts, batched=True, remove_columns=['label'])
            lm_datasets.save_to_disk(path)
        lm_datasets = load_from_disk(path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.lm_train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.lm_val_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)
        pass

class AGNews():
    def group_texts(self, examples):
        block_size = self.block_size
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys() if k != 'label'}
        total_length = len(concatenated_examples[list(examples.keys())[1]])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def lt_dataset(self, tokenized_datasets, tokenizer, ratio=.3):
        all_ids = [sample['input_ids'] for sample in tokenized_datasets['train']]
        concat_ids = list(chain(*all_ids))
        freqs = Counter(concat_ids)

        train_freq = []
        for sample in tokenized_datasets['train']:
            freq = [freqs[w] for w in sample['input_ids'] if w not in tokenizer.all_special_ids]
            train_freq.append(sum(freq) / len(freq))
        _, tail_indices = torch.topk(torch.tensor(train_freq), k=int(ratio*len(train_freq)), largest=False)
        lt_train = Dataset.from_dict(tokenized_datasets['train'][tail_indices])
        lt_train.set_format("torch")
        
        test_freq = []
        for sample in tokenized_datasets['test']:
            freq = [freqs[w] for w in sample['input_ids'] if w not in tokenizer.all_special_ids]
            test_freq.append(sum(freq) / len(freq))
        _, tail_indices = torch.topk(torch.tensor(test_freq), k=int(ratio*len(test_freq)), largest=False)
        lt_test = Dataset.from_dict(tokenized_datasets['test'][tail_indices])
        lt_test.set_format("torch")

        return lt_train, lt_test

    def __init__(self, config) -> None:
        self.block_size = config.seq_len
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.batch_size = config.batch_size

        raw_datasets = load_dataset('ag_news', split=['train[:20%]', 'test[:50%]'])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train', 'test'], raw_datasets)})
        tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text'], padding='max_length', truncation=True), batched=True, num_proc=16, remove_columns=["text"])

        path = os.path.join(config.dataset_cache[config.dataset_name], str(self.block_size))
        if not config.preprocessed:
            lm_datasets = tokenized_datasets.map(self.group_texts, batched=True, remove_columns=['label'])
            lm_datasets.save_to_disk(path)
        lm_datasets = load_from_disk(path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.lm_train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.lm_val_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

        tokenized_datasets.set_format("torch")
        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)

        lt_train, lt_test = self.lt_dataset(tokenized_datasets, self.tokenizer)
        self.lt_train_loader = DataLoader(lt_train, batch_size=self.batch_size, shuffle=True)
        self.lt_test_loader = DataLoader(lt_test, batch_size=self.batch_size, shuffle=False)
        pass

if __name__ == "__main__":
    config = BertConfig.from_json_file('config/moe.json')
    dataset = Wikitext(config)
    # agnews = AGNews(config)
    # cifar10 = Cifar10()
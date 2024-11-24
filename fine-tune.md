## Together.AI file

```shell
together files upload paired_dataset/paired_dataset_gpt_train.jsonl
Uploading file paired_dataset_gpt_train.jsonl: 100%|█
{
    "id": "file-6341ca37-f6f2-4849-a367-0a0ab1f8a7f7",
    "object": "file",
    "created_at": 1732480355,
    "purpose": "fine-tune",
    "filename": "paired_dataset_gpt_train.jsonl",
    "bytes": 0,
    "line_count": 0,
    "processed": false,
    "FileType": "jsonl"
}
```

```shell
together files upload paired_dataset/paired_dataset_gpt_validation.jsonl 
Uploading file paired_dataset_gpt_validation.jsonl: 1Uploading file paired_dataset_gpt_validation.jsonl: 1.99MB [00:01, 1.35MB/s]
{
    "id": "file-43814dd7-8875-493c-9b53-209d91f62149",
    "object": "file",
    "created_at": 1732481280,
    "purpose": "fine-tune",
    "filename": "paired_dataset_gpt_validation.jsonl",
    "bytes": 0,
    "line_count": 0,
    "processed": false,
    "FileType": "jsonl"
}
```


## Experiments

### Full
```shell
FinetuneRequest(
    training_file='file-6341ca37-f6f2-4849-a367-0a0ab
1f8a7f7',
    validation_file='',
    model='meta-llama/Meta-Llama-3.1-8B-Instruct-Refe
rence',
    n_epochs=1,
    learning_rate=1e-05,
    lr_scheduler=FinetuneLRScheduler(
        lr_scheduler_type='linear',
        lr_scheduler_args=FinetuneLinearLRSchedulerAr
gs(
            min_lr_ratio=0.0
        )
    ),
    warmup_ratio=0.0,
    max_grad_norm=1.0,
    weight_decay=0.0,
    n_checkpoints=1,
    n_evals=0,
    batch_size=24,
    suffix=None,
    wandb_key=None,
    training_type=FullTrainingType(type='Full'),
    train_on_inputs='auto'
)
Successfully submitted a fine-tuning job 
ft-0f4e79c4-7f15-48a3-8a2d-0ff88fc6f145 at 
11/24/2024, 15:39:29
```



### Lora
```shell
FinetuneRequest(
    training_file='file-6341ca37-f6f2-4849-a367-0a0ab
1f8a7f7',
    validation_file='',
    model='meta-llama/Meta-Llama-3.1-8B-Instruct-Refe
rence',
    n_epochs=1,
    learning_rate=1e-05,
    lr_scheduler=FinetuneLRScheduler(
        lr_scheduler_type='linear',
        lr_scheduler_args=FinetuneLinearLRSchedulerAr
gs(
            min_lr_ratio=0.0
        )
    ),
    warmup_ratio=0.0,
    max_grad_norm=1.0,
    weight_decay=0.0,
    n_checkpoints=1,
    n_evals=0,
    batch_size=32,
    suffix=None,
    wandb_key=None,
    training_type=LoRATrainingType(
        type='Lora',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_trainable_modules='all-linear'
    ),
    train_on_inputs='auto'
)
Successfully submitted a fine-tuning job 
ft-848e396d-da72-4405-82ed-21f24469f9b7 at 
11/24/2024, 15:42:41
```
```shell
FinetuneRequest(
    training_file='file-6341ca37-f6f2-4849-a367-0a0ab
1f8a7f7',
    validation_file='file-43814dd7-8875-493c-9b53-209
d91f62149',
    model='meta-llama/Meta-Llama-3.1-8B-Instruct-Refe
rence',
    n_epochs=1,
    learning_rate=1e-05,
    lr_scheduler=FinetuneLRScheduler(
        lr_scheduler_type='linear',
        lr_scheduler_args=FinetuneLinearLRSchedulerAr
gs(
            min_lr_ratio=0.0
        )
    ),
    warmup_ratio=0.0,
    max_grad_norm=1.0,
    weight_decay=0.0,
    n_checkpoints=1,
    n_evals=4,
    batch_size=32,
    suffix=None,
    wandb_key=None,
    training_type=LoRATrainingType(
        type='Lora',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_trainable_modules='all-linear'
    ),
    train_on_inputs='auto'
)
Successfully submitted a fine-tuning job 
ft-9cd628c8-9f62-4738-a52b-1fdf45aae37c at 
11/24/2024, 16:03:04
```
from multiprocessing import cpu_count


global_args = {
    "labels_list": None,
    "output_dir": "./models/baseline",
    "best_model_dir": "./models/baseline/best_model",
    "cache_dir": "cache_dir/",
    "config": {},
    "fp16": True, #False
    "fp16_opt_level": "O2", #02
    "max_seq_length": 128, #128, 512 256
    "train_batch_size": 16, #8 16
    "gradient_accumulation_steps": 16, #8 16
    "eval_batch_size": 8, #8
    "num_train_epochs": 6, #8, 3
    "weight_decay": 1e-4,
    "learning_rate": 4e-5, #1e-4 1e-5 1e-8 3e-5 0.01 4e-5 1e-3 3e-5 5e-5
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False, #False
    "optimizer": "AdamW",
    "logging_steps": 50,
    "save_steps": 2000, #2000 1000000000
    "special_tokens_list": [],
    "custom_layer_parameters": [],
    "custom_parameter_groups": [],
    "classification_report": True,
    "use_hf_datasets": False,
    "ModelArgs": False,
    "lazy_loading": False,
    "scheduler": "linear_schedule_with_warmup",
    "train_custom_parameters_only": False,
    "save_optimizer_and_scheduler": True,
    "no_cache": True, #False
    "no_save": False,
    "save_model_every_epoch": True, #True False
    "evaluate_during_training": True,
    "evaluate_each_epoch": True,
    "evaluate_during_training_steps": 2000,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": False,
    "save_eval_checkpoints": True, #True
    "tensorboard_dir": None,
    "overwrite_output_dir": True, #True
    "reprocess_input_data": True,
    "process_count": cpu_count() - 2 if cpu_count() > 2 else 1,
    "onnx": False,
    "n_gpu": 1,
    "dataloader_num_workers": 0,
    "use_multiprocessing": True,
    "multiprocessing_chunksize": 500,
    "use_multiprocessing_for_evaluation": True,
    "silent": False,
    "wandb_project": None,
    "loss_type": None,
    # "wandb_entity": None,
    "wandb_kwargs": {},
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,
    "manual_seed": 123,
    "encoding": None,
}
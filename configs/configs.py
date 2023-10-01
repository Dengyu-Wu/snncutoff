from pydantic import BaseModel, validator
from omegaconf import OmegaConf


class BaseConfig(BaseModel):
    workers: int
    epochs: int
    batch_size: int
    start_epoch: int
    weight_decay: float
    lr: float
    print_freq: int
    seed: int
    log: str
    project: str
    port: str
    gpu_id: str
    nprocs: int
    local_rank: int
    
class SNNConfig(BaseModel):
    T: int
    evaluate: bool
    TET: bool
    means: float
    lamb: float
    alpha: float

class LoggingConfig(BaseModel):
    wandb_logging: bool
    tensorboard_logging: bool
    run_dir: str

class AllConfig(BaseConfig, SNNConfig,LoggingConfig):
    pass
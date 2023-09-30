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

class AllConfig(BaseConfig, SNNConfig):
    pass
from model.vfl_active_model import VFLActiveModel
from torch.optim import Adam
class Server:
    def __init__(self, data,config):
        self.data = data
        self.config=config
        self.model=VFLActiveModel(config)
        self.model.to(config.device)
        self.optimizer=Adam(self.model.parameters(),lr=config.lr)
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []


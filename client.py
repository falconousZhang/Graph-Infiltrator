from model_builder import get_model
from torch.optim import Adam
class Client:
    def __init__(self, client_id, data,config):
        self.client_id = client_id
        self.data = data
        # x_shape, num_label, num_edge_features
        input_shape = data.x.shape, data.edge_attr.shape[1] if data.edge_attr is not None else 0
        self.model=get_model(config,input_shape)
        self.model.to(config.device)
        self.config=config
        is_graph_infiltrator = (
            str(getattr(config, 'attack_mode', '')).lower() == 'graph_infiltrator'
            and 'backdoor' in str(getattr(config, 'training_mode', '')).lower()
        )
        use_lr_augmentation = bool(
            getattr(config, 'enable_lr_augmentation', True)
        ) and is_graph_infiltrator
        if(client_id==config.attack_client_id):
            lr_factor = config.attack_lr_boost if use_lr_augmentation else 1.0
            self.optimizer = Adam(self.model.parameters(), lr=lr_factor * config.lr)
        else:
            self.optimizer = Adam(self.model.parameters(),lr=config.lr)
        self.backdoor_test_loss=[]
        self.backdoor_train_loss=[]
        self.backdoor_train_accuracy = []
        self.backdoor_test_accuracy = []
        self.backdoor_mse = []
        self.backdoor_train_loss_by_target = []
        self.backdoor_test_loss_by_target = []
        self.backdoor_train_accuracy_by_target = []
        self.backdoor_test_accuracy_by_target = []
        self.backdoor_mse_by_target = []
        self.inference_acc=[]
        self.inference_acc_by_target=[]




import torch.optim as optim
def build_optimizer(hparams):
    def optimizer(params): #传入参数
        optim_method = hparams['optim_method'] #选择优化器
        if optim_method == 'adam': #adam优化器
            return optim.Adam(
                params,
                lr = hparams['learning_rate'],
                weight_decay = hparams['weight_decay']
            ) 
        elif optim_method == 'sgd': #sgd优化器
            return optim.SGD(
                params,
                lr=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                momentum=hparams['momentum']
            )
        else:
            raise NotImplementedError

    return optimizer
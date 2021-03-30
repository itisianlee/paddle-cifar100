from paddle import callbacks
from paddle.optimizer.lr import LRScheduler


class LRSchedulerC(callbacks.LRScheduler):
    """Lr scheduler callback function
    Args:
        by_step(bool, optional): whether to update learning rate scheduler 
            by step. Default: True.
        by_epoch(bool, optional): whether to update learning rate scheduler 
            by epoch. Default: False.
    """
    def __init__(self, by_step=False, by_epoch=True, warm_up=True):
        super().__init__(by_step, by_epoch)                                                                                                                          
        assert by_step ^ warm_up
        self.warm_up = warm_up
        
    def on_epoch_end(self, epoch, logs=None):
        if self.by_epoch and not self.warm_up:
            if self.model._optimizer and hasattr(
                self.model._optimizer, '_learning_rate') and isinstance(
                    self.model._optimizer._learning_rate, LRScheduler):                                                                                         
                self.model._optimizer._learning_rate.step()                                                                                          
                                                                                                                                                     
    def on_train_batch_end(self, step, logs=None):                                                                                                   
        if self.by_step or self.warm_up:                                                                                                                             
            if self.model._optimizer and hasattr(
                self.model._optimizer, '_learning_rate') and isinstance(
                    self.model._optimizer._learning_rate, LRScheduler):                                                                                         
                self.model._optimizer._learning_rate.step()
            if self.model._optimizer._learning_rate.last_epoch >= self.model._optimizer._learning_rate.warmup_steps:
                self.warm_up = False


class VisualDLC(callbacks.VisualDL):
    """VisualDL callback function.
    Args:
        log_dir (str): The directory to save visualdl log file.
    """
    def __init__(self, log_dir):
        super().__init__(log_dir)
    
    def on_train_batch_end(self, step, logs=None):
        logs = logs or {}
        logs['lr'] = self.model._optimizer.get_lr()
        self.train_step += 1
        if self._is_write():
            self._updates(logs, 'train')

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        assert self.epochs
        self.train_metrics = self.params['metrics'] + ['lr']
        assert self.train_metrics
        self._is_fit = True
        self.train_step = 0
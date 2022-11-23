from torch.optim.adam import Adam


class NoamOpt(Adam):
    "Optim wrapper that implements rate. Default Transformer Optimizer. Taken from [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)"

    def __init__(self,
                 params,
                 model_size=64,
                 opt_factor=1,
                 opt_warmup=400,
                 opt_beta_1=0.9,
                 opt_beta_2=0.98,
                 eps=1e-9,
                 weight_decay=0):
        super(NoamOpt, self).__init__(params=params,
                                      lr=0,
                                      betas=(opt_beta_1, opt_beta_2),
                                      eps=eps,
                                      weight_decay=weight_decay)

        self._step = 0
        self._rate = 0
        self.factor = opt_factor
        self.warmup = opt_warmup
        self.model_size = model_size

    def step(self, closure=None):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        super().step(closure=closure)

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

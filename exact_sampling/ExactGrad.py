class ExactGrad:
    def __init__(self):
        pass

    def grad(self, F, X, jrandom_key, H): 
        return F.f1(X), 1, None, None, None

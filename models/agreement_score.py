from abc import ABC, abstractmethod
from .taskness_score import cross_entropy_loss


class AgreementScore(ABC, object):
    @abstractmethod
    def __call__(self, p1, p2):
        pass
    
class ClassificationAgreementScore(AgreementScore):
    def __call__(self, p1, p2):
        assert p1.dim() == p2.dim() == 2

        return {
            'ce_agreement': 0.5 * (cross_entropy_loss(p1, p2) + cross_entropy_loss(p2, p1)),
            'acc_agreement': (p1.argmax(1) == p2.argmax(1)).float().mean()
        }
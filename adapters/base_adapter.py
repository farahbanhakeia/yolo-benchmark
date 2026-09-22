from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """
    Classe abstraite.
    Tous les modèles devront hériter de cette classe.
    """

    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.model = None

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, image_path, conf=0.25):
        pass

    def release(self):
        """
        Libération mémoire.
        """
        self.model = None
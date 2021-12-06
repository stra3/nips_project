from abc import ABC, abstractmethod #needed to create abstract class 

class Controller(ABC):
    """
    Abstract class for the controllers.
    """
    def __init__(self):
        pass

    @abstractmethod
    def generateAction(self):
        pass 



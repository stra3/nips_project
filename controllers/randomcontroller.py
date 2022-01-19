import controllers.controller 


class Randomcontroller(controllers.controller.Controller):
    
    """
    random controller
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.name = "random"


    def generateAction(self, obs, e):
        return self.env.action_space.sample() # explore 
    
    def update_q_values(self, e, new_observation, reward, done):
        pass
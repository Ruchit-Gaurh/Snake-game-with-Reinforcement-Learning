# Snake-with-Reinforcement-Learning
**AI learns to protect the snake from colliding into itself within 1hour (1000 games) with a maximum score of 57.**

![Snake Game Demo](https://github.com/Ruchit-Gaurh/Snake-game-with-Reinforcement-Learning/blob/main/images/snake.gif)


**game.py**:- Contains game code and receives move from the agent and each bad move results in the decrease of the score and each good move results in an increase of the score based on which model is further trained.


	class SnakeGameAI:

	    def __init__(self, w=640, h=480):
		self.w = w
		self.h = h
		# init display
		self.display = pygame.display.set_mode((self.w, self.h))
		pygame.display.set_caption('Snake')
		self.clock = pygame.time.Clock()
		self.reset()


**model.py**:- It contains a Linear Qnet model and QTrainer. 
* Linear Qnet creates a neural network with 1 hidden layer and saves the record and n games in a file model.pth.
* QTrainer trains the model on given inputs like state, action, length, reward, etc.

	`class Linear_QNet(nn.Module):`

	    def __init__(self, input_size, hidden_size, output_size):
		super().__init__()

		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, output_size)
	
**agent.py**:- It is the main file our agent will play the game through this file. agnet.py put up all our code together, get the current state of the game, save data in short-term memory and in long-term memory to train our model. It also saves checkpoints in case we have to stop training and resume it again.

`class Agent:`

    def __init__(self):
        
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #remove elements from left if we exceeds memory space
        self.model = Linear_QNet(11, 256, 3)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr= 0.001)

        Agent.load_checkpoints(self)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

**helper.py**:- It plots score and mean scores to analyze the model's performance

    def plot(scores, mean_scores):

	    display.clear_output(wait=True)
	    display.display(plt.gcf())
	    plt.clf()
	    plt.title('Training...')
	    plt.xlabel('Number of Games')
	    plt.ylabel('Score')
	    plt.plot(scores)
	    plt.plot(mean_scores)
	    plt.ylim(ymin=0)
	    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
	    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
	    plt.show(block=False)
	    plt.pause(.1)


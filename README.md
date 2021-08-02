# Snake-with-Reinforcement-Learning
**AI learns to protect the snake from colliding into itself within 1hour (1000 games) with a maximum score of 57.**

![Snake Game Demo](https://github.com/Ruchit-Gaurh/Snake-game-with-Reinforcement-Learning/blob/main/images/snake.gif)


**game.py**:- Contains game code and receives move from the agent and each bad move results in the decrease of the score and each good move results in an increase of the score based on which model is further trained.

**model.py**:- It contains a Linear Qnet model and QTrainer. 
* Linear Qnet creates a neural network with 1 hidden layer and saves the record and n games in a file model.pth.
* QTrainer trains the model on given inputs like state, action, length, reward, etc.
	
**agent.py**:- It is the main file our agent will play the game through this file. agnet.py put up all our code together, get the current state of the game, save data in short-term memory and in long-term memory to train our model. It also saves checkpoints in case we have stop train and resume it again.

**helper.py**:- It plots score and mean scores to analyze the model's performance

import pygame
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.agent import Agent

pygame.init()

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300
CELL_SIZE = 25
GRID_SIZE = SCREEN_WIDTH // CELL_SIZE
SPEED = 1000000000

clock = pygame.time.Clock()

DRONE_COLOR = (248, 168, 0)
BACKGROUND_COLOR = (104, 56, 0)
USER_COLOR = (1, 252, 128)
BORDER_COLOR =  (255, 0, 0)
SCORE_COLOR = (248, 252, 248)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Drone with AI")

font = pygame.font.Font(None, 36)

def plot(scores, mean_scores):
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.legend()
    plt.show()


class DroneGameAI:
    def __init__(self):
        self.drone = (random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2))
        self.direction = (1, 0)
        self.n_games = 0
        self.users = self._place_user()
        self.score = 0
        self.clock_wise = [(0, -1), (1, 0), (0, 1), (-1, 0)] # haut, droite, bas, gauche
        self.debit = self._score(self.drone)
        self.max_debit = self.debit
        self.gradient = 0
        self.gradient_prev = 0
        self.time = 40
        self.opti = self._optimal()
        self.visited = set() # Un 'set' est ultra rapide pour vérifier si une case y est
        self.visited.add(self.drone)

    def _place_user(self):
        while True:
            position = []
            nb_user = 1
            for _ in range(nb_user):
                user_position = (random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2))
                if user_position != self.drone and user_position not in position:
                    position.append(user_position + (random.randint(1, 10),))
            return position
        
    def _optimal(self):
        meilleur_score = -float('inf')
        meilleure_position = (0, 0)
        coords_users = [(u[0], u[1]) for u in self.users]
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) in coords_users:
                    continue
                score_case = 0
                for u in self.users:
                    distance = np.sqrt((x - u[0])**2 + (y - u[1])**2) + 0.1
                    signal = u[2] / distance
                    score_case += signal
                if score_case > meilleur_score:
                    meilleur_score = score_case
                    meilleure_position = (x, y)
        return [meilleure_position, meilleur_score]
    
    def _score(self, position):
        x, y = position
        score_case = 0
        for u in self.users:
            distance = np.sqrt((x - u[0])**2 + (y - u[1])**2) + 0.1
            signal = u[2] / distance
            score_case += signal
        return score_case

    def reset(self):
        self.drone = (random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2))
        self.direction = (1, 0)
        self.n_games += 1
        self.users = self._place_user()
        self.score = 0
        self.debit = self._score(self.drone)
        self.max_debit = self.debit
        self.gradient = 0
        self.gradient_prev = 0
        self.time = 40
        self.opti = self._optimal()
        self.visited = set() # Un 'set' est ultra rapide pour vérifier si une case y est
        self.visited.add(self.drone)

    def get_state(self):
        x, y = self.drone
        debnorm = self.debit / self.opti[1] if self.opti[1] > 0 else 0
        state = [
            debnorm,   
            np.clip(self.gradient / 5, -1, 1),
            np.clip(self.gradient_prev / 5, -1, 1),
            self.time / 40,
            self._is_collision((x, y-1))[0],
            self._is_collision((x+1, y))[0],
            self._is_collision((x, y+1))[0],
            self._is_collision((x-1, y))[0]
        ]
        return np.array(state, dtype=np.float32)

    def _is_collision(self, position=None):
        if position is None:
            position = self.drone
        if not (1 <= position[0] < GRID_SIZE - 1 and 1 <= position[1] < GRID_SIZE - 1):
            return True, -50
        coords_users = [(u[0], u[1]) for u in self.users]
        if position in coords_users:
            return True, -50
        return False, 0

    def play_step(self, action, plot_scores, plot_mean_scores, agent):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                agent.model.save('model_fin.pth')
                pygame.quit()
                plot(plot_scores, plot_mean_scores)
                sys.exit()

        self.time -= 1
        new_idx = action.index(1)
        self.direction = self.clock_wise[new_idx]
        x, y = self.drone
        delta_x, delta_y = self.direction
        self.drone = (x + delta_x, y + delta_y)
        
        self.gradient_prev = self.gradient
        self.gradient = self._score(self.drone) - self.debit
        self.debit = self._score(self.drone)

        with open("historique_gradients.txt", "a") as fichier:
            fichier.write(f"{self.gradient}\n")
        
        reward = 0
        done = False

        collision, r = self._is_collision()
        if collision:
            reward += r
            done = True
            return reward, done, self.score

        if self.time <= 0:
            reward -= 50
            done = True
            return reward, done, self.score

        if self.max_debit < self.debit :
            reward += 20 + np.clip(self.gradient / 5, 0, 1.0) * 10
            self.max_debit = self.debit

        if self.opti[1] > 0 and (self.debit / self.opti[1]) > 0.95:
            self.score += 1
            reward += 100
            self.users = self._place_user()
            self.opti = self._optimal()
            self.debit = self._score(self.drone)
            self.max_debit = self.debit
            self.time = 40
            self.gradient = 0
            self.gradient_prev = 0
            self.visited = set()
            self.visited.add(self.drone)

        return reward, done, self.score


    def draw(self, n_games, record):
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, BORDER_COLOR, pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), CELL_SIZE)

        x, y = self.drone
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, DRONE_COLOR, rect)

        for x, y, _ in self.users:
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, USER_COLOR, rect)

        text = font.render(f"Score: {self.score} Temps: {self.time}", True, SCORE_COLOR)
        screen.blit(text, (10, 10))

        text = font.render(f"Games: {n_games} Record: {record}", True, SCORE_COLOR)
        screen.blit(text, (10, 50))

        pygame.display.flip()

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    rewardtotal = 0
    agent = Agent()
    agent.model.load("model_1_score623.pth")
    game = DroneGameAI()
    run = True
    while run:

        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move, plot_scores, plot_mean_scores, agent)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        rewardtotal += reward
        agent.remember(state_old, final_move, reward, state_new, done)
        game.draw(agent.n_games, record)
        clock.tick(SPEED)
       
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save('model_' + str(record) + '.pth')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Reward Total:', rewardtotal, 'Moyenne Score:', mean_score)
            plot_mean_scores.append(mean_score)
            rewardtotal = 0
    #plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()

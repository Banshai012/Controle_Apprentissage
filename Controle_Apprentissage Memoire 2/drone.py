import pygame
import random
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from src.agent import Agent

pygame.init()

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300
CELL_SIZE = 10
GRID_SIZE = SCREEN_WIDTH // CELL_SIZE
SPEED = 1000000

clock = pygame.time.Clock()

DRONE_COLOR = (248, 168, 0)
BACKGROUND_COLOR = (104, 56, 0)
USER_COLOR = (1, 252, 128)
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
        self.drone = (random.randint(0, GRID_SIZE), random.randint(0, GRID_SIZE))
        self.direction = (1, 0)
        self.dir_idx = 1
        self.dir_idx_prev = 1
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

    def _place_user(self):
        while True:
            position = []
            nb_user = random.randint(1, 5)
            for _ in range(nb_user):
                user_position = (random.randint(0, GRID_SIZE), random.randint(0, GRID_SIZE))
                position.append(user_position + (random.randint(1, 10),))
            return position
        
    def _optimal(self):
        meilleur_score = -float('inf')
        meilleure_position = (0, 0)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                score_case = 0
                for u in self.users:
                    distance = np.sqrt((x - u[0])**2 + (y - u[1])**2 + 1000)
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
            distance = np.sqrt((x - u[0])**2 + (y - u[1])**2 + 1000)
            signal = u[2] / distance
            score_case += signal
        return score_case

    def reset(self):
        self.drone = (random.randint(0, GRID_SIZE), random.randint(0, GRID_SIZE))
        self.direction = (1, 0)
        self.dir_idx = 1
        self.dir_idx_prev = 1
        self.n_games += 1
        self.users = self._place_user()
        self.score = 0
        self.debit = self._score(self.drone)
        self.max_debit = self.debit
        self.gradient = 0
        self.gradient_prev = 0
        self.time = 40
        self.opti = self._optimal()

    def get_state(self):
        debnorm = self.debit / self.opti[1] if self.opti[1] > 0 else 0
        dir_actuelle = [0, 0, 0, 0]
        dir_actuelle[self.dir_idx] = 1
        dir_prev = [0, 0, 0, 0]
        dir_prev[self.dir_idx_prev] = 1
        state = [
            debnorm,   
            np.clip(self.gradient, 0.01, 1.0)*math.copysign(1, self.gradient),
            np.clip(self.gradient_prev, 0.01, 1.0)*math.copysign(1, self.gradient_prev),
            self.time / 40
        ] + dir_actuelle + dir_prev # On fusionne les listes
        return np.array(state, dtype=np.float32)

    def play_step(self, action, plot_scores, plot_mean_scores, agent):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                agent.model.save('model_fin.pth')
                pygame.quit()
                plot(plot_scores, plot_mean_scores)
                sys.exit()

        self.time -= 1
        self.dir_idx_prev = self.dir_idx
        new_idx = action.index(1)
        self.dir_idx = new_idx
        self.direction = self.clock_wise[new_idx]
        x, y = self.drone
        delta_x, delta_y = self.direction
        self.drone = (x + delta_x, y + delta_y)
        
        self.gradient_prev = self.gradient
        self.gradient = self._score(self.drone) - self.debit
        self.debit = self._score(self.drone)
        
        reward = -1
        done = False

        if self.time <= 0:
            reward -= 50
            done = True
            return reward, done, self.score

        if self.max_debit < self.debit :
            reward += 20 + np.clip(self.gradient / 5, 0.01, 1.0) * 10
            self.max_debit = self.debit

        if self.opti[1] > 0 and (self.debit / self.opti[1]) > 0.95:
            self.score += 1
            reward += 500
            self.users = self._place_user()
            self.opti = self._optimal()
            self.debit = self._score(self.drone)
            self.max_debit = self.debit
            self.time = 40
            self.gradient = 0
            self.gradient_prev = 0

        return reward, done, self.score


    def draw(self, n_games, record):
        screen.fill(BACKGROUND_COLOR)

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

def train(epsilon_start=1.0, duration_minutes=5):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    rewardtotal = 0
    mean_score = 0
    
    agent = Agent()
    agent.epsilon = epsilon_start # On injecte l'epsilon de l'instance
    
    # On essaie de charger le meilleur modèle existant
    agent.load_latest_model()
    
    game = DroneGameAI()
    
    start_time = time.time()
    duration_seconds = duration_minutes * 60
    
    run = True
    while run:
        # Condition d'arrêt chronométrée
        if time.time() - start_time > duration_seconds:
            print(f"Fin de l'instance (Epsilon: {epsilon_start}) après {duration_minutes} min.")
            agent.model.save('model_fin_' + str(int(mean_score*1000)) + '.pth')
            break

        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move, plot_scores, plot_mean_scores, agent)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        rewardtotal += reward
        agent.remember(state_old, final_move, reward, state_new, done)
        #game.draw(agent.n_games, record)
        clock.tick(SPEED)
       
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save('model_' + str(record) + '.pth')

            mean_score = total_score / agent.n_games
            print(f'[Eps: {agent.epsilon:.2f}] Game {agent.n_games} | Score: {score} | Record: {record} | Moyenne : {mean_score:.2f}')
            plot_mean_scores.append(mean_score) 
            plot_scores.append(score)
            total_score += score
            rewardtotal = 0

if __name__ == '__main__':
    train(epsilon_start=1, duration_minutes=480)
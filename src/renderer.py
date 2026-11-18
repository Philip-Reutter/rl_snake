"""pygame view of a SnakeEnv, reads the env state only"""
import pygame

BG = (24, 26, 32)
GRID = (36, 39, 47)
BODY = (46, 160, 90)
HEAD = (140, 230, 150)
FOOD = (235, 90, 80)
TEXT = (230, 230, 225)
HUD_HEIGHT = 44


class Renderer:
    def __init__(self, grid_size, cell=30):
        pygame.init()
        self.size = grid_size
        self.cell = cell
        self.screen = pygame.display.set_mode((grid_size * cell, grid_size * cell + HUD_HEIGHT))
        pygame.display.set_caption("Snake DQN")
        self.font = pygame.font.Font(None, 26)
        self.clock = pygame.time.Clock()

    def draw(self, env, title="", note=""):
        c = self.cell
        self.screen.fill(BG)
        board = pygame.Rect(0, HUD_HEIGHT, self.size * c, self.size * c)
        pygame.draw.rect(self.screen, GRID, board, width=1)
        for i in range(1, self.size):
            pygame.draw.line(self.screen, GRID, (i * c, HUD_HEIGHT), (i * c, HUD_HEIGHT + self.size * c))
            pygame.draw.line(self.screen, GRID, (0, HUD_HEIGHT + i * c), (self.size * c, HUD_HEIGHT + i * c))

        if env.food is not None:
            self._cell(env.food, FOOD, inset=5, radius=c // 2)
        for i, pos in enumerate(env.snake):
            self._cell(pos, HEAD if i == 0 else BODY, inset=2, radius=6)

        left = self.font.render(title, True, TEXT)
        right = self.font.render(f"{note}  score {env.score}  step {env.steps}", True, TEXT)
        self.screen.blit(left, (10, 12))
        self.screen.blit(right, (self.size * c - right.get_width() - 10, 12))
        pygame.display.flip()

    def _cell(self, pos, color, inset, radius):
        c = self.cell
        rect = pygame.Rect(pos[0] * c + inset, HUD_HEIGHT + pos[1] * c + inset, c - 2 * inset, c - 2 * inset)
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)

    def poll(self):
        """Returns 'quit' (window closed / ESC / Q), 'skip' (N or right arrow) or None."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return "quit"
                if event.key in (pygame.K_n, pygame.K_RIGHT):
                    return "skip"
        return None

    def tick(self, fps):
        self.clock.tick(fps)

    def save(self, path):
        pygame.image.save(self.screen, str(path))

    def close(self):
        pygame.quit()

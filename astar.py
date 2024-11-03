import pygame
from queue import PriorityQueue, Queue, LifoQueue


pygame.init()
pygame.font.init()

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Path Finding Algos")

# colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # down
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # up
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # right
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # left
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

# heuristic function
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
    path = []
    while current in came_from and came_from[current] is not None:
        path.append(current.get_pos())
        current = came_from[current]
        current.make_path()
        draw()

    # revereresing path to start from the beginning
    path = path[::-1]
    print_path(path)


def print_path(path):
    
    print("Path")
    if path:
        print("Start:")
        for pos in path:
            print(f"  {pos},")
        print("Goal")
        print(f"Length: {len(path)} steps")
    else:
        print("No path found.")

def draw_button(win, text, x, y, width, height, color):
    pygame.draw.rect(win, color, (x, y, width, height))
    font = pygame.font.SysFont(None, 24)
    label = font.render(text, True, WHITE)
    win.blit(label, (x + (width - label.get_width()) // 2, y + (height - label.get_height()) // 2))


#
def a_star(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False




def dijkstra(draw, grid, start, end):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[1]

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                open_set.put((g_score[neighbor], neighbor))
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False  # no path

def uniform_cost_search(draw, grid, start, end):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[1]

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1  # equal weights
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                open_set.put((g_score[neighbor], neighbor))
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False 


def handle_algorithm_selection(algorithm, draw, grid, start, end):
    for row in grid:
        for spot in row:
            spot.update_neighbors(grid)

    if algorithm == "A*":
        a_star(draw, grid, start, end)
    elif algorithm == "BFS":
        bfs(draw, grid, start, end)
    elif algorithm == "DFS":
        dfs(draw, grid, start, end)
    elif algorithm == "Dijkstra":
        dijkstra(draw, grid, start, end)
    elif algorithm == "UCS":
        uniform_cost_search(draw, grid, start, end)

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    draw_button(win, "A*", 650, 50, 100, 30, GREY)
    draw_button(win, "BFS", 650, 100, 100, 30, GREY)
    draw_button(win, "DFS", 650, 150, 100, 30, GREY)
    draw_button(win, "Dijkstra", 650, 200, 100, 30, GREY)
    draw_button(win, "UCS", 650, 250, 100, 30, GREY)
    pygame.display.update()
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    draw_button(win, "A*", 650, 50, 100, 30, GREY)
    draw_button(win, "BFS", 650, 100, 100, 30, GREY)
    draw_button(win, "DFS", 650, 150, 100, 30, GREY)
    draw_button(win, "Dijkstra", 650, 200, 100, 30, GREY)
    draw_button(win, "UCS", 650, 250, 100, 30, GREY)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    return y // gap, x // gap



def bfs(draw, start, end):
    queue = Queue()
    queue.put(start)
    came_from = {start: None}

    while not queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.get()

        if current == end:
            if end in came_from:
                reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if neighbor not in came_from: 
                queue.put(neighbor)
                came_from[neighbor] = current
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def dfs(draw, start, end):
    stack = LifoQueue()
    stack.put(start)
    came_from = {start: None}

    while not stack.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = stack.get()

        if current == end:
            if end in came_from:
                reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if neighbor not in came_from:
                stack.put(neighbor)
                came_from[neighbor] = current
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False



def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)
    start = None
    end = None
    algorithm = None
    run = True

    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # Left click
                pos = pygame.mouse.get_pos()
                if 650 <= pos[0] <= 750:
                    if 50 <= pos[1] <= 80:
                        algorithm = "A*"
                    elif 100 <= pos[1] <= 130:
                        algorithm = "BFS"
                    elif 150 <= pos[1] <= 180:
                        algorithm = "DFS"
                    elif 200 <= pos[1] <= 230:
                        algorithm = "Dijkstra"
                    elif 250 <= pos[1] <= 280:
                        algorithm = "UCS"
                else:
                    row, col = get_clicked_pos(pos, ROWS, width)
                    spot = grid[row][col]
                    if not start and spot != end:
                        start = spot
                        start.make_start()
                    elif not end and spot != start:
                        end = spot
                        end.make_end()
                    elif spot != end and spot != start:
                        spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # right click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end and algorithm:
                    handle_algorithm_selection(algorithm, lambda: draw(win, grid, ROWS, width), grid, start, end)
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()

main(WIN, WIDTH)
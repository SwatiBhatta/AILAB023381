import heapq
import itertools

class NPuzzle:
    def __init__(self, board):
        self.n = len(board)
        self.start = tuple(itertools.chain.from_iterable(board))
        self.goal = tuple(list(range(1, self.n * self.n)) + [0])

    def manhattan(self, state):
        distance = 0
        for i, value in enumerate(state):
            if value == 0:
                continue
            target_x = (value - 1) // self.n
            target_y = (value - 1) % self.n
            current_x = i // self.n
            current_y = i % self.n
            distance += abs(current_x - target_x) + abs(current_y - target_y)
        return distance

    def get_neighbors(self, state):
        neighbors = []
        zero_index = state.index(0)
        x, y = zero_index // self.n, zero_index % self.n

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n:
                new_index = nx * self.n + ny
                new_state = list(state)
                new_state[zero_index], new_state[new_index] = (
                    new_state[new_index],
                    new_state[zero_index],
                )
                neighbors.append(tuple(new_state))
        return neighbors

    def is_solvable(self):
        inv_count = 0
        state = [x for x in self.start if x != 0]
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] > state[j]:
                    inv_count += 1

        if self.n % 2 == 1:
            return inv_count % 2 == 0
        else:
            zero_row = self.start.index(0) // self.n
            return (inv_count + zero_row) % 2 == 1

    def solve(self):
        if not self.is_solvable():
            print("Puzzle is not solvable!")
            return None

        open_set = []
        heapq.heappush(open_set, (0, 0, self.start))
        came_from = {}
        g_score = {self.start: 0}

        counter = itertools.count()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == self.goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.manhattan(neighbor)
                    heapq.heappush(open_set, (f_score, next(counter), neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def print_board(self, state):
        for i in range(self.n):
            row = state[i*self.n:(i+1)*self.n]
            print(" ".join(str(x) if x != 0 else "_" for x in row))
        print()


# Example usage
if __name__ == "__main__":
    board = [
        [1, 2, 3],
        [4, 0, 6],
        [7, 5, 8]
    ]

    puzzle = NPuzzle(board)
    solution = puzzle.solve()

    if solution:
        print(f"Solution found in {len(solution) - 1} moves:\n")
        for state in solution:
            puzzle.print_board(state)


import heapq


def heuristic(node, goal, method="manhattan"):
    """计算启发函数"""
    if method == "zero":
        return 0
    elif method == "manhattan":
        distance = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
        return max(0, distance - 1)
    else:
        raise ValueError(f"Unknown heuristic: {method}")


def reconstruct_path(came_from, current):
    """恢复完整路径"""
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    return path[::-1]


def get_move_cost(cell):
    """进入指定地形的代价"""
    if cell == "E":
        return 0
    if cell == "S":
        return 1
    return int(cell)


def astar(start, goal, grid, method="manhattan"):
    """
    加权地形 A* 搜索

    返回：
        cost: 最小总代价，无解时为 -1
        path: 最优路径
        expanded: 扩展节点数量
    """
    rows, cols = len(grid), len(grid[0])

    open_heap = []
    counter = 0
    heapq.heappush(
        open_heap,
        (heuristic(start, goal, method), counter, start)
    )

    g_score = {start: 0}
    came_from = {}
    closed = set()
    expanded = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            return g_score[current], path, expanded

        for dr, dc in directions:
            nr = current[0] + dr
            nc = current[1] + dc
            neighbor = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if grid[nr][nc] == "X":
                continue

            if neighbor in closed:
                continue

            move_cost = get_move_cost(grid[nr][nc])
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current

                h = heuristic(neighbor, goal, method)
                counter += 1

                heapq.heappush(
                    open_heap,
                    (tentative_g + h, counter, neighbor)
                )

    return -1, [], expanded


def main():
    grid = [
        ["S", "1", "1", "8", "1"],
        ["1", "8", "3", "1", "1"],
        ["1", "8", "1", "8", "1"],
        ["1", "1", "1", "8", "1"],
        ["8", "8", "1", "1", "E"]
    ]

    start = (0, 0)
    goal = (4, 4)

    for method in ["zero", "manhattan"]:
        cost, path, expanded = astar(start, goal, grid, method)

        print(f"\nHeuristic: {method}")
        print(f"Minimum cost: {cost}")
        print(f"Path: {path}")
        print(f"Expanded nodes: {expanded}")

    # 无解测试
    blocked_grid = [
        ["S", "X", "E"],
        ["1", "X", "1"],
        ["1", "X", "1"]
    ]

    cost, path, expanded = astar(
        (0, 0), (0, 2), blocked_grid, "manhattan"
    )

    print("\nNo-path test")
    print(f"Minimum cost: {cost}")
    print(f"Path: {path}")
    print(f"Expanded nodes: {expanded}")



if __name__ == "__main__":
    main()

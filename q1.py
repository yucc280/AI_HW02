
import heapq


def heuristic(node, goal, method="manhattan"):
    """计算启发函数 h(n)"""
    if method == "zero":
        return 0
    elif method == "manhattan":
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    else:
        raise ValueError(f"Unknown heuristic: {method}")


def reconstruct_path(came_from, current):
    """根据前驱节点记录，恢复起点到终点的完整路径"""
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    return path[::-1]


def astar(start, goal, grid, method="manhattan"):
    """
    A* 迷宫寻路

    返回：
        path: 最短路径坐标序列，无解时返回 []
        expanded: 扩展节点数量
    """
    rows, cols = len(grid), len(grid[0])

    # 优先队列元素：(f, 插入顺序, 节点)
    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal, method), counter, start))

    # g_score 保存从起点到各节点的当前最小已知代价
    g_score = {start: 0}

    # 记录最优路径上每个节点的前驱
    came_from = {}

    # 已经扩展过的节点
    closed = set()
    expanded = 0

    # 上、下、左、右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        # 跳过已经扩展过的节点
        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        # 到达终点
        if current == goal:
            return reconstruct_path(came_from, current), expanded

        # 扩展四个方向的相邻节点
        for dr, dc in directions:
            nr = current[0] + dr
            nc = current[1] + dc
            neighbor = (nr, nc)

            # 越界检查
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # 障碍物检查
            if grid[nr][nc] == "X":
                continue

            if neighbor in closed:
                continue

            # 每移动一格，代价为 1
            tentative_g = g_score[current] + 1

            # 只有找到更优路径时才更新
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                h = heuristic(neighbor, goal, method)
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative_g + h, counter, neighbor)
                )

    # 优先队列为空，说明无路可走
    return [], expanded


def main():
    # 指导书提供的测试用例
    grid = [
        ['S', '.', '.', 'X', '.'],
        ['.', 'X', '.', '.', '.'],
        ['.', 'X', '.', 'X', '.'],
        ['.', '.', '.', 'X', 'E'],
        ['X', 'X', '.', '.', '.']
    ]

    start = (0, 0)
    goal = (3, 4)

    for method in ["zero", "manhattan"]:
        path, expanded = astar(start, goal, grid, method)

        print(f"\nHeuristic: {method}")
        print(f"Path: {path}")
        print(f"Path length: {len(path) - 1 if path else -1}")
        print(f"Expanded nodes: {expanded}")

    # 无解测试：障碍物完全隔开起点和终点
    blocked_grid = [
        ['S', 'X', 'E'],
        ['.', 'X', '.'],
        ['.', 'X', '.']
    ]

    path, expanded = astar((0, 0), (0, 2), blocked_grid, "manhattan")

    print("\nNo-path test")
    print(f"Path: {path}")
    print(f"Expanded nodes: {expanded}")



if __name__ == "__main__":
    main()

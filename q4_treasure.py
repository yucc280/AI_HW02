
import heapq
from dataclasses import dataclass


# 地图：X 为障碍物，数字表示进入代价
grid = [
    ["A",  "1", "1", "X", "2", "1", "T1"],
    ["1",  "X", "6", "X", "1", "X", "1"],
    ["1",  "1", "T2", "1", "1", "1", "1"],
    ["X",  "1", "X", "8", "X", "1", "X"],
    ["1",  "1", "1", "1", "T3", "1", "1"],
    ["1",  "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"]
]

# 六个关键位置
positions = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0)
}

# 宝藏价值
values = {
    "T1": 9,
    "T2": 6,
    "T3": 7,
    "T4": 7
}


def heuristic(node, goal):
    """曼哈顿距离"""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def get_move_cost(cell):
    """获取进入一个格子的代价"""
    if cell in ("A", "B", "T1", "T2", "T3", "T4"):
        return 1
    return int(cell)


def astar_distance(start, goal):
    """使用 A* 计算两点之间的最小移动代价"""
    rows, cols = len(grid), len(grid[0])

    open_heap = [(heuristic(start, goal), 0, start)]
    g_score = {start: 0}
    counter = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        # 跳过过期的优先队列条目
        if f != g_score[current] + heuristic(current, goal):
            continue

        if current == goal:
            return g_score[current]

        for dr, dc in directions:
            nr = current[0] + dr
            nc = current[1] + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if grid[nr][nc] == "X":
                continue

            neighbor = (nr, nc)
            move_cost = get_move_cost(grid[nr][nc])
            new_g = g_score[current] + move_cost

            if new_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = new_g
                counter += 1

                heapq.heappush(
                    open_heap,
                    (new_g + heuristic(neighbor, goal),
                     counter, neighbor)
                )

    return -1


def build_distance_matrix():
    """计算六个关键位置之间的距离矩阵"""
    names = list(positions.keys())
    distances = {}

    for a in names:
        distances[a] = {}
        for b in names:
            distances[a][b] = astar_distance(
                positions[a], positions[b]
            )

    return names, distances


@dataclass(frozen=True)
class GameState:
    """记录当前博弈状态"""
    pos_a: str
    pos_b: str
    remaining: tuple
    score_a: int
    score_b: int
    player: str


def make_move(state, target, distances):
    """当前玩家选择宝藏，生成下一状态"""
    remaining = tuple(
        t for t in state.remaining if t != target
    )

    if state.player == "A":
        gain = values[target] - distances[state.pos_a][target]
        return GameState(
            target, state.pos_b, remaining,
            state.score_a + gain, state.score_b, "B"
        )

    else:
        gain = values[target] - distances[state.pos_b][target]
        return GameState(
            state.pos_a, target, remaining,
            state.score_a, state.score_b + gain, "A"
        )


def minimax(state, depth, distances, order="left"):
    """
    返回：(最佳收益差, 当前最佳选择, 叶子访问数)
    """
    if depth == 0 or not state.remaining:
        return state.score_a - state.score_b, None, 1

    is_max = state.player == "A"
    best = float("-inf") if is_max else float("inf")
    best_target = None
    leaf_count = 0

    targets = (
        state.remaining
        if order == "left"
        else reversed(state.remaining)
    )

    for target in targets:
        next_state = make_move(state, target, distances)
        value, _, count = minimax(
            next_state, depth - 1, distances, order
        )
        leaf_count += count

        if (is_max and value > best) or (
            not is_max and value < best
        ):
            best = value
            best_target = target

    return best, best_target, leaf_count


def alpha_beta(state, depth, distances,
               alpha=float("-inf"), beta=float("inf"), order='left'):
    """
    返回：(最佳收益差, 当前最佳选择, 叶子访问数)
    """
    if depth == 0 or not state.remaining:
        return state.score_a - state.score_b, None, 1

    is_max = state.player == "A"
    best = float("-inf") if is_max else float("inf")
    best_target = None
    leaf_count = 0

    targets = (
        state.remaining
        if order == "left"
        else reversed(state.remaining)
    )

    for target in targets:
        next_state = make_move(state, target, distances)

        value, _, count = alpha_beta(
            next_state, depth - 1, distances,
            alpha, beta, order
        )
        leaf_count += count

        if is_max:
            if value > best:
                best = value
                best_target = target
            alpha = max(alpha, best)
        else:
            if value < best:
                best = value
                best_target = target
            beta = min(beta, best)

        if alpha >= beta:
            break

    return best, best_target, leaf_count


def main():
    names, distances = build_distance_matrix()

    # 输出距离矩阵
    print("Distance Matrix:")
    print(f"{'':>5}", end="")
    for name in names:
        print(f"{name:>5}", end="")
    print()

    for a in names:
        print(f"{a:>5}", end="")
        for b in names:
            print(f"{distances[a][b]:>5}", end="")
        print()

    # 初始化博弈状态
    initial_state = GameState(
        pos_a="A",
        pos_b="B",
        remaining=tuple(values.keys()),
        score_a=0,
        score_b=0,
        player="A"
    )

    # 不同深度的算法对照
    for H in [2, 4]:
        print(f"\n===== H = {H} =====")
    
        for order in ["left", "right"]:
            value1, target1, leaves1 = minimax(
                initial_state, H, distances, order
            )
    
            value2, target2, leaves2 = alpha_beta(
                initial_state, H, distances,
                order=order
            )
    
            print(f"\nOrder: {order}")
            print(f"Minimax: target={target1}, value={value1}, "
                  f"visited leaves={leaves1}")
            print(f"Alpha-Beta: target={target2}, value={value2}, "
                  f"visited leaves={leaves2}")


if __name__ == "__main__":
    main()

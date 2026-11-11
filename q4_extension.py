
from q4_treasure import (
    GameState, make_move, build_distance_matrix,
    values, alpha_beta, minimax
)

def order_targets(state, distances):
    """根据当前玩家的即时收益动态排列宝藏"""
    position = state.pos_a if state.player == "A" else state.pos_b

    return sorted(
        state.remaining,
        key=lambda target: values[target] - distances[position][target],
        reverse=True
    )


def alpha_beta_dynamic(
    state, depth, distances,
    alpha=float("-inf"), beta=float("inf")
):
    """使用即时收益排序的 Alpha-Beta 搜索"""
    if depth == 0 or not state.remaining:
        return state.score_a - state.score_b, None, 1

    is_max = state.player == "A"
    best = float("-inf") if is_max else float("inf")
    best_target = None
    leaf_count = 0

    # 核心改进：不再使用固定的宝藏顺序
    targets = order_targets(state, distances)

    for target in targets:
        next_state = make_move(state, target, distances)

        value, _, count = alpha_beta_dynamic(
            next_state, depth - 1, distances, alpha, beta
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
    _, distances = build_distance_matrix()

    initial_state = GameState(
        pos_a="A",
        pos_b="B",
        remaining=tuple(values.keys()),
        score_a=0,
        score_b=0,
        player="A"
    )

    # 展示根节点的动态排序结果
    print("Initial dynamic order:")
    print(order_targets(initial_state, distances))

    for H in [2, 4]:
        print(f"\n===== H = {H} =====")

        value, target, leaves = minimax(initial_state, H, distances)
        print(f"Minimax: target={target}, value={value}, leaves={leaves}")

        for order in ["left", "right"]:
            
            value, target, leaves = alpha_beta(initial_state, H, distances, order=order)
            print(f"Fixed-{order}: target={target}, value={value}, leaves={leaves}")

        value, target, leaves = alpha_beta_dynamic(initial_state, H, distances)
        print(f"Dynamic: target={target}, value={value}, leaves={leaves}")


if __name__ == "__main__":
    main()

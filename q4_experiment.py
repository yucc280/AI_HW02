
from time import perf_counter
from q4_treasure import (
    GameState, astar_distance, positions, values
)


# 扩展场景：保留原始4个宝藏，增加T5、T6
extra_positions = {
    "T5": (0, 2),
    "T6": (2, 5)
}

extra_values = {
    "T5": 5,
    "T6": 8
}


def create_scenario(n, custom_values=None):
    """创建场景，可选自定义宝藏价值"""
    pos = positions.copy()
    treasure_values = values.copy()

    if n == 6:
        pos.update(extra_positions)
        treasure_values.update(extra_values)
    elif n != 4:
        raise ValueError("Currently supports 4 or 6 treasures")

    if custom_values is not None:
        if set(custom_values) != set(treasure_values):
            raise ValueError("Treasure values do not match scenario")
        treasure_values.update(custom_values)

    distances = {
        a: {
            b: astar_distance(pos[a], pos[b])
            for b in pos
        }
        for a in pos
    }

    return treasure_values, distances


def make_next_state(state, target, distances, treasure_values):
    """选择宝藏后，生成新的博弈状态"""
    remaining = tuple(
        t for t in state.remaining if t != target
    )

    if state.player == "A":
        gain = (
            treasure_values[target]
            - distances[state.pos_a][target]
        )
        return GameState(
            target, state.pos_b, remaining,
            state.score_a + gain, state.score_b, "B"
        )

    gain = (
        treasure_values[target]
        - distances[state.pos_b][target]
    )
    return GameState(
        state.pos_a, target, remaining,
        state.score_a, state.score_b + gain, "A"
    )


def order_targets(state, distances, treasure_values, method):
    """根据不同策略决定宝藏访问顺序"""
    if method == "minimax" or method == "left":
        return state.remaining

    if method == "right":
        return tuple(reversed(state.remaining))

    if method == "dynamic":
        position = (
            state.pos_a if state.player == "A"
            else state.pos_b
        )
        return sorted(
            state.remaining,
            key=lambda t: (
                treasure_values[t] - distances[position][t]
            ),
            reverse=True
        )

    raise ValueError(f"Unknown method: {method}")


def search(state, depth, distances, treasure_values,
           method, stats,
           alpha=float("-inf"), beta=float("inf"), level=0):
    """统一实现Minimax及三种Alpha-Beta策略"""

    # 统计递归访问的全部节点
    stats["nodes"] += 1
    stats["nodes_by_depth"][level] += 1

    if depth == 0 or not state.remaining:
        stats["leaves"] += 1
        return state.score_a - state.score_b, None

    is_max = state.player == "A"
    best = float("-inf") if is_max else float("inf")
    best_target = None

    if method == "dynamic":
        stats["sort_calls"] += 1
        stats["sort_items"] += len(state.remaining)

    targets = order_targets(
        state, distances, treasure_values, method
    )

    for target in targets:
        next_state = make_next_state(
            state, target, distances, treasure_values
        )

        value, _ = search(
            next_state, depth - 1, distances,
            treasure_values, method, stats, alpha, beta, level+1
        )

        if is_max:
            if value > best:
                best = value
                best_target = target

            if method != "minimax":
                alpha = max(alpha, best)

        else:
            if value < best:
                best = value
                best_target = target

            if method != "minimax":
                beta = min(beta, best)

        # Minimax不剪枝，其他三种方法使用Alpha-Beta
        if method != "minimax" and alpha >= beta:
            if target != targets[-1]:
                stats["cutoffs_by_depth"][level] += 1
            break

    return best, best_target


def run_experiment(n, depth, method,
                   distances, treasure_values):
    """运行一次实验，记录搜索节点数和耗时"""
    initial_state = GameState(
        pos_a="A",
        pos_b="B",
        remaining=tuple(treasure_values.keys()),
        score_a=0,
        score_b=0,
        player="A"
    )

    stats = {
    "nodes": 0,
    "leaves": 0,
    "nodes_by_depth": [0] * (depth + 1),
    "cutoffs_by_depth": [0] * (depth + 1),
    "sort_calls": 0,
    "sort_items": 0
    }

    start_time = perf_counter()

    best_value, best_target = search(
        initial_state, depth, distances,
        treasure_values, method, stats
    )

    elapsed_ms = (perf_counter() - start_time) * 1000

    return {
        "target": best_target,
        "value": best_value,
        "nodes": stats["nodes"],
        "leaves": stats["leaves"],
        "time_ms": elapsed_ms,
        "nodes_by_depth": stats["nodes_by_depth"],
        "cutoffs_by_depth": stats["cutoffs_by_depth"],
        "sort_calls": stats["sort_calls"],
        "sort_items": stats["sort_items"]
    }



def test_value_distributions():
    """比较不同价值分布下的搜索效率"""

    scenarios = {
        "original": [9, 6, 7, 7, 5, 8],
        "high_variance": [15, 3, 12, 4, 2, 14],
        "similar_values": [7, 7, 8, 7, 8, 7]
    }

    methods = ["minimax", "left", "right", "dynamic"]
    names = ["T1", "T2", "T3", "T4", "T5", "T6"]

    print("\n===== Value Distribution Experiments =====")

    for scenario_name, config in scenarios.items():
        custom_values = dict(zip(names, config))

        treasure_values, distances = create_scenario(
            6, custom_values
        )

        print(f"\nScenario: {scenario_name}")

        results = {}

        for method in methods:
            result = run_experiment(
                6, 6, method, distances, treasure_values
            )
            results[method] = result

            print(
                f"{method:8s} | "
                f"target={result['target']} | "
                f"value={result['value']:3d} | "
                f"nodes={result['nodes']:5d} | "
                f"leaves={result['leaves']:5d} | "
                f"sort_calls={result['sort_calls']:4d} | "
                f"sort_items={result['sort_items']:4d} | "
                f"time={result['time_ms']:.3f} ms"
            )

        # 验证不同算法的最优收益差一致
        baseline = results["minimax"]["value"]
        assert all(
            r["value"] == baseline
            for r in results.values()
        )


def main():
    methods = ["minimax", "left", "right", "dynamic"]
    
    for n in [4, 6]:
        treasure_values, distances = create_scenario(n)

        depths = [2, 4] if n == 4 else [2, 4, 6]

        for H in depths:
            print(f"\n===== N={n}, H={H} =====")

            results = {}

            for method in methods:
                result = run_experiment(
                    n, H, method, distances, treasure_values
                )
                results[method] = result

                print(
                    f"{method:8s} | "
                    f"target={result['target']} | "
                    f"value={result['value']:3d} | "
                    f"nodes={result['nodes']:5d} | "
                    f"leaves={result['leaves']:5d} | "
                    f"time={result['time_ms']:.3f} ms"
                )

            # 所有算法应得到相同的最优收益差
            baseline = results["minimax"]["value"]
            assert all(
                result["value"] == baseline
                for result in results.values()
            ), "Different optimal values detected!"

    test_value_distributions()

    # N=6, H=4 的反常分析
    # print("\n===== Detailed Analysis: N=6, H=4 =====")

    # treasure_values, distances = create_scenario(6)

    # for method in ["right", "dynamic"]:
    #     result = run_experiment(
    #         6, 4, method, distances, treasure_values
    #     )

    #     print(f"\nMethod: {method}")
    #     print("Nodes by depth:", result["nodes_by_depth"])
    #     print("Cutoffs by depth:", result["cutoffs_by_depth"])
    #     print("Total nodes:", result["nodes"])
    #     print("Total leaves:", result["leaves"])

if __name__ == "__main__":
    main()

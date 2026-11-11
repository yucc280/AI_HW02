
def minimax(node, is_maximizing, order='left'):
    """普通 Minimax，返回 (最佳得分, 访问叶子数)"""
    if not node.children:
        return node.value, 1

    leaf_count = 0

    children = (
    node.children
    if order == "left"
    else reversed(node.children)
    )


    if is_maximizing:
        best = float("-inf")
        for child in children:
            value, count = minimax(child, False, order)
            best = max(best, value)
            leaf_count += count
    else:
        best = float("inf")
        for child in children:
            value, count = minimax(child, True, order)
            best = min(best, value)
            leaf_count += count

    return best, leaf_count


def alpha_beta(node, alpha, beta, is_maximizing, order='left'):
    """Alpha-Beta 剪枝，返回 (最佳得分, 访问叶子数)"""
    if not node.children:
        return node.value, 1

    leaf_count = 0

    children = (
        node.children
        if order == "left"
        else reversed(node.children)
        )

    if is_maximizing:
        best = float("-inf")

        for child in children:
            value, count = alpha_beta(child, alpha, beta, False, order)
            best = max(best, value)
            leaf_count += count
            alpha = max(alpha, best)

            if alpha >= beta:
                break

    else:
        best = float("inf")

        for child in children:
            value, count = alpha_beta(child, alpha, beta, True, order)
            best = min(best, value)
            leaf_count += count
            beta = min(beta, best)

            if alpha >= beta:
                break

    return best, leaf_count


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


def main():
    # 测试树
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])

    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])

    root = TreeNode(children=[min1, min2])

    for order in ["left", "right"]:
        value1, leaves1 = minimax(root, True, order)

        value2, leaves2 = alpha_beta(
            root, float("-inf"), float("inf"), True, order
        )

        print(f"\nOrder: {order}")
        print(f"Minimax: Best value = {value1}, Visited leaves = {leaves1}")
        print(f"Alpha-Beta: Best value = {value2}, Visited leaves = {leaves2}")

    # 自定义测试树：比较不同访问顺序的剪枝效果
    custom_root = TreeNode(children=[
        TreeNode(children=[
            TreeNode(value=8),
            TreeNode(value=9)
        ]),
        TreeNode(children=[
            TreeNode(value=2),
            TreeNode(value=3)
        ])
    ])

    print("\n===== Custom Tree =====")

    for order in ["left", "right"]:
        value1, leaves1 = minimax(custom_root, True, order)

        value2, leaves2 = alpha_beta(
            custom_root, float("-inf"), float("inf"), True, order
        )

        print(f"\nOrder: {order}")
        print(f"Minimax: Best value = {value1}, Visited leaves = {leaves1}")
        print(f"Alpha-Beta: Best value = {value2}, Visited leaves = {leaves2}")


if __name__ == "__main__":
    main()

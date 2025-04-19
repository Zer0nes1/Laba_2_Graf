import random
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import time
from typing import List, Optional, Set, Tuple

class TreeNode:
    """Узел N-дерева"""
    __slots__ = ['data', 'children']
    
    def __init__(self, data: int):
        self.data = data          # Значение узла
        self.children = []        # Список дочерних узлов

class NaryTree:
    """Класс для работы с N-деревьями"""
    
    def __init__(self):
        self.root = None          # Корень дерева
        self._size = 0            # Количество узлов

    def get_size(self) -> int:
        """Возвращает количество узлов в дереве"""
        return self._size

    def create_random_tree(self, num_nodes: int, max_children: int = 3, none_prob: float = 0.2) -> None:
        """
        Создает случайное N-дерево
        :param num_nodes: максимальное количество узлов
        :param max_children: максимальное количество потомков у узла
        :param none_prob: вероятность отсутствия узла
        """
        if num_nodes <= 0:
            raise ValueError("Количество узлов должно быть положительным")
        
        values = [random.randint(1, 1000) if random.random() > none_prob else None 
                 for _ in range(num_nodes)]
        
        if values[0] is None:
            values[0] = random.randint(1, 1000)
        
        self.root = TreeNode(values[0])
        self._size = 1
        nodes_queue = deque([self.root])
        current_index = 1
        
        while nodes_queue and current_index < num_nodes:
            current_node = nodes_queue.popleft()
            
            if current_node is None:
                continue
                
            num_children = random.randint(1, max_children)
            
            for _ in range(num_children):
                if current_index >= num_nodes:
                    break
                    
                if values[current_index] is not None:
                    child_node = TreeNode(values[current_index])
                    current_node.children.append(child_node)
                    nodes_queue.append(child_node)
                    self._size += 1
                else:
                    nodes_queue.append(None)
                
                current_index += 1
        
        self.save_to_file("random_tree.txt")

    def find_subtrees_with_leaf_depths(self, min_depth: int, max_depth: int) -> Tuple[List['NaryTree'], float]:
        """
        Находит все поддеревья, где все листья находятся на глубине 
        (в рёбрах) от корня поддерева в диапазоне [min_depth, max_depth]
        Возвращает кортеж: (список поддеревьев, время выполнения в миллисекундах)
        """
        if self.root is None:
            return [], 0.0
            
        if min_depth < 0 or max_depth < min_depth:
            raise ValueError("Некорректный диапазон глубин")
        
        start_time = time.perf_counter()
        valid_subtrees = []
        queue = deque([self.root])
        
        while queue:
            current_node = queue.popleft()
            
            # Пропускаем листья (они не могут быть корнями поддеревьев)
            if not current_node.children:
                continue
                
            # Проверяем все листья текущего поддерева
            if self._check_leaf_depths(current_node, min_depth, max_depth):
                subtree = NaryTree()
                subtree.root = self._copy_subtree(current_node)
                subtree._size = self._count_nodes(subtree.root)
                valid_subtrees.append(subtree)
            
            # Добавляем детей для проверки их поддеревьев
            for child in current_node.children:
                queue.append(child)
        
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000  # в миллисекундах
        return valid_subtrees, elapsed_time

    def _check_leaf_depths(self, node: TreeNode, min_d: int, max_d: int) -> bool:
        """
        Проверяет, что ВСЕ листья поддерева находятся на глубине 
        от корня поддерева в диапазоне [min_d, max_d] (в рёбрах)
        """
        stack = [(node, 0)]  # (узел, текущая_глубина)
        
        while stack:
            current_node, depth = stack.pop()
            
            if not current_node.children:  # Лист
                if not (min_d <= depth <= max_d):
                    return False
            else:
                for child in current_node.children:
                    stack.append((child, depth + 1))
        
        return True

    def _copy_subtree(self, node: TreeNode) -> Optional[TreeNode]:
        """Глубокое копирование поддерева"""
        if node is None:
            return None
            
        new_node = TreeNode(node.data)
        new_node.children = [self._copy_subtree(child) for child in node.children]
        return new_node

    def _count_nodes(self, node: TreeNode) -> int:
        """Подсчет узлов в поддереве"""
        if node is None:
            return 0
        return 1 + sum(self._count_nodes(child) for child in node.children)

    def visualize(self, title: str = "N-дерево", node_size: int = 800, level_spacing: float = 1.5, sibling_spacing: float = 1.2) -> None:
        """
        Визуализация дерева без наложения узлов.
        
        Параметры:
            title: заголовок графа
            node_size: размер узлов (по умолчанию 800)
            level_spacing: вертикальное расстояние между уровнями (по умолчанию 1.5)
            sibling_spacing: горизонтальное расстояние между узлами одного уровня (по умолчанию 1.2)
        """
        if self.root is None:
            print("Дерево пустое")
            return

        plt.clf()
        G = nx.DiGraph()
        pos = {}
        labels = {}

        # Рекурсивное вычисление позиций с учетом размера поддеревьев
        def _calculate_positions(node, x_offset: float, y: float, spacing: float):
            if node is None:
                return x_offset

            node_id = id(node)
            
            # Вычисляем ширину поддерева (рекурсивно для всех детей)
            child_x = x_offset
            child_widths = []
            for child in node.children:
                child_width = _calculate_positions(child, child_x, y - level_spacing, spacing * 0.9)
                child_widths.append(child_width - child_x)
                child_x = child_width + sibling_spacing

            # Центрируем родителя относительно детей
            total_width = sum(child_widths) + max(0, len(node.children) - 1) * sibling_spacing
            x = x_offset + total_width / 2
            
            pos[node_id] = (x, y)
            labels[node_id] = str(node.data)
            G.add_node(node_id)

            # Добавляем связи с детьми
            for child in node.children:
                G.add_edge(node_id, id(child))

            return x_offset + total_width

        # Вычисляем позиции всех узлов
        _calculate_positions(self.root, 0, 0, sibling_spacing)

        # Автоматическая подгонка размера графа
        if not pos:
            return

        x_values = [p[0] for p in pos.values()]
        y_values = [p[1] for p in pos.values()]
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)

        # Настройка размеров фигуры
        fig_width = max(10, min(20, x_range * 0.5))
        fig_height = max(8, min(15, y_range * 0.7))
        plt.figure(figsize=(fig_width, fig_height))

        # Определение цветов узлов
        node_colors = ['skyblue' if G.out_degree(node) > 0 else 'lightgreen' for node in G.nodes()]

        # Отрисовка графа
        nx.draw(G, pos,
                labels=labels,
                node_size=node_size,
                node_color=node_colors,
                font_size=10,
                font_weight='bold',
                arrows=False,
                edge_color='gray',
                width=1.5,
                alpha=0.8)

        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    def save_to_file(self, filename: str) -> None:
        """Сохранение дерева в файл (префиксный обход)"""
        with open(filename, 'w', encoding='utf-8') as f:
            def _write_node(node):
                if node is None:
                    f.write("None\n")
                    return
                f.write(f"{node.data} {len(node.children)}\n")
                for child in node.children:
                    _write_node(child)
            _write_node(self.root)
        print(f"Дерево сохранено в {filename}")

    def load_from_file(self, filename: str) -> None:
        """Загрузка дерева из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            print(f"Файл {filename} не найден")
            return
            
        def _build_tree(it):
            try:
                line = next(it).strip()
            except StopIteration:
                return None
                
            if line == "None":
                return None
                
            parts = line.split()
            data = int(parts[0])
            num_children = int(parts[1])
            
            node = TreeNode(data)
            node.children = [_build_tree(it) for _ in range(num_children)]
            return node
            
        self.root = _build_tree(iter(lines))
        self._size = self._count_nodes(self.root) if self.root else 0
        print(f"Дерево загружено из {filename} ({self._size} узлов)")

def manual_tree_creation() -> Optional[NaryTree]:
    """Создание дерева вручную"""
    print("\nСоздание дерева вручную (префиксный обход)")
    print("Для каждого узла введите: значение количество_детей")
    print("Например: '5 2' - узел 5 с 2 детьми")
    print("'None' - пустой узел")
    
    def _build_tree():
        try:
            line = input().strip()
            if line.lower() == 'none':
                return None
                
            parts = line.split()
            data = int(parts[0])
            num_children = int(parts[1])
            
            node = TreeNode(data)
            print(f"Узел {data}. Введите {num_children} детей:")
            node.children = [_build_tree() for _ in range(num_children)]
            return node
        except (ValueError, IndexError):
            print("Ошибка ввода! Формат: 'значение количество_детей'")
            return _build_tree()
    
    print("Введите корень дерева:")
    tree = NaryTree()
    tree.root = _build_tree()
    if tree.root:
        tree._size = tree._count_nodes(tree.root)
        print(f"Дерево создано. Узлов: {tree._size}")
        return tree
    return None

def performance_test():
    """Тест производительности для деревьев разного размера"""
    sizes = [100, 1000, 10000, 100000, 1000000]
    min_depth, max_depth = 5, 7
    
    print("\nТест производительности:")
    print(f"Поиск поддеревьев с глубиной листьев [{min_depth}, {max_depth}]")
    print("Размер дерева | Время поиска (мс)")
    print("--------------------------------")
    
    for size in sizes:
        tree = NaryTree()
        tree.create_random_tree(size)
        
        _, exec_time = tree.find_subtrees_with_leaf_depths(min_depth, max_depth)
        print(f"{size:11} | {exec_time:.6f}")

def main():
    """Основная функция с интерфейсом командной строки"""
    current_tree = None
    
    while True:
        print("\nМеню:")
        print("1. Создать случайное дерево")
        print("2. Загрузить дерево из файла")
        print("3. Создать дерево вручную")
        print("4. Визуализировать дерево")
        print("5. Найти поддеревья по глубине листьев")
        print("6. Сохранить дерево в файл")
        print("7. Тест производительности")
        print("8. Выход")
        
        choice = input("Выберите действие: ").strip()
        
        if choice == '1':
            try:
                num_nodes = int(input("Количество узлов: "))
                max_children = int(input("Максимальное число детей у узла: "))
                current_tree = NaryTree()
                current_tree.create_random_tree(num_nodes, max_children)
                print(f"Создано дерево с {current_tree.get_size()} узлами")
            except Exception as e:
                print(f"Ошибка: {e}")
        
        elif choice == '2':
            filename = input("Имя файла: ").strip()
            current_tree = NaryTree()
            current_tree.load_from_file(filename)
        
        elif choice == '3':
            current_tree = manual_tree_creation()
        
        elif choice == '4':
            if current_tree:
                current_tree.visualize()
            else:
                print("Дерево не загружено!")
        
        elif choice == '5':
            if not current_tree:
                print("Дерево не загружено!")
                continue
                
            try:
                min_d = int(input("Минимальная глубина листьев: "))
                max_d = int(input("Максимальная глубина листьев: "))
                
                subtrees, exec_time = current_tree.find_subtrees_with_leaf_depths(min_d, max_d)
                
                # Форматируем время для вывода
                if exec_time < 0.001:  # меньше 1 микросекунды
                    time_str = f"{exec_time * 1000:.3f} наносекунд"
                elif exec_time < 1:    # меньше 1 миллисекунды
                    time_str = f"{exec_time:.3f} микросекунд"
                else:
                    time_str = f"{exec_time:.3f} миллисекунд"
                
                print(f"\nНайдено {len(subtrees)} поддеревьев за {time_str}:")
                
                for i, subtree in enumerate(subtrees, 1):
                    print(f"{i}. Корень: {subtree.root.data}, узлов: {subtree.get_size()}")
                    if input("Показать? (y/n): ").lower() == 'y':
                        subtree.visualize(f"Поддерево {i} (корень {subtree.root.data})")
            except ValueError as e:
                print(f"Ошибка: {e}")
        
        elif choice == '6':
            if current_tree:
                filename = input("Имя файла: ").strip()
                current_tree.save_to_file(filename)
            else:
                print("Дерево не загружено!")
        
        elif choice == '7':
            performance_test()
        
        elif choice == '8':
            print("Выход")
            break
        
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main()
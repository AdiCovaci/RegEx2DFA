import networkx as nx
import matplotlib.pyplot as plt

from regex import RegExTree

LAMBDA = "λ"

class LambdaNFA:
    def __init__(self, graph: nx.DiGraph = None):
        self.graph = graph
        self.sigma = None

        self.initial_state = None
        self.final_state = None

    def parse(self, word, current=None):
        if current is None:
            current = self.initial_state

        if word == '':
            return True if current == self.final_state else False

        char = word[0]

        for _, next_node, att in self.graph.edges(current, data=True):
            if att['char'] == LAMBDA:
                if self.parse(word, next_node):
                    return True
            elif att['char'] == char:
                if self.parse(word[1:], next_node):
                    return True
        
        return False

    def draw(self, path: str):
        plt.figure()

        position = nx.spring_layout(self.graph)
        nx.draw(self.graph, position, with_labels=True)
        nx.draw_networkx_nodes(self.graph, position,
            nodelist=[self.initial_state],
            node_color='g')
        nx.draw_networkx_nodes(self.graph, position,
            nodelist=[self.final_state],
            node_color='r')

        edge_att = nx.get_edge_attributes(self.graph, 'char')
        nx.draw_networkx_edge_labels(self.graph, position, edge_labels=edge_att)

        plt.savefig(path)
        plt.close()

    @staticmethod
    def fromRegExTree(tree):
        nfa = LambdaNFA()
        nfa.sigma = tree.sigma
        nfa.graph, (nfa.initial_state, nfa.final_state) = \
            LambdaNFA.fromRegExNode(tree.root)
        return nfa
    
    @staticmethod
    def fromRegExNode(node, prefix='0'):
        if isinstance(node, str):
            graph = nx.DiGraph()
            first = f'{prefix}_0'
            last = f'{prefix}_1'
            graph.add_edge(first, last, char=node)
        elif node.op == 'C':
            left_graph, (left_first, left_last) = LambdaNFA.fromRegExNode(node.left, prefix+'<')
            right_graph, (right_first, right_last) = LambdaNFA.fromRegExNode(node.right, prefix+'>')

            graph = nx.DiGraph()

            graph.add_nodes_from(left_graph)
            graph.add_nodes_from(right_graph)
            graph.add_edges_from(left_graph.edges(data=True))
            graph.add_edges_from(right_graph.edges(data=True))

            graph.add_edge(left_last, right_first, char=LAMBDA)

            first = left_first
            last = right_last
        elif node.op == '*':
            left_graph, (left_first, left_last) = LambdaNFA.fromRegExNode(node.left, prefix+'*')

            graph = nx.DiGraph()

            graph.add_nodes_from(left_graph)
            graph.add_edges_from(left_graph.edges(data=True))

            graph.add_edge(f'{prefix}_0', left_first, char=LAMBDA)
            graph.add_edge(f'{prefix}_1', left_first, char=LAMBDA)
            graph.add_edge(f'{prefix}_0', left_last, char=LAMBDA)
            graph.add_edge(left_last, f'{prefix}_1', char=LAMBDA)

            first = f'{prefix}_0'
            last = f'{prefix}_1'
        elif node.op == '|':
            left_graph, (left_first, left_last) = LambdaNFA.fromRegExNode(node.left, prefix+'<|')
            right_graph, (right_first, right_last) = LambdaNFA.fromRegExNode(node.right, prefix+'|>')

            graph = nx.DiGraph()

            graph.add_nodes_from(left_graph)
            graph.add_nodes_from(right_graph)
            graph.add_edges_from(left_graph.edges(data=True))
            graph.add_edges_from(right_graph.edges(data=True))

            graph.add_edge(f'{prefix}_0', left_first, char=LAMBDA)
            graph.add_edge(f'{prefix}_0', right_first, char=LAMBDA)
            graph.add_edge(left_last, f'{prefix}_1', char=LAMBDA)
            graph.add_edge(right_last, f'{prefix}_1', char=LAMBDA)

            first = f'{prefix}_0'
            last = f'{prefix}_1'

        
        return graph, (first, last)


class DFA:
    def __init__(self, lnfa):
        self.lnfa = lnfa
        self.lambda_closures = dict()
        self.get_lambda_closures()
        self.initial_state = None
        self.final_states = []

    def get_lambda_closures(self):
        for node in self.lnfa.graph.nodes:
            self.lambda_closures[node] = self._get_lambda_closure(node)

    def _get_lambda_closure(self, node, visited = None):
        if visited is None:
            visited = set()

        if node in visited:
            return set()

        print()
        print(f'{node=}')
        print(f'{visited=}')
        
        visited.add(node)

        closure = set([node])

        for _, next_node, att in self.lnfa.graph.edges(node, data=True):
            if att['char'] == LAMBDA:
                next_closure = self._get_lambda_closure(next_node, visited)
                print(next_closure)
                closure = closure.union(next_closure)

        return closure

    def build_transition_matrix(self):
        self.transition = dict()

        known_states = [self.lambda_closures[self.lnfa.initial_state]]
        state_queue = [self.lambda_closures[self.lnfa.initial_state]]

        self.initial_state = self.lambda_closures[self.lnfa.initial_state]
        
        while len(state_queue) > 0:
            current_node = state_queue.pop()
            if self.lnfa.final_state in current_node:
                self.final_states.append(current_node)
            reachables = list()
            for char in self.lnfa.sigma:
                reachable = self.get_reachable_states(current_node, char)
                reachables.append(reachable)
                if len(reachable) > 0 and reachable not in known_states:
                    known_states.append(reachable)
                    state_queue.append(reachable)

            self.transition[str(current_node)] = reachables

    def get_reachable_states(self, node, char):
        reachable = set()

        for n in node:
            for _, next_node, att in self.lnfa.graph.edges(node, data=True):
                if att['char'] == char:
                    reachable = reachable.union(self.lambda_closures[next_node])

        return reachable

    def build_graph(self):
        self.graph = nx.DiGraph()

        for node in self.transition.keys():
            self.graph.add_node(node)
        
        for i_node, nodes in self.transition.items():
            for char, j_node in enumerate(nodes):
                if len(j_node) == 0:
                    continue
                self.graph.add_edge(i_node, str(j_node), char=self.lnfa.sigma[char])

    def draw(self, path: str):
        plt.figure()

        position = nx.spring_layout(self.graph)
        nx.draw(self.graph, position, with_labels=False, alpha=0.5)
        nx.draw_networkx_nodes(self.graph, position,
            nodelist=[str(self.initial_state)],
            alpha=0.5,
            node_color='g')
        nx.draw_networkx_nodes(self.graph, position,
            nodelist=[str(x) for x in self.final_states],
            alpha=0.5,
            node_color='r')

        edge_att = nx.get_edge_attributes(self.graph, 'char')
        nx.draw_networkx_edge_labels(self.graph, position, edge_labels=edge_att)

        plt.savefig(path)
        plt.close()

    def parse(self, word, current=None):
        if current is None:
            current = str(self.initial_state)

        if word == '':
            return True if current in [str(x) for x in self.final_states] else False

        char = word[0]

        for _, next_node, att in self.graph.edges(current, data=True):
            if att['char'] == LAMBDA:
                if self.parse(word, next_node):
                    return True
            elif att['char'] == char:
                if self.parse(word[1:], next_node):
                    return True
        
        return False
        
with open('regex.txt', 'r') as f:
    regex = f.readline().strip()

# Create RegEx Parse Tree
tree = RegExTree(regex)

# Create LambdaNFA
lnfa = LambdaNFA.fromRegExTree(tree)
lnfa.draw('lambdaNFA.png')

dfa = DFA(lnfa)
dfa.build_transition_matrix()
dfa.build_graph()
dfa.draw('DFA.png')

with open('output.txt', 'w+') as o:
        with open('words.txt', 'r') as f:
            words = []
            while line := f.readline().strip():
                words.append(line)
        
        for word in words:
            print(f'{word} | {"LNFA OK" if lnfa.parse(word) else "LNFA NO"} | {"DFA OK" if dfa.parse(word) else "DFA NO"}', file=o)
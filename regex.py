class RegExNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.left = 'NONE'
        self.right = 'NONE'
        self.op = 'NONE'

    def __str__(self):
        s = self.op + ' {\n'
        s += str(self.left) + '\n'
        s += str(self.right) +'\n'
        s += '}'

        return s

class RegExTree:
    def __init__(self, regex):
        self.sigma = list(set(regex).difference(set('()|*')))
        self.root, regex_left = self.nodeFromRegex(regex)

        while regex_left is not None:
            new_root = RegExNode()
            new_root.left = self.root
            new_root.op = '|'
            if not isinstance(self.root, str):
                self.root.parent = new_root
            self.root = new_root
            self.root.right, regex_left = self.nodeFromRegex(regex_left)

    def nodeFromRegex(self, regex, parent=None):
        node = RegExNode(parent)
        print(regex)

        if len(regex) == 0:
            return None, None
        if len(regex) == 1:
            if regex != '*':
                return regex, None
            else:
                node.parent.op = '*'
                return None, None
        if regex[1] == '|':
            if regex[0] != '*':
                return regex[0], regex[2:]
            else:
                node.parent.op = '*'
                return None, regex[2:]

        if regex[0] not in '(|*':
            node.left = regex[0]
            node.op = 'C'
            node.right, regex_left = self.nodeFromRegex(regex[1:], node)
        elif regex[0] == '*':
            new_node = RegExNode(node.parent)
            new_node.left = node.parent.left
            new_node.op = '*'
            node.parent.left = new_node
            node, regex_left = self.nodeFromRegex(regex[1:], node)
        elif regex[0] == '(':
            level = 0
            for i, c in enumerate(regex):
                if c == '(':
                    level += 1
                elif c == ')':
                    if level == 1:
                        break
                    else:
                        level -= 1
            
            print(i)
            node.left = RegExTree(regex[1:i]).root
            node.left.parent = node
            node.op = 'C'
            node.right, regex_left = self.nodeFromRegex(regex[i + 1:], node)
            
        return node, regex_left

# n = RegExTree('ab(c|d)*')
# print()
# print(n.root)
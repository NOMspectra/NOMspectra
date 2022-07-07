from typing import Optional, Mapping, Union
from collections import Counter


class Node(object):
    def __init__(
            self,
            element: str = "",
            amount: int = 1,
            next: Optional['Node'] = None,
            prev: Optional['Node'] = None,
            child: Optional['Node'] = None,
            parent: Optional['Node'] = None,
            isotope: Optional['Node'] = None
    ):
        self.element = element
        self.next = next
        self.isotope = isotope
        self.amount = amount
        self.prev = prev
        self.child = child
        self.parent = parent

    def __repr__(self):
        if self.element == "NEW_END":
            return f"({''.join([x.__repr__() for x in self.child.get_chain()])}){self.amount}"

        return f"{self.element}{self.amount}"

    def get_chain(self):
        nodes = []
        node = self
        while node is not None:
            nodes.append(node)
            node = node.next

        return nodes

    def __str__(self):
        return str(self.__repr__())

    @staticmethod
    def copy():
        pass

    def dfs(self):
        v = self

        if v.child:
            print("(", end="")
            v.child.dfs()
            print(")", end="")
            if v.amount != 1:
                print(v.amount, end="")
        else:
            print(v.element, end="")
            if v.amount != 1:
                print(v.amount, end="")
        if v.next:
            v.next.dfs()

    def get_dict(self, coefficient=1) -> Mapping[str, int]:
        c = Counter()
        v = self

        if v.child:
            c += v.child.get_dict(coefficient=coefficient * v.amount)
        else:
            c[self.element] += self.amount * coefficient

        if v.next:
            c += v.next.get_dict(coefficient)

        return c


def read(s: str, i: int) -> [str, Union[str, int], int]:
    if i == len(s):
        return "EOF", "EOF", i

    if s[i] == "(":
        return "new", "NEW", i + 1

    if s[i] == ")":
        return "end", "END", i + 1

    if s[i].isdigit():
        j = i + 1
        while j <= len(s) and s[i:j].isdigit():
            j += 1

        j -= 1
        return "number", int(s[i:j]), j

    if s[i].isalpha() and s[i].isupper():
        j = i + 2
        while j <= len(s) and s[i:j].isalpha() and s[i+1:j].islower():
            j += 1

        j -= 1
        return "element", s[i:j], j


def brutto_iterator(brutto: str):
    index = 0
    while index < len(brutto):
        status, value, index = read(brutto, index)  # unknown
        yield status, value


# without automaton programming
def build_brutto_tree(brutto: str) -> Node:

    current = Node()
    start = current
    for status, value in brutto_iterator(brutto):
        if status == "element":
            prev = current
            current = Node(element=value)

            if prev.element == "NEW":
                current.parent = prev
                prev.child = current

            else:
                prev.next = current
                current.prev = prev
                current.parent = prev.parent

        if status == "number":
            current.amount = value

        if status == "new":
            prev = current
            current = Node(element="NEW")

            if prev.element == "NEW":
                current.parent = prev
                prev.child = current

            else:
                prev.next = current
                current.prev = prev
                current.parent = prev.parent

        if status == "end":
            if current.parent is None:
                raise Exception()

            current = current.parent
            current.element = "NEW_END"

    start = start.next
    start.prev = None
    return start


# automaton programming
def bind_nodes(prev: Node, next: Node):
    prev.next = next
    next.prev = prev


def build_brutto_tree_2(brutto: str) -> Node:
    raise NotImplementedError


if __name__ == '__main__':

    brutto = "((((((((((CH2)2)2)2)2)2)2)2)2)2)2"
    # print(parse_brutto(brutto))
    print(brutto)
    node = build_brutto_tree(brutto)
    print(node.get_dict())

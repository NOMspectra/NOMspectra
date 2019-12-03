from typing import Optional, Mapping, Union


class Node(object):
    def __init__(
            self,
            element: str = "",
            amount: int = 1,
            next: Optional['Node'] = None,
            prev: Optional['Node'] = None,
            child: Optional['Node'] = None,
            parent: Optional['Node'] = None
    ):
        self.element = element
        self.next = next
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
        if v is None:
            return

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


# wihtout automaton programming
def build_brutto_tree(brutto: str) -> Node:

    current = Node()
    begin = current
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

            prev.next = current
            current.prev = prev
            current.parent = prev.parent

        if status == "end":
            if current.parent is None:
                raise Exception()

            current = current.parent
            current.element = "NEW_END"

    return begin.next


if __name__ == '__main__':
    brutto = "A2BCu(C2(CN)4)F3"
    # print(parse_brutto(brutto))
    print(brutto)
    node = build_brutto_tree(brutto)
    # print(node.next.__repr__())
    # print(node.next.next.__repr__())
    # print(node.next.next.next.__repr__())
    # print(node.next.next.next.next.__repr__())

    print(node.dfs())

    # Node.dfs(node)


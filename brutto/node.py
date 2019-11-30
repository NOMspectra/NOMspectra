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
        return self.element, self.amount

    def __str__(self):
        return str(self.__repr__())

    @staticmethod
    def copy():
        pass

    @staticmethod
    def dfs():
        pass


def read(s: str, i: int) -> [str, Union[str, int], int]:
    if i == len(s):
        return "EOF", "EOF", i

    if s[i] == "(":
        return "new", "(", i + 1

    if s[i] == ")":
        return "end", ")", i + 1

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

        if status == "(":
            prev = current
            current = Node(element="NEW")

            prev.next = current
            current.prev = prev

        if status == ")":
            current = current.parent

    return begin


if __name__ == '__main__':
    brutto = "A2BCu(C2(CN)4)F3"
    # print(parse_brutto(brutto))

    node = build_brutto_tree(brutto)

    print(node)
    print(node.next)
    print(node.next.next)
    print(node.next.next.next)
    print(node.next.next.next.next)



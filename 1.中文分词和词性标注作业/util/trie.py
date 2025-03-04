class Node:
    def __init__(self, c):
        self.data = c
        self.next = dict()
        self.is_leaf = False

    def insert(self, c):
        self.next[c] = Node(c)


class Trie:
    def __init__(self, reverse=False):
        self.root = Node("")
        self.reverse = reverse
        self.size = 0

    def __len__(self):
        return self.size

    def insert(self, word):
        if self.reverse:
            word = word[::-1]
        current = self.root
        not_inserted = False
        for c in word:
            if c not in current.next:
                not_inserted = True
                current.insert(c)
            current = current.next[c]
        if not_inserted:
            self.size += 1
        current.is_leaf = True

    def search(self, text):
        # if self.reverse:
        #     text = text[::-1]
        result = []  # [(span1, span2), ...]
        i = 0
        cur = self.root
        while i < len(text):
            start_tmp = i
            end_tmp = None
            while i < len(text) and text[i] in cur.next:
                cur = cur.next[text[i]]
                i += 1
                if cur.is_leaf:
                    end_tmp = i
            if end_tmp is None:
                end_tmp = start_tmp + 1
            i = end_tmp
            if self.reverse:
                result.append((len(text) - end_tmp, len(text) - start_tmp))
            else:
                result.append((start_tmp, end_tmp))
            cur = self.root
        return result


# %%
if __name__ == "__main__":
    text = "迈向 充满 希望 的 新 世纪 —— 一九九八年 新年 讲话 （ 附 图片 １ 张 ）"
    dic = text.split(" ")
    print(dic)

    i = 0
    labels = []
    for w in dic:
        labels.append((i, i + len(w)))
        i += len(w)

    trie = Trie()
    for w in dic:
        trie.insert(w)
    res = trie.search(text.replace(" ", ""))

    print(res)

    print(set(labels).difference(set(res)))
# %%

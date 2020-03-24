---
layout: post
title: "Daily Python Snippet 3"
comments: true
---
*(De)Serializing binary trees is not special to python but I find it interesting.*

## Binary Tree
```python
class TreeNode(object):
    def __init__(self, val: int):
        '''
        Definition for a binary tree node:
         1
        / \
       2   3
          / \
         4   5
        '''
        self.val = val
        self.left = None
        self.right = None
```

## Serializing
```python
# Serialize recursively using pre-order traversal.
def serialize(root: TreeNode, delimiter: str=',', void: str='X') -> str:
    if not root:
        return void
    left = serialize(root.left, delimiter, void)
    right = serialize(root.right, delimiter, void)
    return '{:d}{:s}{:s}{:s}{:s}'.format(root.val, delimiter, left, delimiter, right)
```

## Deserializing
```python
from collections import deque

def deserialize(data: str, delimiter: str=',', void: str='X') -> TreeNode:
    # Deserialize helper using a queue.
    # Queues can be used to perform pre-order traversal in an iterative manner.
    def helper(queue):
        if not queue:
            return None
        val = queue.popleft()
        if val is not None:
            root = TreeNode(val)
            root.left, root.right = helper(queue), helper(queue)
            return root
        else:
            return None

    queue = deque([int(d) if d != void else None for d in data.split(delimiter)])
    return helper(queue)
```

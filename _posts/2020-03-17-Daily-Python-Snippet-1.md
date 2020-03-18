---
layout: post
title: "Daily Python Snippet 1"
comments: true
---
*I find these Python snippets more pythonic than their alternative.*

## Counting Elements in a List
Don't:
```python
x = ['a', 'a', 'b', 'c', 'd', 'c', 'a']
count = {}
for e in x:
    if e not in count.keys():
        count[e] = 0
    else:
        count[e] += 1
```

Do:
```python
x = ['a', 'a', 'b', 'c', 'd', 'c', 'a']
count = {}
for e in x:
    count[e] = count.get(e, 0) + 1
```

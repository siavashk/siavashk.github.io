---
layout: post
title: "Daily Python Snippet 2"
comments: true
---

## K Most Frequent Elements from a List of N Elements

```python
import heapq


# Five "a", one "b", three "c", three "d" and two "e"
x = ["e", "a", "d", "a", "b", "c", "a", "a", "d", "d", "c", "a", "e", "c"]


# Count the occurrence of each element in x.
# See Daily Python Snippet 1:
# http://siavashk.github.io/2020/03/17/Daily-Python-Snippet-1/
frequency = {}
for element in x:
    frequency[element] = frequency.get(element, 0) + 1


k = 2  # return top 2 items
heap = []  # Use a min-heap to discard the least frequent item
for element, count in frequency.items():
    if len(heap) < k:
        heapq.heappush(heap, (count, element))
    else:
        heapq.heappushpop(heap, (count, element))


# reverse the min-heap to sort by decreasing frequency
top_k_elements = [element for _, element in heap[::-1]]
```

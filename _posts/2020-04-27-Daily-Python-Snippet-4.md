---
layout: post
title: "Daily Python Snippet 4"
comments: true
---

## Function Caching
You can use the `lru_cache` decorator to memoize the return value of a function given its arguments. This is useful for computationally expensive and/or IO bound functions:

```python
from functools import lru_cache  # since python 3.2

@lru_cache(maxsize=None)  # cache all values
def fibonacci(n: int) -> int:
    if n < 2:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
```

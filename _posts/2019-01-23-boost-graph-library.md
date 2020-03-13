---
layout: post
title: "Dijkstra's Algorithm in BGL"
comments: true
---
## Introduction
This is hopefully the first in a series of posts about algorithms in the [Boost Graph Library](https://www.boost.org/doc/libs/1_69_0/libs/graph/doc/index.html) (BGL). I think boost is one of the most useful pieces of software ever written, but its documentation is not that great. I struggled a bit while trying out some graph-based algorithms and decided to document what I have learnt. Hopefully others will find this useful as well.

*If you are just looking for examples on how to use BGL, look at this [git](https://github.com/siavashk/lignum-vitae).*

## Graph Theory
If you are already familiar with graph theory, you can skip this section. Let's start things with an example. Assume that we have five cities, denoted by \\(\left\\{a, b, c, d, e\right\\}\\), and that we have roads connecting them. Now these roads could be one-way similar as shown in Figure 1 or traffic might be flowing in both directions as shown in Figure 2:

<figure>
<img alt="Figure 1: Directed Graph" src="/assets/djikstra/digraph.svg" style="width:100%">
<figcaption>Figure 1: Cities connected by one-way roads. Mathematically this is equivalent to a directed graph, where vertices correspond to cities and edges represent roads.</figcaption>
</figure>

<figure>
<img alt="Figure 2: Undirected Graph" src="/assets/djikstra/undigraph.svg" style="width:100%">
<figcaption>Figure 2: Cities connected using undirected roads. This is equivalent to an undirected graph.</figcaption>
</figure>

Formally, we call this representation of cities and roads a graph \\(G\left\(V, E\right\)\\), where vertices \\(V = \left\\{a, b, c, d, e\right\\}\\) correspond to cities and edges \\(E = \left\\{\left\(a, c\right\), \left\(c, b\right\), \left\(c, d\right\), \left\(c, b\right\), \left\(c, d\right\), \left\(b, d\right\), \left\(b, e\right\), \left\(d, e\right\), \left\(e, a\right\)\right\\}\\) represents roads. Depending on edges being one-way or two-way, the graph is called directed or undirected.

Edges in graphs can also have values associated with them as well. In this case the graph is said to be weighted and these values are referred to as edge weights or simply weights. Figure 3 is an example of a weighted graph, where weights denote the distance between cities in miles.

<figure>
<img alt="Figure 3: Weighted Graph" src="/assets/djikstra/wdigraph.svg" style="width:100%">
<figcaption>Figure 3: Example of a weighted graph where weights represent distances between cities in miles.</figcaption>
</figure>

## Graph Representation
There are two popular ways of representing graphs: 1) adjacency lists; and 2) adjacency matrices. An adjacency list is a collection of unordered lists, where each list describes the set of neighbours of a vertex in the graph. This representation is suitable for storing sparsely connected graphs.

An adjacency matrix is a square matrix, where the \\(\left\(i, j\right\)\\)-th element is one if there is an edge connecting the \\(i\\)-th vertex to the \\(j\\)-th vertex and zero otherwise. This representation is suitable if the graph is densely connected.

## Graph Representation in BGL
BGL supports both adjacency lists and matrices for representing graphs. Throughout this example we are going to use adjacency lists, but we could have used adjacency matrices as well. Let's first start by defining a vertex type:

```cpp
using namespace boost;
using VertexPropertyType = property<vertex_name_t, std::string, property<vertex_color_t, default_color_type>>;
```

This vertex type allows us to assign a string and a colour to our vertices so we can, well, name and colour our vertices. Similarly, we can also define an edge type:

```cpp
using EdgePropertyType = property<edge_weight_t, int, property<edge_color_t, default_color_type>>;
```

This allows us to assign integer weights to our edges for defining distances and colouring. Now that we have an edge and vertex type, we can define a directed graph as an adjacency list:

```cpp
using DirectedGraphType = adjacency_list<vecS, vecS, directedS, VertexPropertyType, EdgePropertyType>;
```

In BGL, vertices and edges are manipulated through opaque handlers called **vertex descriptors** and **edge descriptors**. These descriptors are always  accessible through the **graph traits** class:

```cpp
using VertexDescriptor = graph_traits<DirectedGraphType>::vertex_descriptor;
```

With these definitions, we are ready to create our first directed graph. For example, this code snippet can be used to create the directed graph in Figure 3:

```cpp
DirectedGraphType makeDirectedGraphWithCycles()
{
    DirectedGraphType g;

    VertexDescriptor a = add_vertex(VertexPropertyType("a", white_color), g);
    VertexDescriptor b = add_vertex(VertexPropertyType("b", white_color), g);
    VertexDescriptor c = add_vertex(VertexPropertyType("c", white_color), g);
    VertexDescriptor d = add_vertex(VertexPropertyType("d", white_color), g);
    VertexDescriptor e = add_vertex(VertexPropertyType("e", white_color), g);

    add_edge(a, c, EdgePropertyType(1, black_color), g);
    add_edge(b, d, EdgePropertyType(1, black_color), g);
    add_edge(b, e, EdgePropertyType(2, black_color), g);
    add_edge(c, b, EdgePropertyType(5, black_color), g);
    add_edge(c, d, EdgePropertyType(10, black_color), g);
    add_edge(d, e, EdgePropertyType(4, black_color), g);
    add_edge(e, a, EdgePropertyType(3, black_color), g);
    add_edge(e, b, EdgePropertyType(7, black_color), g);

    return g;
}
```

## Finding Shortest Paths in Directed Graphs
Assume that we are interested in traveling from city **a** to **d** and are given distances in the form of Figure 3. As you probably already know, we can use Dijkstra's algorithm to find the shortest path between these two cities. If you are not familiar with this algorithm, I suggest reading through [Erik's](http://vasir.net/blog/game_development/dijkstras_algorithm_shortest_path) excellent blog post.

BGL has a very efficient implementation of Dijkstra's algorithm. Using the above definitions, we can wrap the algorithm using the following functions:

```cpp
std::vector<VertexDescriptor> djikstra(
    const DirectedGraphType& graph,
    const VertexDescriptor* source,
    const VertexDescriptor* destination
) {
    const int numVertices = num_vertices(graph);
    std::vector<int> distances(numVertices);
    std::vector<VertexDescriptor> pMap(numVertices);

    auto distanceMap = predecessor_map(
        make_iterator_property_map(pMap.begin(), get(vertex_index, graph))).distance_map(
        make_iterator_property_map(distances.begin(), get(vertex_index, graph)));

    dijkstra_shortest_paths(graph, source, distanceMap);
    return getPath(graph, pMap, source, destination);
}
```

The **predecessor_map** starts from the destination and works backwards towards the source. **getPath** is simply a utility function for reversing this path so that it starts from the source to the destination:

```cpp
std::vector<VertexDescriptor> getPath(
    const DirectedGraphType& graph,
    const std::vector<VertexDescriptor>& pMap,
    const VertexDescriptor& source,
    const VertexDescriptor& destination
) {
    std::vector<VertexDescriptor> path;
    VertexDescriptor current = destination;
    while (current != source)
    {
        path.push_back(current);
        current = pMap[current];
    }
    path.push_back(source);
    return path;
}
```

Since we defined our **EdgePropertyType** to have colours, we can highlight the shortest path to red in order to check if the result is correct. This can be done using the following snippet:

```cpp
DirectedGraphType markPathAlongGraph(
    const DirectedGraphType& graph,
    const std::vector<VertexDescriptor>& path,
    const default_color_type nodeColor,
    const default_color_type edgeColor
) {
    DirectedGraphType marked;
    copy_graph(graph, marked);

    auto nodeColorMap = get(vertex_color, marked);
    nodeColorMap[path.front()] = nodeColor;
    nodeColorMap[path.back()] = nodeColor;

    auto nodeIndexMap = get(vertex_index, marked);
    for (auto first = path.rbegin(); first < path.rend() - 1; ++first)
    {
        auto second = next(first);
        VertexDescriptor from = nodeIndexMap[*first];
        VertexDescriptor to = nodeIndexMap[*second];
        auto edge = edge(from, to, marked).first;
        put(edge_color, marked, edge, edgeColor);
    }
    return marked;
}
```

Writing the marked graph using [Graphviz](https://www.graphviz.org/) allows us to visualize the shortest path along the graph from **a** to **d** as seen Figure 4:

<figure>
<img alt="Figure 4: Dijkstra's Algorithm" src="/assets/djikstra/djikstra.svg" style="width:100%">
<figcaption>Figure 4: Red edges denote the shortest path between <b>a</b> and <b>d</b> using Dijkstra's algorithm.</figcaption>
</figure>

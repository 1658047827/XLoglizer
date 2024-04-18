<template>
    <div id="diagram-container"></div>
    <div id="node-info"></div>
</template>

<script setup>
import axios from "axios";
import * as d3 from "d3";
import { onMounted } from "vue";

const width = 1080
const height = 720

const colorScale = d3.scaleLinear()
    .domain([0, 1])
    .range(['lightblue', 'darkblue']);

function loopArc(d) {
    const r = d.target.size;
    return `M ${d.target.x - r} ${d.target.y}
            A ${r} ${r} 1 1 1 ${d.target.x} ${d.target.y - r}`;
}

function linkArc(d) {
    const r = Math.hypot(d.target.x - d.source.x, d.target.y - d.source.y);
    const offsetX = ((d.target.x - d.source.x) * d.target.size) / r;
    const offsetY = ((d.target.y - d.source.y) * d.target.size) / r;
    return `M ${d.source.x} ${d.source.y}
            A ${r} ${r} 0 0 1 ${d.target.x - offsetX} ${d.target.y - offsetY}`;
}

function drag(simulation) {

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
}

onMounted(async () => {
    const response = await axios.get("/src/assets/force_kmeans_64_39.json");
    const graph = response.data;
    const selfLoops = graph.links.filter((el) => el.source === el.target)
    const edgeLinks = graph.links.filter((el) => el.source !== el.target)

    const simulation = d3.forceSimulation(graph.nodes)
        .force("link", d3.forceLink(graph.links).id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("x", d3.forceX())
        .force("y", d3.forceY())

    const svg = d3.select("#diagram-container")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto; font: 12px sans-serif;");

    svg.append("defs").selectAll("marker")
        .data(graph.links)
        .join("marker")
        .attr("id", d => `end-arrow-${d.source.id}-${d.target.id}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 10)
        .attr("markerWidth", 8.5)
        .attr("markerHeight", 8.5)
        .attr("orient", "auto")
        .attr("markerUnits", "userSpaceOnUse")
        .append("path")
        .attr("fill", d => colorScale(d.weight))
        .attr("d", "M0,-5L10,0L0,5");

    const loop = svg.append("g")
        .attr("fill", "none")
        .selectAll("path")
        .data(selfLoops)
        .join("path")
        .attr("stroke", d => colorScale(d.weight))
        .attr("stroke-width", d => d.weight * 3)
        .attr("marker-end", d => `url(#end-arrow-${d.source.id}-${d.target.id})`)

    const link = svg.append("g")
        .attr("fill", "none")
        .selectAll("path")
        .data(edgeLinks)
        .join("path")
        .attr("stroke", d => colorScale(d.weight))
        .attr("stroke-width", d => d.weight * 3.5)
        .attr("marker-end", d => `url(#end-arrow-${d.source.id}-${d.target.id})`)

    const node = svg.append("g")
        .attr("stroke-linecap", "round")
        .attr("stroke-linejoin", "round")
        .selectAll("g")
        .data(graph.nodes)
        .join("g")
        .call(drag(simulation));

    node.append("circle")
        .attr("cursor", "pointer")
        .attr("stroke", "white")
        .attr("stroke-width", 1.5)
        .attr("r", d => d.size)
        .attr("fill", "#ff3399");

    node.append("title")
        .text(d => `S${d.id}`);

    node.append("text")
        .attr("x", d => d.size)
        .attr("y", "0.31em")
        .text(d => `S${d.id}`)
        .clone(true).lower()
        .attr("fill", "none")
        .attr("stroke", "white")
        .attr("stroke-width", 3);

    simulation.on("tick", () => {
        node.attr("transform", d => `translate(${d.x},${d.y})`);
        loop.attr("d", loopArc)
        link.attr("d", linkArc);
    });
})

</script>

<style scoped></style>
<template>
    <div id="diagram-container"></div>
    <div id="node-info"></div>
</template>

<script setup>
import data from "../assets/force0.json"
import * as d3 from "d3";
import { onMounted } from "vue";

const width = 1080
const height = 720

onMounted(() => {
    const svg = d3.select("#diagram-container")
        .append("svg")
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('width', width)
        .attr('height', height);

    const simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(data.links)
        .enter()
        .append("line")
        .attr("fill", "none")
        .attr("stroke-width", d => d.width) // edge width
        .attr("stroke", "gray")

    const node = svg.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(data.nodes)
        .enter().append("circle")
        .attr("r", d => d.size)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("title")
        .text(d => d.id);

    simulation.nodes(data.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(data.links);

    function ticked() {
        link.attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node.attr("cx", d => d.x)
            .attr("cy", d => d.y);
    }

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
})

</script>

<style>
.nodes circle {
    cursor: pointer;
    fill: #ff3399;
    stroke: #000;
    stroke-width: .5px;
}
</style>